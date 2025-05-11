from collections.abc import Iterable

import torch as th
from torch_geometric.utils import mask_to_index
from torch_geometric.utils.map import map_index

from research.data.base import MainData, SnapshotInfo, SubData
from research.utils import edge_subgraph


class SnapshotManager:
    def __init__(self, data: MainData, device="cuda") -> None:
        self.main_data = data
        self.device = device

        self.curr_snapshot_id = None
        self.curr_cuda_data = None

        self.snapshots: dict[int, SnapshotInfo] = {}

    def get_generator(
        self, snapshot_ids: Iterable[int] | None = None, reset_after_fin: bool = True
    ):
        if snapshot_ids is None:
            snapshot_ids = self.snapshots.keys()
        else:
            assert all(id in self.snapshots for id in snapshot_ids)

        for i in snapshot_ids:
            yield i, self.load_snapshot(i)

        if reset_after_fin:
            self.reset_state()

    def register_snapshot(self, snapshot_id: int, nmask: th.Tensor, emask: th.Tensor):
        assert nmask.dtype == th.bool and emask.dtype == th.bool
        nid = mask_to_index(nmask)
        eid = mask_to_index(emask)
        self.snapshots[snapshot_id] = SnapshotInfo(nmask, emask, nid, eid)

    def load_snapshot(self, snapshot_id):
        assert snapshot_id in self.snapshots

        target_info = self.snapshots[snapshot_id]

        # If it is first time to load snapshot
        if self.curr_snapshot_id is None or self.curr_cuda_data is None:
            snapshot_data = edge_subgraph(self.main_data, target_info.eid)
            self.curr_snapshot_id = snapshot_id
            self.curr_cuda_data = snapshot_data.to(self.device, non_blocking=True)
            return self.curr_cuda_data

        # Found the difference between current and target snapshot
        curr_info = self.snapshots[self.curr_snapshot_id]

        nmask_add = target_info.nmask & ~curr_info.nmask
        emask_add = target_info.emask & ~curr_info.emask
        nmask_rm = ~target_info.nmask & curr_info.nmask
        emask_rm = ~target_info.emask & curr_info.emask

        self._incremental_update(nmask_add, emask_add, nmask_rm, emask_rm)

        self.curr_snapshot_id = snapshot_id
        return self.curr_cuda_data

    def _incremental_update(
        self,
        nmask_add: th.Tensor,
        emask_add: th.Tensor,
        nmask_rm: th.Tensor,
        emask_rm: th.Tensor,
    ):
        """

        Args:
        - nodes_add: Mask of nodes to add
        - edges_add: Mask of edges to add
        - nodes_rm: Mask of nodes to remove
        - edges_rm: Mask of edges to remove
        """
        assert self.curr_cuda_data is not None

        if nmask_rm.any() or emask_rm.any():
            self._remove_old(nmask_rm, emask_rm)

        if nmask_add.any() or emask_add.any():
            nid_add = mask_to_index(nmask_add)
            eid_add = mask_to_index(emask_add)

            self._merge_new(nid_add, eid_add)

    def _merge_new(self, new_nid: th.Tensor, new_eid: th.Tensor):
        """

        Args:
        - new_nid: Original NID of new nodes
        - new_eid: Orignial EID of new edges
        """
        assert self.curr_cuda_data is not None

        new_x = self.main_data.x[new_nid].to(self.device, non_blocking=True)
        new_edge_index = self.main_data.edge_index[:, new_eid].to(
            self.device, non_blocking=True
        )
        new_nid = new_nid.to(self.device, non_blocking=True)
        new_eid = new_eid.to(self.device, non_blocking=True)

        # Reverse src, dst of edges to orig nid
        rev_curr_edge_index = self.curr_cuda_data.gnid[self.curr_cuda_data.edge_index]

        # Combine current snapshot and new part (to orig nid)
        nid = th.cat((self.curr_cuda_data.gnid, new_nid))
        eid = th.cat((self.curr_cuda_data.geid, new_eid))
        x = th.cat((self.curr_cuda_data.x, new_x))
        edge_index = th.cat((rev_curr_edge_index, new_edge_index), dim=1)

        # Update cache snapshot
        self.curr_cuda_data.gnid = nid
        self.curr_cuda_data.geid = eid
        self.curr_cuda_data.x = x
        self.curr_cuda_data.edge_index = edge_index

        if (
            hasattr(self.curr_cuda_data, "edge_attr")
            and self.curr_cuda_data.edge_attr is not None
        ):
            new_edge_attr = self.main_data.edge_attr[new_eid].to(  # type: ignore
                self.device, non_blocking=True
            )  # type:ignore
            self.curr_cuda_data.edge_attr = th.cat(
                (
                    self.curr_cuda_data.edge_attr,
                    new_edge_attr,
                ),
                dim=0,
            )

        self.curr_cuda_data = self._map_edge_index(self.curr_cuda_data)

    def _remove_old(self, nmask_rm: th.Tensor, emask_rm: th.Tensor):
        assert self.curr_cuda_data is not None
        nmask_rm = nmask_rm.to(self.device, non_blocking=True)
        emask_rm = emask_rm.to(self.device, non_blocking=True)

        keep_lnid = mask_to_index(~nmask_rm[self.curr_cuda_data.gnid])
        keep_leid = mask_to_index(~emask_rm[self.curr_cuda_data.geid])

        # Reverse src, dst of edges to orig nid
        rev_curr_edge_index = self.curr_cuda_data.gnid[self.curr_cuda_data.edge_index]

        # Remove old part
        self.curr_cuda_data.gnid = self.curr_cuda_data.gnid[keep_lnid]
        self.curr_cuda_data.geid = self.curr_cuda_data.geid[keep_leid]
        self.curr_cuda_data.x = self.curr_cuda_data.x[keep_lnid]
        self.curr_cuda_data.edge_index = rev_curr_edge_index[:, keep_leid]

        if (
            hasattr(self.curr_cuda_data, "edge_attr")
            and self.curr_cuda_data.edge_attr is not None
        ):
            self.curr_cuda_data.edge_attr = self.curr_cuda_data.edge_attr[keep_leid]

        self.curr_cuda_data = self._map_edge_index(self.curr_cuda_data)

    @staticmethod
    def _map_edge_index(data: SubData):
        # Map src, dst of edges to local nid
        # FIXME: Remove test
        assert data.gnid.unique(sorted=False).size(0) == data.gnid.size(0)

        edge_index, _ = map_index(
            data.edge_index.view(-1),
            data.gnid,
            max_index=data.gnid.max(),
            inclusive=True,
        )
        data.edge_index = edge_index.view(2, -1)

        return data

    def reset_state(self, reset_register: bool = False):
        self.curr_snapshot_id = None
        self.curr_cuda_data = None
        if reset_register:
            self.snapshots.clear()

    def __len__(self):
        return len(self.snapshots)
