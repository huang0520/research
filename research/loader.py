from collections.abc import Iterable

import torch as th
from torch.utils.data import DataLoader
from torch_geometric.utils import mask_to_index
from torch_geometric.utils.map import map_index

from research.base import (
    SnapshotContext,
    SnapshotDiffInfo,
    SnapshotMetadata,
    SubData,
)
from research.utils import edge_subgraph


class SnapshotManager:
    def __init__(self, context: SnapshotContext) -> None:
        self.context = context

    def get_generator(
        self, snapshot_ids: Iterable[int] | None = None, reset_after_fin: bool = True
    ):
        if snapshot_ids is None:
            snapshot_ids = self.context.metadata.keys()
        else:
            assert all(id in self.context.metadata for id in snapshot_ids)

        for i in snapshot_ids:
            yield i, self.load_snapshot(i)

        if reset_after_fin:
            self.context.reset_state()

    def register_snapshot(self, snapshot_id: int, nmask: th.Tensor, emask: th.Tensor):
        assert nmask.dtype == th.bool and emask.dtype == th.bool
        nid = mask_to_index(nmask)
        eid = mask_to_index(emask)
        self.context.metadata[snapshot_id] = SnapshotMetadata(
            snapshot_id, nid, eid, nmask, emask
        )

    def load_snapshot(self, snapshot_id: int) -> SubData:
        assert snapshot_id in self.context.metadata

        # If it is first time to load snapshot
        if self.context.curr_id is None or self.context.curr_data is None:
            snapshot_meta = self.context.metadata[snapshot_id]
            snapshot_data = edge_subgraph(self.context.main_data, snapshot_meta.geid)

            self.context.curr_id = snapshot_id
            self.context.curr_data = snapshot_data.to(
                self.context.device, non_blocking=True
            )
        else:
            self.context.step()
            self.context.curr_id = snapshot_id

            # Found the difference between current and target snapshot
            assert self.context.prev_id is not None
            diff_info = self.context.diff

            self._incremental_update(self.context, diff_info)

        assert self.context.curr_data is not None
        return self.context.curr_data

    def _incremental_update(
        self, context: SnapshotContext, diff_info: SnapshotDiffInfo
    ):
        """

        Args:
        - nodes_add: Mask of nodes to add
        - edges_add: Mask of edges to add
        - nodes_rm: Mask of nodes to remove
        - edges_rm: Mask of edges to remove
        """
        assert context.prev_data is not None and context.curr_data is not None

        if diff_info.rm_nmask.any() or diff_info.rm_emask.any():
            self._remove_old(context, diff_info)

        if diff_info.add_nmask.any() or diff_info.add_emask.any():
            self._merge_new(context, diff_info)

    def _merge_new(self, context: SnapshotContext, diff_info: SnapshotDiffInfo):
        """

        Args:
        - new_nid: Original NID of new nodes
        - new_eid: Orignial EID of new edges
        """
        assert context.curr_data is not None

        total_n_nodes = context.curr_data.gnid.size(0) + diff_info.add_gnid.size(0)
        total_n_edges = context.curr_data.geid.size(0) + diff_info.add_geid.size(0)

        add_x = context.main_data.x[diff_info.add_gnid].to(
            context.device, non_blocking=True
        )
        add_edge_index = context.main_data.edge_index[:, diff_info.add_geid].to(
            context.device, non_blocking=True
        )

        # Reverse src, dst of edges to orig nid
        rev_curr_edge_index = context.curr_data.gnid[context.curr_data.edge_index]

        # Combine current snapshot and new part (to orig nid)
        nid = th.empty(
            total_n_nodes, device=context.device, dtype=context.curr_data.gnid.dtype
        )
        nid[: context.curr_data.gnid.size(0)] = context.curr_data.gnid
        nid[context.curr_data.gnid.size(0) :] = diff_info.add_gnid.to(context.device)

        eid = th.empty(
            total_n_edges, device=context.device, dtype=context.curr_data.geid.dtype
        )
        eid[: context.curr_data.geid.size(0)] = context.curr_data.geid
        eid[context.curr_data.geid.size(0) :] = diff_info.add_geid.to(context.device)

        x = th.empty(
            (total_n_nodes, context.curr_data.x.size(-1)),
            device=context.device,
            dtype=context.curr_data.x.dtype,
        )
        x[: context.curr_data.gnid.size(0), :] = context.curr_data.x
        x[context.curr_data.gnid.size(0) :, :] = add_x

        edge_index = th.empty(
            (2, total_n_edges), device=context.device, dtype=rev_curr_edge_index.dtype
        )
        edge_index[:, : context.curr_data.geid.size(0)] = rev_curr_edge_index
        edge_index[:, context.curr_data.geid.size(0) :] = add_edge_index

        # Update cache snapshot
        context.curr_data.gnid = nid
        context.curr_data.geid = eid
        context.curr_data.x = x
        context.curr_data.edge_index = edge_index

        if (
            hasattr(context.curr_data, "edge_attr")
            and context.curr_data.edge_attr is not None
        ):
            assert (
                hasattr(context.main_data, "edge_attr")
                and context.main_data.edge_attr is not None
            )

            new_edge_attr = context.main_data.edge_attr[diff_info.add_geid].to(
                context.device, non_blocking=True
            )
            context.curr_data.edge_attr = th.cat(
                (
                    context.curr_data.edge_attr,
                    new_edge_attr,
                ),
                dim=0,
            )

        self.curr_cuda_data = self._map_edge_index(context.curr_data)

    def _remove_old(self, context: SnapshotContext, diff_info: SnapshotDiffInfo):
        assert context.curr_data is not None
        nmask_rm = diff_info.rm_nmask.to(context.device, non_blocking=True)
        emask_rm = diff_info.rm_emask.to(context.device, non_blocking=True)

        keep_lnid = mask_to_index(~nmask_rm[context.curr_data.gnid])
        keep_leid = mask_to_index(~emask_rm[context.curr_data.geid])

        # Reverse src, dst of edges to orig nid
        rev_curr_edge_index = context.curr_data.gnid[context.curr_data.edge_index]

        # Remove old part
        context.curr_data.gnid = context.curr_data.gnid[keep_lnid]
        context.curr_data.geid = context.curr_data.geid[keep_leid]
        context.curr_data.x = context.curr_data.x[keep_lnid]
        context.curr_data.edge_index = rev_curr_edge_index[:, keep_leid]

        if (
            hasattr(context.curr_data, "edge_attr")
            and context.curr_data.edge_attr is not None
        ):
            context.curr_data.edge_attr = context.curr_data.edge_attr[keep_leid]

        context.curr_data = self._map_edge_index(context.curr_data)

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

    def __len__(self):
        return len(self.context)
