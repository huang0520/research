from collections.abc import Callable, Iterator
from queue import Queue
from threading import Event, Lock, Thread

import torch as th
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.typing import EdgeType, NodeOrEdgeType, NodeType
from torch_geometric.utils import mask_to_index
from torch_geometric.utils.map import map_index

from research.base import (
    SnapshotContext,
    SnapshotDiffInfo,
    SnapshotMetadata,
    SubData,
)
from research.dataset.base import TransferPackage
from research.utils import edge_subgraph


class AsyncPipeline:
    def __init__(
        self,
        dataloader: DataLoader,
        device: str = "cuda",
    ):
        self.dataloader = dataloader
        self.device = device

        self.transfer_stream = th.cuda.Stream(device)
        self.compose_stream = th.cuda.Stream(device)
        self.compute_stream = th.cuda.Stream(device)

        self.compose_fn = PackageProcessor()

    def __iter__(self):
        return PipelinedIterator(
            iter(self.dataloader),
            self.transfer_stream,  # type:ignore
            self.compose_stream,  # type:ignore
            self.compute_stream,  # type:ignore
            self.compose_fn,
            self.device,
        )


class PipelinedIterator:
    def __init__(
        self,
        dataloader_iter: Iterator,
        transfer_stream: th.cuda.Stream,
        compose_stream: th.cuda.Stream,
        compute_stream: th.cuda.Stream,
        compose_fn: Callable,
        device: str,
    ) -> None:
        self.dataloader_iter = dataloader_iter
        self.transfer_stream = transfer_stream
        self.compose_stream = compose_stream
        self.compute_stream = compute_stream
        self.compose_fn = compose_fn
        self.device = device

        # State tracking
        self.prev_data: HeteroData | None = None  # Previous composed data
        self.stop_threads = Event()

        self.package_queue: Queue[
            tuple[TransferPackage | Exception | None, th.cuda.Event | None]
        ] = Queue(maxsize=1)
        self.data_queue: Queue[
            tuple[int | Exception | None, HeteroData | None, th.cuda.Event | None]
        ] = Queue(maxsize=1)

        self._start_loading_thread(self.dataloader_iter)
        self._start_compose_thread()

    def __next__(self):
        item = self.data_queue.get()

        if item[0] is None:
            self.stop_threads.set()
            raise StopIteration

        if isinstance(item[0], Exception):
            self.stop_threads.set()
            raise item[0]

        id, data, compose_event = item
        assert (
            isinstance(id, int)
            and isinstance(data, HeteroData)
            and isinstance(compose_event, th.cuda.Event)
        )

        self.compute_stream.wait_event(compose_event)
        return id, data

    def _start_loading_thread(self, dataloader_iter: Iterator):
        def worker():
            try:
                for package in dataloader_iter:
                    if self.stop_threads.is_set():
                        break

                    with th.cuda.stream(self.transfer_stream):  # type:ignore
                        package_gpu: TransferPackage = package.to(self.device)
                        transfer_event = th.cuda.Event()
                        transfer_event.record(self.transfer_stream)

                    self.package_queue.put((package_gpu, transfer_event))  # type:ignore

                self.package_queue.put((None, None))

            except Exception as e:
                self.package_queue.put((e, None))

        self.transfer_thread = Thread(target=worker, daemon=True)
        self.transfer_thread.start()

    def _start_compose_thread(self):
        def worker():
            try:
                while not self.stop_threads.is_set():
                    item = self.package_queue.get()

                    if item[0] is None:
                        self.data_queue.put((None, None, None))
                        break
                    elif isinstance(item[0], Exception):
                        self.data_queue.put((item[0], None, None))
                        break

                    package, transfer_event = item
                    assert isinstance(package, TransferPackage) and isinstance(
                        transfer_event, th.cuda.Event
                    )

                    with th.cuda.stream(self.compose_stream):  # type:ignore
                        self.compose_stream.wait_event(transfer_event)  # type:ignore

                        id, data = self.compose_fn(package, self.prev_data)

                        compose_event = th.cuda.Event()
                        compose_event.record(self.compose_stream)

                        self.prev_data = data

                    self.data_queue.put((id, data, compose_event))  # type: ignore

            except Exception as e:
                self.data_queue.put((e, None, None))

        self.compose_thread = Thread(target=worker, daemon=True)
        self.compose_thread.start()


class PackageProcessor:
    def __call__(self, package: TransferPackage, prev_data: HeteroData | None = None):
        if prev_data is None:
            return self._process_init_package(package)
        else:
            return self._process_incremental_package(package, prev_data)

    def _process_init_package(self, package: TransferPackage):
        data = HeteroData()

        for t, gid in package.new_gids.items():
            data[t]["gid"] = gid
        for ntype, x in package.add_xs.items():
            data[ntype]["x"] = x
        for etype, edge_index in package.add_edge_indexes.items():
            data[etype]["edge_index"] = edge_index
            data[etype]["edge_index"] = self._map_edge_index_to_local(data)
        for etype, edge_attr in package.add_edge_attrs.items():
            data[etype]["edge_attr"] = edge_attr

        data = data.contiguous()
        return package.curr_id, data

    def _process_incremental_package(
        self, package: TransferPackage, prev_data: HeteroData
    ):
        all_types = (*prev_data.node_types, *prev_data.edge_types)

        # Reverse src, dst local nid of edge index to global nid
        global_edge_index = self._map_edge_index_to_global(prev_data)

        # Update data
        data = HeteroData()
        for t in all_types:
            data[t]["gid"] = package.new_gids[t]

            if isinstance(t, NodeType):
                data[t]["x"] = th.cat((
                    prev_data[t]["x"][package.keep_prev_lids[t]],
                    package.add_xs[t],
                ))

            if isinstance(t, tuple):
                data[t]["edge_index"] = th.cat(
                    (
                        global_edge_index[:, package.keep_prev_lids[t]],
                        package.add_edge_indexes[t],
                    ),
                    dim=1,
                )
                data[t]["edge_index"] = self._map_edge_index_to_local(data)

            if isinstance(t, tuple) and package.add_edge_attrs:
                data[t]["edge_attr"] = th.cat((
                    prev_data[t]["edge_attr"][package.keep_prev_lids[t]],
                    package.add_edge_attrs[t],
                ))

        data = data.contiguous()
        return package.curr_id, data

    @staticmethod
    def _map_edge_index_to_global(data: HeteroData):
        local_edge_index: Tensor = data.edge_stores[0]["edge_index"]
        etype = data.edge_types[0]

        if etype[0] == etype[2]:
            global_edge_index: Tensor = data[etype[0]]["gid"][local_edge_index]
        else:
            global_edge_index = th.zeros_like(local_edge_index)
            global_edge_index[0] = data[etype[0]]["gid"][local_edge_index]
            global_edge_index[1] = data[etype[2]]["gid"][local_edge_index]

        return global_edge_index

    @staticmethod
    def _map_edge_index_to_local(data: HeteroData):
        # Map src, dst of edges to local nid
        global_edge_index: Tensor = data.edge_stores[0]["edge_index"]
        etype = data.edge_types[0]

        if etype[0] == etype[2]:
            # Same src, dst node type
            edge_index, _ = map_index(
                global_edge_index.view(-1),
                data.node_stores[0]["gid"],
                max_index=data.node_stores[0]["gid"].max(),
                inclusive=True,
            )
            local_edge_index = edge_index.view(2, -1)
        else:
            src_nid: Tensor = data[etype[0]]["gid"]
            dst_nid: Tensor = data[etype[2]]["gid"]
            src_index, _ = map_index(
                global_edge_index[0], src_nid, src_nid.max(), inclusive=True
            )
            dst_index, _ = map_index(
                global_edge_index[1], dst_nid, dst_nid.max(), inclusive=True
            )
            local_edge_index = th.zeros_like(global_edge_index)
            local_edge_index[0] = src_index
            local_edge_index[1] = dst_index

        return local_edge_index


class SnapshotManager:
    def __init__(self, context: SnapshotContext) -> None:
        self.context = context

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
