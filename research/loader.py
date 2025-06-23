from collections.abc import Callable, Iterator
from queue import Queue
from threading import Event, Thread

import torch as th
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.utils.map import map_index

from research.dataset.base import ComputeInfo, TransferPackage


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
    def __init__(  # noqa: PLR0913, PLR0917
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
        ] = Queue(maxsize=2)
        self.data_queue: Queue[
            tuple[
                int | Exception | None,
                HeteroData | None,
                ComputeInfo | None,
                th.cuda.Event | None,
            ]
        ] = Queue(maxsize=2)

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

        id, data, compute_info, compose_event = item
        with th.cuda.stream(self.compute_stream):
            th.cuda.current_stream().wait_event(compose_event)  # type:ignore
            th.cuda.current_stream().synchronize()

        return id, data, compute_info

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
                        transfer_event.synchronize()

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
                        self.data_queue.put((None, None, None, None))
                        break
                    elif isinstance(item[0], Exception):
                        self.data_queue.put((item[0], None, None, None))
                        break

                    package, transfer_event = item
                    assert isinstance(package, TransferPackage) and isinstance(
                        transfer_event, th.cuda.Event
                    )

                    with th.cuda.stream(self.compose_stream):  # type:ignore
                        self.compose_stream.wait_event(transfer_event)  # type:ignore

                        id, data = self.compose_fn(package, self.prev_data)
                        self.prev_data = data.clone().detach()

                        compose_event = th.cuda.Event()
                        compose_event.record(self.compose_stream)

                    self.data_queue.put((id, data, package.compute_info, compose_event))  # type: ignore

            except Exception as e:
                self.data_queue.put((e, None, None, None))

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
        # Reverse src, dst local nid of edge index to global nid
        global_edge_index = self._map_edge_index_to_global(prev_data)

        # Update data
        data = HeteroData()
        for t, gid in package.new_gids.items():
            data[t]["gid"] = gid
        for t, x in package.add_xs.items():
            if not package.only_edge:
                data[t]["x"] = th.cat((
                    prev_data[t]["x"][package.keep_prev_lids[t]],
                    x,
                ))
            else:
                data[t]["x"] = x
        for t, edge_index in package.add_edge_indexes.items():
            data[t]["edge_index"] = th.cat(
                (
                    global_edge_index[:, package.keep_prev_lids[t]],
                    edge_index,
                ),
                dim=1,
            )
            data[t]["edge_index"] = self._map_edge_index_to_local(data)
        for t, edge_attr in package.add_edge_attrs.items():
            if not package.only_edge:
                data[t]["edge_attr"] = th.cat((
                    prev_data[t]["edge_attr"][package.keep_prev_lids[t]],
                    edge_attr,
                ))
            else:
                data[t]["edge_attr"] = edge_attr

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
