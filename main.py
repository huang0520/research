import cProfile
import time

import torch as th
from torch.cuda import synchronize
from torch_geometric.nn.conv.gcn_conv import GCNConv

from research.base import SnapshotContext
from research.compute.cache_manager import AggCacheManager, create_cached_model
from research.dataset import EllipticTxTx
from research.loader import SnapshotManager
from research.model.layer.gcn import CacheableGCNConv
from research.transform import edge_life
from research.utils import edge_subgraph

dataset = EllipticTxTx()
dataset = edge_life(dataset, life=5)

context = SnapshotContext(dataset._data)
manager = SnapshotManager(context)
_layer = GCNConv(182, 10, bias=True, normalize=False).to("cuda")
layer = CacheableGCNConv(182, 10, bias=True, normalize=False).to("cuda")

for id, (nmask, emask) in enumerate(zip(dataset._data.nmasks, dataset._data.emasks)):
    manager.register_snapshot(id, nmask, emask)

cached_layer = create_cached_model(layer, context)


def tmp():
    synchronize()
    for t in range(5):
        start = time.perf_counter()
        for i, snapshot in manager.get_generator():
            _ = cached_layer(snapshot.x, snapshot.edge_index)
        synchronize()
        end = time.perf_counter()
        print(end - start)
        th.cuda.empty_cache()


cProfile.run("tmp()")

th.cuda.empty_cache()
print()

synchronize()
for _ in range(5):
    start = time.perf_counter()
    for i, meta in context.metadata.items():
        snapshot = edge_subgraph(dataset._data, meta.geid).cuda(non_blocking=True)
        _ = _layer(snapshot.x, snapshot.edge_index)
    synchronize()
    end = time.perf_counter()
    print(end - start)
    th.cuda.empty_cache()

breakpoint()
