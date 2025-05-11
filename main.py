import cProfile
import time

import torch as th
from torch.cuda import synchronize

from research.compute.cache_manager import AggregationCache
from research.data import SnapshotManager
from research.data.dataset import EllipticTxTx
from research.model.layer.gcn import CacheableGCNConv
from research.transform import edge_life
from research.utils import edge_subgraph

dataset = EllipticTxTx()
dataset = edge_life(dataset, life=5)

manager = SnapshotManager(dataset._data)
cache = AggregationCache(manager)
layer = CacheableGCNConv(182, 10, bias=True, normalize=False).to("cuda")

for id, (nmask, emask) in enumerate(zip(dataset._data.nmasks, dataset._data.emasks)):
    manager.register_snapshot(id, nmask, emask)

cached_layer = cache.register_model(layer)

for i, snapshot in manager.get_generator():
    print(i)
    cached_layer(snapshot.x, snapshot.edge_index, i)

breakpoint()


synchronize()
for _ in range(5):
    start = time.perf_counter()
    for i, snapshot in manager.get_generator():
        # print(snapshot.coo())
        # breakpoint()
        pass
    synchronize()
    end = time.perf_counter()
    print(end - start)

breakpoint()

th.cuda.empty_cache()

synchronize()
for _ in range(5):
    start = time.perf_counter()
    for i, info in manager.snapshots.items():
        snapshot = edge_subgraph(dataset._data, info.eid).cuda(non_blocking=True)
    synchronize()
    end = time.perf_counter()
    print(end - start)

breakpoint()
