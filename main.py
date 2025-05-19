import cProfile
import time

import torch as th
from torch.cuda import synchronize
from torch_geometric.nn.conv.gcn_conv import GCNConv

from research.base import SnapshotContext
from research.compute.cache_manager import AggCacheManager, create_cached_model
from research.dataset import EllipticTxTx
from research.loader import SnapshotManager
from research.model import TGCN
from research.model.layer.gcn import CacheableGCNConv
from research.transform import edge_life
from research.utils import edge_subgraph

th.backends.cudnn.benchmark = True

dataset = EllipticTxTx()
dataset = edge_life(dataset, life=3)

context = SnapshotContext(dataset._data)
manager = SnapshotManager(context)
model = TGCN(182, 3, 16, gcn_norm=False).to("cuda")

for id, (nmask, emask) in enumerate(zip(dataset._data.nmasks, dataset._data.emasks)):
    manager.register_snapshot(id, nmask, emask)

cached_model = create_cached_model(model, context)
breakpoint()


def tmp():
    synchronize()
    for t in range(5):
        start = time.perf_counter()
        hn = None
        for i, snapshot in manager.get_generator():
            _, hn = cached_model(snapshot.x, snapshot.edge_index, hn)
        synchronize()
        end = time.perf_counter()
        print(end - start)
        th.cuda.empty_cache()


# cProfile.run("tmp()")
tmp()
th.cuda.empty_cache()
breakpoint()


def base():
    synchronize()
    for _ in range(5):
        start = time.perf_counter()
        hn = None
        for meta in context.metadata.values():
            snapshot = edge_subgraph(dataset._data, meta.geid).cuda(non_blocking=True)
            _, hn = model(snapshot.x, snapshot.edge_index, hn)
        synchronize()
        end = time.perf_counter()
        print(end - start)
        th.cuda.empty_cache()


# cProfile.run("base()")
base()
breakpoint()
