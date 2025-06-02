import cProfile
import time

import torch as th
from torch import nn
from torch.autograd import profiler
from torch.cuda import synchronize
from torch_geometric.nn.conv.gcn_conv import GCNConv

from research.base import SnapshotContext
from research.compute.cache_manager import (
    AggCacheManager,
    CachedModule,
    create_cached_model,
)
from research.dataset import EllipticTxTx, RedditBodyDataset
from research.loader import SnapshotManager
from research.model import TGCN
from research.model.layer.gcn import CacheableGCNConv
from research.transform import edge_life
from research.utils import edge_subgraph

th.backends.cudnn.benchmark = True

# dataset = EllipticTxTx()
dataset = RedditBodyDataset(force_reload=True)
dataset = edge_life(dataset, life=3)

breakpoint()

context = SnapshotContext(dataset._data)
manager = SnapshotManager(context)
model = TGCN(182, 3, 100, gcn_norm=False).to("cuda")

for id, (nmask, emask) in enumerate(zip(dataset._data.nmasks, dataset._data.emasks)):
    manager.register_snapshot(id, nmask, emask)

cached_model = create_cached_model(model, context)


class TimingHook:
    def __init__(self, name):
        self.name = name
        self.times = []

    def __call__(self, module, input, output):
        end_time = time.time()
        if hasattr(module, "start_time"):
            self.times.append((end_time - module.start_time) * 1000)
        module.start_time = time.time()


our_hooks = []
for name, module in cached_model.named_modules():
    if isinstance(module, (CachedModule, nn.GRU)):
        hook = TimingHook(name)
        module.register_forward_hook(hook)
        our_hooks.append(hook)


base_hooks = []
for name, module in model.named_modules():
    if isinstance(module, (CacheableGCNConv, nn.GRU)):
        hook = TimingHook(name)
        module.register_forward_hook(hook)
        base_hooks.append(hook)


def our():
    cached_model.eval()
    synchronize()
    for _ in range(5):
        hn = None
        for _, snapshot in manager.get_generator():
            _, hn = cached_model(snapshot.x, snapshot.edge_index, hn)

    synchronize()
    for t in range(10):
        # start = time.perf_counter()
        hn = None
        with profiler.profile(use_device="cuda") as prof:
            for i, snapshot in manager.get_generator():
                _, hn = cached_model(snapshot.x, snapshot.edge_index, hn)

        print(prof.key_averages().table("cuda_time_total"))
        # synchronize()
        # end = time.perf_counter()
        # print(end - start)
        # th.cuda.empty_cache()
        breakpoint()


def base():
    model.eval()
    synchronize()
    for t in range(15):
        start = time.perf_counter()
        hn = None
        with th.no_grad():
            for meta in context.metadata.values():
                snapshot = edge_subgraph(dataset._data, meta.geid).cuda(
                    non_blocking=True
                )
                _, hn = model(snapshot.x, snapshot.edge_index, hn)
        synchronize()
        end = time.perf_counter()
        if t > 4:
            print(end - start)
        th.cuda.empty_cache()


# cProfile.run("base()")
# base()
th.cuda.empty_cache()
print()
# cProfile.run("our()")
our()
