import cProfile
import time

import dgl
import polars as pl
import torch as th
from rich.progress import track

from research.dataset import RedditBodyDataset
from research.model.layer.gcn import GraphConv
from research.snapshot_loader import SnapshotLoader
from research.transform import edge_life
from statistic.function import find_edge_overlap_

num_iterations = 20
warmup_cycles = 5

dataset = RedditBodyDataset(force_reload=True)
dataset = edge_life(dataset, 7)

# gconv = GraphConv(300, 10, allow_zero_in_degree=True).to("cuda")
gconv = GraphConv(300, 10, bias=False, norm=False).to("cuda")


def tmp():
    times = []
    for i in range(15):
        with SnapshotLoader(dataset, 0, 173) as iter:
            th.cuda.synchronize()
            start = time.perf_counter()
            for idx, (graph, compute_eids) in enumerate(iter):
                # print(idx)
                rst = gconv(graph, graph.ndata["feat"], compute_eids)
            th.cuda.synchronize()
            end = time.perf_counter()

            # breakpoint()

        if i >= 5:
            times.append(end - start)
        print(end - start)
        th.cuda.empty_cache()

    print(f"Avg: {sum(times) / len(times)}")


cProfile.run("tmp()")

# tmp()
breakpoint()


th.cuda.empty_cache()

times = []
for i in range(15):
    th.cuda.synchronize()
    start = time.perf_counter()
    for snapshot in dataset:
        snapshot = snapshot.to("cuda")
        rst = gconv(snapshot, snapshot.ndata["feat"])
    th.cuda.synchronize()
    end = time.perf_counter()

    if i >= 5:
        times.append(end - start)
    print(end - start)
    th.cuda.empty_cache()

print(f"Avg: {sum(times) / len(times)}")
breakpoint()


def profile_transfed():
    """Profile transfer time for specified method"""
    torch.cuda.synchronize()  # Clear pending operations [3][6]

    # Warmup
    # for _ in track(range(warmup_cycles), description="Warmup..."):
    for _ in range(warmup_cycles):
        for i, snapshot in enumerate(dataset):
            if i == 0:
                continue
            else:
                find_edge_overlap_(dataset, i - 1, i)

            snapshot.to("cuda")
            snapshot.ndata["feat"].cuda()
            snapshot.edata["feat"].cuda()
            snapshot.edata["label"].cuda()

    breakpoint()

    # Timed runs
    times = []
    for _ in track(range(num_iterations), description="Profiling..."):
        snapshot_times = []
        for mask_id in range(len(dataset)):
            node_mask = dataset.df_nodes[f"mask_{mask_id}"].to_torch()
            edge_mask = dataset.df_edges[f"mask_{mask_id}"].to_torch()

            torch.cuda.synchronize()
            start = time.perf_counter()

            src[edge_mask].cuda()
            dst[edge_mask].cuda()
            prev_nfeat[node_mask].cuda()
            prev_efeat[edge_mask].cuda()
            elabel[edge_mask].cuda()

            graph_ = dgl.graph((src_, dst_))
            graph_.ndata["feat"] = nfeat_
            graph_.edata["feat"] = efeat_
            graph_.edata["label"] = elabel_

            torch.cuda.synchronize()
            snapshot_times.append(time.perf_counter() - start)
        times.append(snapshot_times)

    return [sum(snapshot_times_) * 1000 for snapshot_times_ in times]  # Average ms


# Profile both methods
transfer_time = profile_transfer()

breakpoint()

# print(f"Average transfer times: {transfer_time:.3f} ms")
