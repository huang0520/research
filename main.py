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

with SnapshotLoader(dataset, 0, 173) as iter:
    for idx, (graph, compute_eids) in enumerate(iter):
        print(idx)
        rst = gconv(graph, graph.ndata["feat"], compute_eids)
        # breakpoint()

breakpoint()

# Aggregation selection
g0_node_mask = dataset.df_nodes.select("mask_0").to_torch().ravel()
g0_edge_mask = dataset.df_edges.select("mask_0").to_torch().ravel()
g1_node_mask = dataset.df_nodes.select("mask_1").to_torch().ravel()
g1_edge_mask = dataset.df_edges.select("mask_1").to_torch().ravel()

g0_node_extra = g0_node_mask & ~g1_node_mask
g1_node_extra = g1_node_mask & ~g0_node_mask

g0_node_extra_index = torch.where(g0_node_extra)[0]
g1_node_extra_index = torch.where(g1_node_extra)[0]

g0_node_extra_out_nodes = (
    dataset.df_edges.filter(pl.col("src_nid").is_in(g0_node_extra_index.tolist()))
    .select(pl.col("dst_nid").unique())
    .to_torch()
    .ravel()
)
g1_node_extra_out_nodes = (
    dataset.df_edges.filter(pl.col("src_nid").is_in(g1_node_extra_index.tolist()))
    .select(pl.col("dst_nid").unique())
    .to_torch()
    .ravel()
)


breakpoint()

times_our = []
for _ in range(10):
    with SnapshotLoader(dataset, 0, 173) as iter:
        times = []
        for _ in range(173):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = next(iter)
            torch.cuda.synchronize()
            end = time.perf_counter()
            del _
            torch.cuda.empty_cache()
            times.append((end - start) * 1000)
        times_our.append(sum(times))


times_dgl = []
for _ in range(10):
    times = []
    for snapshot in dataset:
        torch.cuda.synchronize()
        start = time.perf_counter()
        snapshot.to("cuda")
        snapshot.ndata["feat"].cuda()
        snapshot.edata["feat"].cuda()
        torch.cuda.synchronize()
        end = time.perf_counter()
        del snapshot
        torch.cuda.empty_cache()
        times.append((end - start) * 1000)
    times_dgl.append(sum(times))

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
