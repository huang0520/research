import cProfile
from pathlib import Path

import polars as pl
import torch
from rich.progress import track

from research.dataset import RedditBodyDataset
from research.transform import edge_life
from statistic.function import (
    find_amount_computes,
    find_amount_data_transfer,
    find_edge_overlap,
    find_edge_overlap_,
    find_node_overlap,
    find_nodes,
    find_num_nodes,
    find_overlap_aggregation,
)

output_dir = Path("output/reddit-body")
output_dir.mkdir(parents=True, exist_ok=True)

# Create dataset
dataset = RedditBodyDataset()
dataset = edge_life(dataset, life=5)

stats = {
    "num_nodes": [],
    "num_edges": [],
    "num_overlap_edges": [],
    "num_non_overlap_edges": [],
    "num_overlap_nodes": [],
    "num_non_overlap_nodes": [],
    "ratio_overlap_edges": [],
    "ratio_overlap_nodes": [],
    "amount_total_edge_transfer": [],
    "amount_total_edge_feature_transfer": [],
    "amount_total_node_feature_transfer": [],
    "amount_extra_edge_transfer": [],
    "amount_extra_edge_feature_transfer": [],
    "amount_extra_node_feature_transfer": [],
    "amount_total_computes": [],
    "amount_extra_computes": [],
}
lf_nodes = dataset.lf_nodes
lf_edges = dataset.lf_edges
for snapshot_idx in range(len(dataset)):
    # for snapshot_idx in track(range(0, len(dataset))):
    curr_snapshot = dataset[snapshot_idx]
    # num_nodes = find_num_nodes(curr_snapshot)

    if snapshot_idx == 0:
        continue

    find_edge_overlap_(lf_edges, snapshot_idx - 1, snapshot_idx)

    continue

    if snapshot_idx > 0:
        prev_snapshot = dataset[snapshot_idx - 1]

        overlap_eids, prev_non_overlap_eids, curr_non_overlap_eids = find_edge_overlap(
            prev_snapshot, curr_snapshot
        )
        overlap_nids, prev_non_overlap_nids, curr_non_overlap_nids = find_node_overlap(
            prev_snapshot, curr_snapshot
        )
        overlap_agg_nids, non_overlap_agg_nids = find_overlap_aggregation(
            prev_snapshot,
            curr_snapshot,
            (prev_non_overlap_eids, curr_non_overlap_eids),
            (prev_non_overlap_nids, curr_non_overlap_nids),
        )
    else:
        curr_nids = find_nodes(curr_snapshot)
        overlap_eids = torch.zeros(0)
        prev_non_overlap_eids = torch.zeros(0)
        curr_non_overlap_eids = curr_snapshot.edges("eid")
        overlap_nids = torch.zeros(0)
        prev_non_overlap_nids = torch.zeros(0)
        curr_non_overlap_nids = curr_nids
        non_overlap_agg_nids = curr_nids

    total_transfer, only_extra_transfer = find_amount_data_transfer(
        curr_snapshot,
        (prev_non_overlap_eids, curr_non_overlap_eids),
        (prev_non_overlap_nids, curr_non_overlap_nids),
    )
    total_computes, only_extra_computes = find_amount_computes(
        curr_snapshot, non_overlap_agg_nids
    )

    edge_overlap_ratio = len(overlap_eids) / curr_snapshot.num_edges()
    node_overlap_ratio = len(overlap_nids) / num_nodes

    stats["num_nodes"].append(num_nodes)
    stats["num_edges"].append(curr_snapshot.num_edges())
    stats["num_overlap_edges"].append(len(overlap_eids))
    stats["num_non_overlap_edges"].append(len(curr_non_overlap_eids))
    stats["num_overlap_nodes"].append(len(overlap_nids))
    stats["num_non_overlap_nodes"].append(len(curr_non_overlap_nids))
    stats["ratio_overlap_edges"].append(edge_overlap_ratio)
    stats["ratio_overlap_nodes"].append(node_overlap_ratio)
    stats["amount_total_edge_transfer"].append(total_transfer[0])
    stats["amount_total_edge_feature_transfer"].append(total_transfer[1])
    stats["amount_total_node_feature_transfer"].append(total_transfer[2])
    stats["amount_extra_edge_transfer"].append(only_extra_transfer[0])
    stats["amount_extra_edge_feature_transfer"].append(only_extra_transfer[1])
    stats["amount_extra_node_feature_transfer"].append(only_extra_transfer[2])
    stats["amount_total_computes"].append(total_computes)
    stats["amount_extra_computes"].append(only_extra_computes)

breakpoint()

df_stats = pl.DataFrame(stats)
df_stats.write_csv(output_dir / "statistic.csv")
# breakpoint()
