from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from research.dataset import EllipticTxTx
from research.transform import edge_life

output_dir = Path("./output/EllipticTxTx")
output_dir.mkdir(parents=True, exist_ok=True)

# Create dataset
dataset = EllipticTxTx()
dataset = edge_life(dataset, life=3)

# Basic dataset statistic
df_statistic = pd.DataFrame()
df_statistic.attrs["embedding_size"] = dataset.graph.ndata["feature"].shape[1]  # type:ignore
df_statistic["num_nodes"] = [
    torch.nonzero(snapshot_mask).shape[0] for snapshot_mask in dataset.snapshot_masks
]
df_statistic["num_edges"] = [snapshot.num_edges() for snapshot in dataset]

# Find overlap of edges between snapshots
num_non_overlap_edges = []
num_overlap_edges = []
num_transfer_edges = []
num_non_overlap_embeddings = []
num_overlap_embeddings = []
num_non_overlap_aggregations = []
num_overlap_aggregations = []
amount_all_transfers = []
amount_extra_edge_transfers = []
amount_both_transfers = []
amount_all_computes = []
amount_needed_computes = []
amount_unneeded_computes = []

prev_edges: set[tuple[int, int]] = set()
prev_embeddings: dict[int, Tensor] = {}
for snapshot_idx in range(len(dataset)):
    snapshot = dataset[snapshot_idx]
    snapshot_mask = dataset.snapshot_masks[snapshot_idx]
    embedding_size = dataset.graph.ndata["feature"].shape[1]  # type:ignore

    # 1. Find overlap of edges between snapshots
    # * Directed graph
    #   (Indirected graph need to ignore the direction of tuple)
    curr_edges: set[tuple[int, int]] = set(
        zip(snapshot.edges()[0].tolist(), snapshot.edges()[1].tolist())
    )

    overlap_edges = prev_edges & curr_edges
    prev_ext = prev_edges - overlap_edges
    curr_ext = curr_edges - overlap_edges

    num_non_overlap_edges.append(len(curr_ext))
    num_overlap_edges.append(len(overlap_edges))
    num_transfer_edges.append(len(prev_ext) + len(curr_ext))
    prev_edges = curr_edges

    # 2. Find overlap of embedding between snapshots
    node_idx = torch.nonzero(snapshot_mask).flatten().tolist()
    curr_embeddings = {i: dataset.graph.ndata["feature"][i] for i in node_idx}

    overlap_embedding_keys = {
        key
        for key in prev_embeddings.keys() & curr_embeddings.keys()
        if torch.equal(prev_embeddings[key], curr_embeddings[key])
    }
    non_overlap_embedding_keys = curr_embeddings.keys() - overlap_embedding_keys

    num_non_overlap_embeddings.append(len(non_overlap_embedding_keys))
    num_overlap_embeddings.append(len(overlap_embedding_keys))
    prev_embeddings = curr_embeddings

    # 3. Find overlap of aggregation result
    # * Find updated aggregation result by edge
    #   (result of dst of a new/removed edge will change)
    updated_result_keys_by_edge = {v for _, v in curr_ext | prev_ext}

    # * Find updated aggregation result by embedding
    #   (self and neighbor result will change)
    out_edges = snapshot.out_edges(tuple(non_overlap_embedding_keys))  # type:ignore
    updated_result_keys_by_node = set((*out_edges[0].tolist(), *out_edges[1].tolist()))

    updated_result_keys = updated_result_keys_by_edge | updated_result_keys_by_node

    num_non_overlap_aggregations.append(len(updated_result_keys))
    num_overlap_aggregations.append(len(node_idx) - len(updated_result_keys))

    # 4. Amount of data transfer
    #    (Each data need 8 bytes)
    amount_all_edge_transfer = snapshot.num_edges() * 8
    amount_extra_edge_transfer = (len(curr_ext) + len(prev_ext)) * 8

    amount_all_embedding_transfer = len(node_idx) * embedding_size * 8
    amount_non_overlap_embedding_transfer = (
        len(non_overlap_embedding_keys) * embedding_size * 8
    )

    amount_all_transfers.append(
        amount_all_edge_transfer + amount_all_embedding_transfer
    )
    amount_extra_edge_transfers.append(
        amount_extra_edge_transfer + amount_all_embedding_transfer
    )
    amount_both_transfers.append(
        amount_extra_edge_transfer + amount_non_overlap_embedding_transfer
    )

    # 5. Amount of computation
    amount_all_compute = snapshot.num_edges() * embedding_size
    amount_needed_compute = (
        snapshot.in_degrees(tuple(updated_result_keys)).sum().item() * embedding_size  # type:ignore
    )
    amount_unneeded_compute = amount_all_compute - amount_needed_compute

    # Each compute need 2 flops (add + multiply)
    amount_all_computes.append(amount_all_compute * 2)
    amount_needed_computes.append(amount_needed_compute * 2)
    amount_unneeded_computes.append(amount_unneeded_compute * 2)


df_statistic["num_non_overlap_edges"] = num_non_overlap_edges
df_statistic["num_overlap_edges"] = num_overlap_edges
df_statistic["num_transfer_edges"] = num_transfer_edges
df_statistic["num_non_overlap_embeddings"] = num_non_overlap_embeddings
df_statistic["num_overlap_embeddings"] = num_overlap_embeddings
df_statistic["num_non_overlap_aggregations"] = num_non_overlap_aggregations
df_statistic["num_overlap_aggregations"] = num_overlap_aggregations
df_statistic["amount_all_transfers"] = amount_all_transfers
df_statistic["amount_extra_edge_transfers"] = amount_extra_edge_transfers
df_statistic["amount_both_transfers"] = amount_both_transfers
df_statistic["amount_all_computes"] = amount_all_computes
df_statistic["amount_needed_computes"] = amount_needed_computes
df_statistic["amount_unneeded_computes"] = amount_unneeded_computes


# Output
df_statistic.to_csv(output_dir / "statistic.csv")
