from pathlib import Path

import pandas as pd
import torch
from torch._tensor import Tensor

from research.dataset import EpllipticTxTx
from research.transform import edge_life

output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

dataset = EpllipticTxTx()
dataset = edge_life(dataset, 3)

# Find overlap of edges between snapshots
extra_edges: list[set[tuple[int, int]]] = []
edge_overlap_ratios = []
extra_edge_transfer = []

prev_edges: set[tuple[int, int]] = set()
for snapshot in dataset:
    # Directed graph
    # Indirected graph need to ignore the direction of tuple
    curr_edges: set[tuple[int, int]] = set(
        zip(snapshot.edges()[0].tolist(), snapshot.edges()[1].tolist())
    )

    overlap_edges = prev_edges & curr_edges
    prev_ext = prev_edges - overlap_edges
    curr_ext = curr_edges - overlap_edges

    overlap_ratio = len(overlap_edges) / len(curr_edges)
    extra_transfer = len(prev_ext) + len(curr_ext)

    # print(f"Current number of edges: {len(curr_edges)}")
    # print(f"Previous number of edges: {len(prev_edges)}")
    # print(f"Overlap edges: {len(overlap_edges)} | {overlap_ratio * 100}")
    # print(f"Amount of extra edges transfer: {extra_transfer}")

    extra_edges.append(curr_ext | prev_ext)
    edge_overlap_ratios.append(overlap_ratio)
    extra_edge_transfer.append(extra_transfer)
    prev_edges = curr_edges

# Find overlap of embedding between snapshots
embedding_extra_keys: list[set[int]] = []
embedding_overlap_ratios = []
extra_embedding_transfer = []

prev_embeddings = dict()
for snapshot_mask in dataset.snapshot_masks:
    node_idx = torch.nonzero(snapshot_mask).flatten().tolist()
    curr_embeddings = {idx: dataset.graph.ndata["feature"][idx] for idx in node_idx}

    overlap_keys = prev_embeddings.keys() & curr_embeddings.keys()
    actual_overlap_keys = {
        key
        for key in overlap_keys
        if torch.equal(prev_embeddings[key], curr_embeddings[key])
    }
    extra_keys = curr_embeddings.keys() - actual_overlap_keys

    overlap_ratio = len(actual_overlap_keys) / len(curr_embeddings.keys())
    extra_transfer = len(extra_keys)

    # print(f"Current number of embeddings: {len(curr_embeddings)}")
    # print(f"Previous number of embeddings: {len(prev_embeddings)}")
    # print(f"Overlap embeddings: {len(actual_overlap_keys)} | {overlap_ratio * 100}")
    # print(f"Amount of extra embedding transfer: {extra_transfer}")

    embedding_extra_keys.append(extra_keys)
    embedding_overlap_ratios.append(overlap_ratio)
    extra_embedding_transfer.append(extra_transfer)
    prev_embeddings = curr_embeddings

# Find overlap of aggregation result
update_aggregation_keys = []
overlap_aggregation_ratios = []
for snapshot_idx in range(len(dataset)):
    snapshot = dataset[snapshot_idx]
    snapshot_mask = dataset.snapshot_masks[snapshot_idx]
    extra_edge = extra_edges[snapshot_idx]
    extra_embedding_keys = embedding_extra_keys[snapshot_idx]

    # Updated aggregation result due to edge
    update_aggregation_keys_by_edge = {v for _, v in extra_edge}

    # Updated aggregation result due to embedding
    update_aggregation_keys_by_node = set()
    for node_idx in extra_embedding_keys:
        out_edges: tuple[Tensor, Tensor] = snapshot.out_edges(node_idx)  # type:ignore
        update_aggregation_keys_by_node.update(
            out_edges[0].tolist(), out_edges[1].tolist()
        )

    update_aggregation_keys_ = (
        update_aggregation_keys_by_edge | update_aggregation_keys_by_node
    )

    total_aggregation_keys_ = torch.nonzero(snapshot_mask).flatten().tolist()
    overlap_ratio = 1 - len(update_aggregation_keys_) / len(total_aggregation_keys_)

    update_aggregation_keys.append(update_aggregation_keys_)
    overlap_aggregation_ratios.append(overlap_ratio)

# Record snapshot statistic
num_nodes = [
    len(torch.nonzero(snapshot_mask)) for snapshot_mask in dataset.snapshot_masks
]
num_edges = [snapshot.num_edges() for snapshot in dataset]

df_statistic = pd.DataFrame({
    "num_nodes": num_nodes,
    "num_edges": num_edges,
    "edge_overlap_ratio": edge_overlap_ratios,
    "edge_transfer": extra_edge_transfer,
    "embedding_overlap_ratio": embedding_overlap_ratios,
    "embedding_transfer": extra_embedding_transfer,
    "overlap_aggregation_ratio": overlap_aggregation_ratios,
})
df_statistic.to_csv(output_dir / "EpllipticTxTx.csv")
