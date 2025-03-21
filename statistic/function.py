from collections import defaultdict

import dgl
import numpy as np
import torch
from dgl.heterograph import DGLGraph
from torch import Tensor


def find_nodes(g: DGLGraph) -> Tensor:
    return torch.cat(g.edges()).unique()


def find_num_nodes(g: DGLGraph) -> int:
    return len(find_nodes(g))


def find_edge_overlap(g1: DGLGraph, g2: DGLGraph) -> tuple[Tensor, Tensor, Tensor]:
    # Find common edges with same u, v between g1, g2
    g1_uv_to_eid: dict[tuple[int, int], list[int]] = defaultdict(list)
    g2_uv_to_eid: dict[tuple[int, int], list[int]] = defaultdict(list)

    g1_uv = torch.stack(g1.edges(), dim=1).tolist()
    g2_uv = torch.stack(g2.edges(), dim=1).tolist()

    for eid, (u, v) in enumerate(g1_uv):
        g1_uv_to_eid[u, v].append(eid)
    for eid, (u, v) in enumerate(g2_uv):
        g2_uv_to_eid[u, v].append(eid)

    common_uvs = g1_uv_to_eid.keys() & g2_uv_to_eid.keys()

    g1_feats: Tensor | None = g1.edata.get("feature")
    g2_feats: Tensor | None = g2.edata.get("feature")
    if g1_feats is None:
        # Return eid with no consider edge feature (not exist)
        for uv in common_uvs:
            if len(g1_uv_to_eid[uv]) != 1 or len(g2_uv_to_eid[uv]) != 1:
                raise ValueError(
                    f"Edge {uv} has multiple occurrences without feature differentiation"
                )
        overlap_eids = [(g1_uv_to_eid[uv], g2_uv_to_eid[uv]) for uv in common_uvs]

    else:
        # Find common uv with same features
        overlap_eids = []
        for uv in common_uvs:
            g1_uv_eids = g1_uv_to_eid[uv]
            g2_uv_eids = g2_uv_to_eid[uv]

            equal_feat_mask = torch.all(
                g1_feats[g1_uv_eids][:, None, :] == g2_feats[g2_uv_eids][None, :, :],  # type:ignore
                dim=2,
            )
            overlap_eids.extend(
                (g1_uv_eids[i], g2_uv_eids[j])
                for i, j in torch.nonzero(equal_feat_mask).tolist()
            )

    g1_eids: Tensor = g1.edges("eid")
    g2_eids: Tensor = g2.edges("eid")
    overlap_eids = torch.Tensor(overlap_eids)
    g1_non_overlap_eid = g1_eids[~torch.isin(g1_eids, overlap_eids[:, 0])]
    g2_non_overlap_eid = g2_eids[~torch.isin(g2_eids, overlap_eids[:, 1])]

    return overlap_eids, g1_non_overlap_eid, g2_non_overlap_eid


def find_node_overlap(g1: DGLGraph, g2: DGLGraph) -> tuple[Tensor, Tensor, Tensor]:
    g1_nids: Tensor = find_nodes(g1)
    g2_nids: Tensor = find_nodes(g2)

    # Find common nids
    cat_nids, counts = torch.cat((g1_nids, g2_nids)).unique(return_counts=True)
    common_nids = cat_nids[counts > 1]

    # Get feature of common nodes
    g1_common_feature = g1.ndata["feature"][common_nids]
    g2_common_feature = g2.ndata["feature"][common_nids]

    # Check feature of common nid is same between g1 & g2
    equal_mask = torch.all(g1_common_feature == g2_common_feature, dim=1)

    overlap_nids = common_nids[equal_mask]
    g1_non_overlap_nids = g1_nids[~torch.isin(g1_nids, overlap_nids)]
    g2_non_overlap_nids = g2_nids[~torch.isin(g2_nids, overlap_nids)]

    return overlap_nids, g1_non_overlap_nids, g2_non_overlap_nids


def find_overlap_aggregation(
    g1: DGLGraph,
    g2: DGLGraph,
    non_overlap_eids: tuple[Tensor, Tensor],
    non_overlap_nids: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    g2_nids = find_nodes(g2)
    # * Find updated aggregation result by edge
    #   (result of dst of a new/removed/modified edge will change)
    _, g1_v = g1.edges()
    _, g2_v = g2.edges()
    # g1_uv = torch.stack((g1_u, g1_v), dim=1)
    # g2_uv = torch.stack((g2_u, g2_v), dim=1)

    updated_nid_by_edge = torch.cat((
        g1_v[non_overlap_eids[0]],
        g2_v[non_overlap_eids[1]],
    ))

    # * Find updated aggregation result by embedding
    #   (result of dst of out edge of new/removed/modified node will change)
    updated_nid_by_node = torch.cat([
        g1.out_edges(non_overlap_nids[0])[1],
        g2.out_edges(non_overlap_nids[1])[1],
    ])

    updated_nids = torch.cat((updated_nid_by_edge, updated_nid_by_node)).unique()
    non_updated_nids = g2_nids[~torch.isin(g2_nids, updated_nids)]

    return non_updated_nids, updated_nids


def find_amount_data_transfer(
    g: DGLGraph,
    non_overlap_eids: tuple[Tensor, Tensor],
    non_overlap_nids: tuple[Tensor, Tensor],
    bytes: int = 8,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    # Find amount of edge transfer, without consider format of data
    num_nodes = find_num_nodes(g)
    num_edges = len(g.edges("eid"))

    # Topology
    total_edge_transfer = num_edges * bytes
    only_extra_edge_transfer = (
        len(non_overlap_eids[0]) + len(non_overlap_eids[1])
    ) * bytes

    # Edge feature
    if g.edata.get("feature") is not None:
        edge_feat_size = g.edata["feature"].shape[1]  # type:ignore

        total_edge_feat_transfer = num_edges * edge_feat_size * bytes
        only_extra_edge_feat_transfer = (
            (len(non_overlap_eids[0]) + len(non_overlap_eids[1]))
            * edge_feat_size
            * bytes
        )
    else:
        total_edge_feat_transfer = 0
        only_extra_edge_feat_transfer = 0

    # Node feature
    node_feat_size = g.ndata["feature"].shape[1]  # type:ignore
    total_node_feat_transfer = num_nodes * node_feat_size * bytes
    only_extra_node_feat_transfer = len(non_overlap_nids[1]) * node_feat_size * bytes

    return (total_edge_transfer, total_edge_feat_transfer, total_node_feat_transfer), (
        only_extra_edge_transfer,
        only_extra_edge_feat_transfer,
        only_extra_node_feat_transfer,
    )


def find_amount_computes(g: DGLGraph, non_overlap_agg_nid: Tensor) -> tuple[int, int]:
    # Computes = number of nonzero * feature size
    num_edges = len(g.edges("eid"))
    node_feat_size = g.ndata["feature"].shape[1]  # type:ignore
    edge_feat_size = (
        0 if (efeat_size := g.edata.get("feature")) is None else efeat_size.shape[1]
    )

    amount_total_computes = num_edges * (node_feat_size + edge_feat_size) * 2
    amount_only_extra_computes = (
        g.in_degrees(non_overlap_agg_nid).sum().item()  # type:ignore
        * (node_feat_size + edge_feat_size)
        * 2
    )

    return amount_total_computes, amount_only_extra_computes
