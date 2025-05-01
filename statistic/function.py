from collections import defaultdict

import dgl
import polars as pl
import polars.selectors as cs
import torch
from dgl import DGLGraph
from torch import Tensor

from research.dataset.base import BaseDataset


def find_nodes(g: DGLGraph) -> Tensor:
    return torch.cat(g.edges()).unique()


def find_num_nodes(g: DGLGraph) -> int:
    return len(find_nodes(g))


def find_edge_overlap_(dataset: BaseDataset, g1_idx: int, g2_idx: int):
    g1 = dataset[g1_idx]
    g2 = dataset[g2_idx]

    # Find orignal nid/eid to new nid/eid map
    g1_nid_map = {orig: new for new, orig in enumerate(g1.ndata[dgl.NID].tolist())}
    g1_eid_map = {orig: new for new, orig in enumerate(g1.edata[dgl.EID].tolist())}
    g2_nid_map = {orig: new for new, orig in enumerate(g2.ndata[dgl.NID].tolist())}
    g2_eid_map = {orig: new for new, orig in enumerate(g2.edata[dgl.EID].tolist())}

    # Filter dataframe will use
    lf_ir = dataset.lf_edges.select(
        cs.exclude(cs.starts_with("mask_")),
        pl.col(f"mask_{g1_idx}").alias("mask_g1"),
        pl.col(f"mask_{g2_idx}").alias("mask_g2"),
    ).filter(pl.any_horizontal("mask_g1", "mask_g2"))

    # Transform nid, eid to snapshot nid, eid
    lf_ir = lf_ir.with_columns(
        g1_src_nid=pl.col("src_nid").replace(g1_nid_map),
        g1_dst_nid=pl.col("dst_nid").replace(g1_nid_map),
        g2_src_nid=pl.col("src_nid").replace(g2_nid_map),
        g2_dst_nid=pl.col("dst_nid").replace(g2_nid_map),
        g1_eid=pl.col("eid").replace(g1_eid_map),
        g2_eid=pl.col("eid").replace(g2_eid_map),
    ).drop("eid")

    # Find common edge created by edge-life
    lf_ir = lf_ir.with_columns(mask_common=pl.all_horizontal("mask_g1", "mask_g2"))

    common_eid_pairs = lf_ir.filter("mask_common").select(
        eid_pair=pl.concat_list("g1_eid", "g2_eid")
    )

    # duplicated_index = ("src_nid", "dst_nid", "feat")
    # duplicated_eids = (
    #     lf_ir.filter(~pl.col("mask_common"))
    #     .group_by(duplicated_index)
    #     .agg(pl.len(), pl.col("eid", "mask_g1", "mask_g2"))
    #     .filter(
    #         # Find the edges with same (u, v, feat)
    #         pl.col("len") > 1,
    #         # Drop the edges that all from same graph,
    #         # e.g. e5, e7 is same but both at g1
    #         pl.all_horizontal(pl.col("mask_g1", "mask_g2").list.any()),
    #     )
    #     .drop(*duplicated_index, "len")
    #     .explode("*")
    #     # If number of edges is odd, drop additional ones
    #     .group_by("mask_g1", "mask_g2")
    #     .agg(pl.col("eid"), pl.len())
    #     .with_columns(pl.col("eid").list.sample(pl.min("len")))
    #     .drop("mask_g1", "mask_g2", "len")
    #     # .with_columns(pl.lit(True).alias("mask_duplicated"))
    #     # .explode("eid")
    # )

    # Find non-common edges
    g1_ext_edges = lf_ir.filter(~pl.col("mask_g2")).drop(
        "mask_g1", "mask_g2", "g2_src_nid", "g2_dst_nid", "g2_eid"
    )
    g2_ext_edges = lf_ir.filter(~pl.col("mask_g1")).drop(
        "mask_g1", "mask_g2", "g1_src_nid", "g1_dst_nid", "g1_eid"
    )

    overlap_eid_pairs = (
        g1_ext_edges.join(g2_ext_edges, on=("src_nid", "dst_nid", "feat"))
        .select("g1_eid", "g2_eid")
        .with_row_index()
        .collect()
    )

    selected_idx = []
    if not overlap_eid_pairs.is_empty():
        g1_selected_eids = set()
        g2_selected_eids = set()
        for i, g1_eid, g2_eid in overlap_eid_pairs.iter_rows():
            if g1_eid not in g1_selected_eids and g2_eid not in g2_selected_eids:
                selected_idx.append(i)
                g1_selected_eids.add(g1_eid)
                g2_selected_eids.add(g2_eid)

        overlap_eid_pairs = overlap_eid_pairs[selected_idx, ["g1_eid", "g2_eid"]]

    g1_ext_eids = (
        g1_ext_edges.select("g1_eid")
        .join(overlap_eid_pairs.lazy().select("g1_eid"), on="g1_eid", how="anti")
        .collect()
    )
    g2_ext_eids = (
        g2_ext_edges.select("g2_eid")
        .join(overlap_eid_pairs.lazy().select("g2_eid"), on="g2_eid", how="anti")
        .collect()
    )

    overlap_eid_pairs = (
        common_eid_pairs.collect().to_series().to_list()
        + overlap_eid_pairs.select(pl.concat_list("*")).to_series().to_list()
    )

    breakpoint()


def find_edge_overlap(g1: DGLGraph, g2: DGLGraph) -> tuple[Tensor, Tensor, Tensor]:
    # Find common edges with same u, v between g1, g2
    g1_uv_to_eid: dict[tuple[int, int], list[int]] = defaultdict(list)
    g2_uv_to_eid: dict[tuple[int, int], list[int]] = defaultdict(list)

    for eid, (i, j) in enumerate(torch.stack(g1.edges(), dim=1).tolist()):
        g1_uv_to_eid[i, j].append(eid)
    for eid, (i, j) in enumerate(torch.stack(g2.edges(), dim=1).tolist()):
        g2_uv_to_eid[i, j].append(eid)

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

            # Handle multi-edge situation that creat 1 to many common edges
            # Like g1 has (n0, 5, 5, 3), (n1, 5, 5, 4)
            # and g2 also has (m0, 5, 5, 3) and (m1, 5, 5, 4)
            # It will create 4 common edges, but we only need 2
            selected_i = set()
            selected_j = set()
            for i, j in torch.nonzero(equal_feat_mask).tolist():
                if i in selected_i or j in selected_j:
                    continue

                selected_i.add(i)
                selected_j.add(j)
                overlap_eids.append((g1_uv_eids[i], g2_uv_eids[j]))

    g1_eids: Tensor = g1.edges("eid")
    g2_eids: Tensor = g2.edges("eid")
    overlap_eids = torch.Tensor(overlap_eids)
    g1_non_overlap_eids = g1_eids[~torch.isin(g1_eids, overlap_eids[:, 0])]
    g2_non_overlap_eids = g2_eids[~torch.isin(g2_eids, overlap_eids[:, 1])]

    return overlap_eids, g1_non_overlap_eids, g2_non_overlap_eids


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
