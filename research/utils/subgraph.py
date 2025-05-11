import torch as th
from torch_geometric.utils.map import map_index

from research.data import MainData, SubData


def _subgraph(subset: th.Tensor, edge_index: th.Tensor):
    """
    Based on the torch_geometric.utils.subgraph().
    Assume edge_index already mask to requirement. Only need to map the src, dst nid.
    """
    assert subset.dtype != th.bool

    # FIXME: Remove test assertion
    assert th.isin(edge_index.view(-1), subset).all()

    edge_index, _ = map_index(
        edge_index.view(-1), subset, max_index=subset.max(), inclusive=True
    )
    edge_index = edge_index.view(2, -1)

    return edge_index


def edge_subgraph(data: MainData, eid: th.Tensor):
    """
    Args:
    - data: Main graph
    - eid: Subset EID to create subgraph

    return:
    - PyG Data with edges given and associate nodes
    - Original NID of subgraph
    - Original EID of subgraph
    """
    edge_index_subset: th.Tensor = data.edge_index[:, eid]
    nid: th.Tensor = edge_index_subset.view(-1).unique(sorted=False)
    edge_index = _subgraph(nid, edge_index_subset)

    x = data.x[nid]
    data_dict = {"x": x, "edge_index": edge_index, "gnid": nid, "geid": eid}

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data_dict["edge_attr"] = data.edge_attr[eid]

    if hasattr(data, "y") and data.y is not None:
        data_dict["y"] = data.y[nid]  # type:ignore

    return SubData(**data_dict)
