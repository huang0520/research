import torch as th
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.typing import EdgeType
from torch_geometric.utils.map import map_index

from research.base import MainData, SubData


def edge_subgraph(
    data: HeteroData, eid: th.Tensor, etype: EdgeType | None = None
) -> HeteroData:
    if etype is None and len(data.edge_types) != 1:
        raise ValueError()

    etype = etype if etype is not None else data.edge_types[0]
    edge_index_subset: th.Tensor = data[etype]["edge_index"][:, eid]

    subdata = HeteroData()
    if etype[0] == etype[2]:
        # Same src, dst node type
        nid = edge_index_subset.view(-1).unique(sorted=False)
        edge_index, _ = map_index(
            edge_index_subset.view(-1), nid, nid.max(), inclusive=True
        )
        edge_index = edge_index.view(2, -1)

        subdata[etype[0]]["x"] = data[etype[0]]["x"][nid]
    else:
        # Different src, dst node type
        src_nid: th.Tensor = edge_index_subset[0].unique(sorted=False)
        dst_nid: th.Tensor = edge_index_subset[1].unique(sorted=False)

        src_index, _ = map_index(
            edge_index_subset[0], src_nid, src_nid.max(), inclusive=True
        )
        dst_index, _ = map_index(
            edge_index_subset[1], dst_nid, dst_nid.max(), inclusive=True
        )
        edge_index = th.zeros_like(edge_index_subset)
        edge_index[0] = src_index
        edge_index[1] = dst_index

        subdata[etype[0]]["x"] = data[etype[0]]["x"][src_nid]
        subdata[etype[1]]["x"] = data[etype[1]]["x"][dst_nid]

    subdata[etype]["edge_index"] = edge_index

    if hasattr(data, "edge_attr") and data[etype]["edge_attr"] is not None:
        subdata[etype]["edge_attr"] = data[etype]["edge_attr"][eid]

    return subdata.contiguous()
