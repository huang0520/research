from typing import override

import torch as th
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, SparseTensor

from research.compute import CacheableMixin


class CacheableGCNConv(GCNConv, CacheableMixin):
    @override
    def compute_aggregate(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Tensor | None = None,
        compute_eid: Tensor | None = None,
    ):
        if isinstance(x, (tuple, list)):
            raise ValueError(
                f"'{self.__class__.__name__}' received a tuple "
                f"of node features as input while this layer "
                f"does not support bipartite message passing. "
                f"Please try other layers such as 'SAGEConv' or "
                f"'GraphConv' instead"
            )

        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # type:ignore
                    edge_index,
                    edge_weight,
                    x.size(self.node_dim),
                    self.improved,
                    self.add_self_loops,
                    self.flow,
                    x.dtype,
                )
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(
                    edge_index,  # type:ignore
                    edge_weight,
                    x.size(self.node_dim),
                    self.improved,
                    self.add_self_loops,
                    self.flow,
                    x.dtype,
                )

        x = self.lin(x)

        if compute_eid is None or compute_eid.size(0) == edge_index.size(1):
            out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        else:
            # TODO: Allow sparse tensor
            assert isinstance(edge_index, Tensor)
            filtered_edge_index = edge_index[:, compute_eid]
            dst_nid = filtered_edge_index[1]
            out = th.zeros_like(x)
            out[dst_nid] = self.propagate(
                filtered_edge_index, x=x, edge_weight=edge_weight
            )[dst_nid]

        if self.bias is not None:
            out += self.bias

        return out

    @override
    def compute_update(self, x: Tensor) -> Tensor:
        out = self.lin(x)
        if self.bias is not None:
            out += self.bias

        return out

    @override
    def edge_update(self) -> Tensor:
        raise NotImplementedError
