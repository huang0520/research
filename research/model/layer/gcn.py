import dgl
import torch as th
from dgl import DGLGraph
from dgl import function as fn
from dgl.utils import expand_as_pair
from torch import nn


class GraphConv(nn.Module):
    def __init__(
        self, in_feats, out_feats, bias: bool = True, norm: bool = True
    ) -> None:
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self, graph: DGLGraph, feat: th.Tensor, compute_eid: th.Tensor | None = None
    ):
        feat_src, feat_dst = expand_as_pair(feat, graph)

        graph.srcdata["h"] = feat_src
        # Aggregate all node in the graph
        if compute_eid is None:
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            rst = graph.dstdata["h"]

        # Aggregate node which result will change
        else:
            graph.send_and_recv(compute_eid, fn.copy_u("h", "m"), fn.sum("m", "out"))
            rst = graph.dstdata["out"]

            # TODO: Need to combine previous result with current result

        # Update
        if self.weight is not None:
            rst @= self.weight

        # Normalize
        if self._norm:
            deg: th.Tensor = graph.in_degrees().to(feat_dst).clamp(min=1.0)
            norm = deg**-0.5
            norm = norm.reshape(norm.shape + (1,) * (feat_dst.dim() - 1))
            rst *= norm

        if self.bias is not None:
            rst += self.bias

        return rst

    @staticmethod
    def aggregate(nodes):
        breakpoint()
        return {"h": nodes.data["h"] + nodes.mailbox["m"].sum(1)}
