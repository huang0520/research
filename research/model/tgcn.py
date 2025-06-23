import torch as th
from torch import Tensor, nn
from torch_geometric.nn.conv.gcn_conv import GCNConv

from research.dataset.base import ComputeInfo


class TGCN(nn.Module):
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        gcn_in: int,
        gcn_out: int,
        rnn_hidden: int,
        gcn_norm: bool = True,
        gcn_bias: bool = True,
        rnn_n_layer: int = 1,
        rnn_bias: bool = True,
        cache_gconv: bool = False,
        max_nodes: int = 0,
    ):
        super().__init__()
        self.gconv = GCNConv(gcn_in, gcn_out, normalize=gcn_norm, bias=gcn_bias)
        self.gru = nn.GRU(
            input_size=gcn_out,
            hidden_size=rnn_hidden,
            num_layers=rnn_n_layer,
            bias=rnn_bias,
            batch_first=True,
        )

        h0 = th.randn((rnn_n_layer, 1, rnn_hidden))
        self.register_buffer("initial_hidden", h0)

        self.cache_gconv = cache_gconv
        self.max_nodes = max_nodes

        if self.cache_gconv:
            if max_nodes > 0:
                self.register_buffer("gconv_cache", th.zeros(max_nodes, gcn_out))
            else:
                self.gconv_cache = None
            self.cache_valid = False
        else:
            self.gconv_cache = None
            self.cache_valid = False

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        hidden: Tensor | None = None,
        compute_info: ComputeInfo | None = None,
    ) -> tuple[Tensor, Tensor]:
        if hidden is None:
            hidden = self.initial_hidden

        if (
            self.cache_gconv
            and self.cache_valid
            and compute_info is not None
            and compute_info.use_cache
            and compute_info.keep_curr_lrid.nelement() != 0
        ):
            edge_index_ = edge_index[:, compute_info.compute_leids]
            x = self.gconv(x, edge_index_)

            x[compute_info.keep_curr_lrid] = self.gconv_cache[
                compute_info.keep_prev_lrid
            ]
        else:
            x = self.gconv(x, edge_index)

        if self.cache_gconv:
            if self.max_nodes:
                self.gconv_cache[: x.size(0)] = x
            elif self.gconv_cache is None or self.gconv_cache.size(0) < x.size(0):
                self.gconv_cache = x
            else:
                self.gconv_cache[: x.size(0)] = x

            self.cache_valid = True

        x = x.view([1, *x.shape])
        return self.gru(x, hidden)

    def reset_cache(self):
        """Reset cache state"""
        self.cache_valid = False
        if self.gconv_cache is not None:
            self.gconv_cache.zero_()
