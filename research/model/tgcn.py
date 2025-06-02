import torch as th
from torch import Tensor, nn

from research.model.layer import CacheableGCNConv


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
    ):
        super().__init__()
        self.gconv = CacheableGCNConv(
            gcn_in, gcn_out, normalize=gcn_norm, bias=gcn_bias
        )
        self.gru = nn.GRU(
            input_size=gcn_out,
            hidden_size=rnn_hidden,
            num_layers=rnn_n_layer,
            bias=rnn_bias,
            batch_first=True,
        )

        h0 = th.randn((rnn_n_layer, 1, rnn_hidden))
        self.register_buffer("initial_hidden", h0)

    def forward(
        self, x: Tensor, edge_index: Tensor, hidden=None
    ) -> tuple[Tensor, Tensor]:
        if hidden is None:
            hidden = self.initial_hidden
            self.gru.flatten_parameters()

        x = self.gconv(x, edge_index)
        x = x.view([1, *x.shape])
        return self.gru(x, hidden)
