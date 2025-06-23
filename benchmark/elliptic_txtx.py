from functools import partial

from benchmark.base import DatasetConfig
from research.dataset import EllipticTxTx
from research.transform import edge_life

pre_transform = partial(edge_life, life=7)

no_reduce = DatasetConfig(
    name="EllipticTxTx_NoReduce",
    dataset_fn=lambda: EllipticTxTx(incremental=False, pre_transform=(edge_life)),
)
reduce_edge = DatasetConfig(
    name="EllipticTxTx_ReduceEdge",
    dataset_fn=lambda: EllipticTxTx(
        incremental=True, only_edge=True, pre_transform=(edge_life)
    ),
)
reduce_both = DatasetConfig(
    name="EllipticTxTx_ReduceBoth",
    dataset_fn=lambda: EllipticTxTx(incremental=True, pre_transform=(edge_life)),
)
