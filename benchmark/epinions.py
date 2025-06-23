from functools import partial

from benchmark.base import DatasetConfig
from research.dataset import Epinions
from research.transform import edge_life

pre_transform = partial(edge_life, life=40)

no_reduce = DatasetConfig(
    name="Epinions_NoReduce",
    dataset_fn=lambda: Epinions(incremental=False, pre_transform=(edge_life)),
)
reduce_edge = DatasetConfig(
    name="Epinions_ReduceEdge",
    dataset_fn=lambda: Epinions(
        incremental=True, only_edge=True, pre_transform=(edge_life)
    ),
)
reduce_both = DatasetConfig(
    name="Epinions_ReduceBoth",
    dataset_fn=lambda: Epinions(incremental=True, pre_transform=(edge_life)),
)
