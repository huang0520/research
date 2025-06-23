from functools import partial

from benchmark.base import DatasetConfig
from research.dataset import RedditBody
from research.transform import edge_life

pre_transform = partial(edge_life, life=20)

no_reduce = DatasetConfig(
    name="RedditBody_NoReduce",
    dataset_fn=lambda: RedditBody(incremental=False, pre_transform=(edge_life)),
)
reduce_edge = DatasetConfig(
    name="RedditBody_ReduceEdge",
    dataset_fn=lambda: RedditBody(
        incremental=True, only_edge=True, pre_transform=(edge_life)
    ),
)
reduce_both = DatasetConfig(
    name="RedditBody_ReduceBoth",
    dataset_fn=lambda: RedditBody(incremental=True, pre_transform=(edge_life)),
)
