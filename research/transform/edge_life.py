from copy import deepcopy

import polars as pl

from research.dataset.base import BaseDataset


def edge_life(dataset: BaseDataset, life: int = 2) -> BaseDataset:
    dataset_ = deepcopy(dataset)
    mask_windows = tuple(
        tuple(f"mask_{i}" for i in range(max(0, idx - life + 1), idx + 1))
        for idx in range(len(dataset))
    )
    dataset_._df_nodes = dataset.df_nodes.with_columns(
        pl.any_horizontal(masks).alias(f"mask_{i}")
        for i, masks in enumerate(mask_windows)
    )
    dataset_._df_edges = dataset.df_edges.with_columns(
        pl.any_horizontal(masks).alias(f"mask_{i}")
        for i, masks in enumerate(mask_windows)
    )

    return dataset_
