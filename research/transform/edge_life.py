import torch as th
from torch import Tensor

from research.dataset import BaseDataset


def edge_life(dataset: BaseDataset, life: int = 2) -> BaseDataset:
    types = (*dataset._data.node_types, *dataset._data.edge_types)
    for t in types:
        orig_mask: Tensor = dataset._data[t]["mask"]
        mask_ = th.zeros_like(orig_mask)

        for i in dataset.snapshot_ids:
            start = max(0, i - life + 1)
            mask_[i] = orig_mask[start : i + 1].any(dim=0)

        dataset._data[t]["mask"] = mask_

    dataset._reset_cache()
    return dataset
