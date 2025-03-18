from copy import deepcopy

from research.dataset.base import BaseDataset


def edge_life(dataset: BaseDataset, life: int = 2) -> BaseDataset:
    dataset_ = deepcopy(dataset)

    for t in range(len(dataset)):  # type:ignore
        for i in range(life):
            if t - i < 0:
                continue

            dataset_.snapshot_masks[t] |= dataset.snapshot_masks[t - i]

    return dataset_
