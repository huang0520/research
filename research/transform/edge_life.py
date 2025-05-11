import torch as th

from research.dataset import BaseDataset


def edge_life(dataset: BaseDataset, life: int = 2) -> BaseDataset:
    orig_nmask = dataset._data.nmasks
    orig_emask = dataset._data.emasks
    nmask_ = th.zeros_like(orig_nmask)
    emask_ = th.zeros_like(orig_emask)

    assert orig_nmask.size(0) == orig_emask.size(0)

    for i in range(orig_nmask.size(0)):
        start = max(0, i - life)
        nmask_[i] = orig_nmask[start : i + 1].any(dim=0)
        emask_[i] = orig_emask[start : i + 1].any(dim=0)

    dataset._data.nmasks = nmask_
    dataset._data.emasks = emask_

    return dataset
