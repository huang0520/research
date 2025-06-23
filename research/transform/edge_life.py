from pathlib import Path

import torch as th
from torch import Tensor
from torch_geometric.data.hetero_data import HeteroData
from tqdm import tqdm

from research.dataset import BaseDataset


def edge_life(data: HeteroData, life: int = 2) -> HeteroData:
    types = (*data.node_types, *data.edge_types)
    for t in types:
        orig_mask: Tensor = data[t]["mask"]
        mask_ = th.zeros_like(orig_mask)

        for i in range(orig_mask.size(0)):
            start = max(0, i - life + 1)
            mask_[i] = orig_mask[start : i + 1].any(dim=0)

        data[t]["mask"] = mask_
    return data
