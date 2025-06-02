from torch_geometric.data import InMemoryDataset

from research.base import MainData


class BaseDataset(InMemoryDataset):
    _data: MainData
