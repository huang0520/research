from abc import ABC, abstractmethod
from pathlib import Path

import dgl
import torch
from dgl.data import DGLDataset
from dgl.data.graph_serialize import save_graphs
from dgl.heterograph import DGLGraph


class BaseDataset(DGLDataset, ABC):
    def __init__(self, *args, **kwargs):
        self._graph = dgl.graph(([0], [0]))
        self._snapshot_masks = torch.zeros(0)
        super().__init__(*args, **kwargs)

    @property
    def graph(self):
        return self._graph

    @property
    def snapshot_masks(self):
        return self._snapshot_masks

    @property
    @abstractmethod
    def snapshot_masks_path(self) -> Path:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> DGLGraph:
        pass

    def __len__(self):
        return len(self.snapshot_masks)

    def save(self):
        torch.save(self.snapshot_masks, self.snapshot_masks_path)
        save_graphs(str(self.save_path), self.graph)

    @property
    def raw_dir(self):
        return Path(self._raw_dir).absolute() / self.name

    @property
    def save_path(self):
        return self.raw_dir / f"{self.name}.bin"
