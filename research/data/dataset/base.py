from abc import ABC, abstractmethod
from pathlib import Path

import dgl
import polars as pl
from dgl.data import DGLDataset
from dgl.heterograph import DGLGraph
from torch_geometric.data import InMemoryDataset

from research.data.base import MainData


class BaseDataset(InMemoryDataset):
    _data: MainData


class BaseDataset_(DGLDataset, ABC):
    def __init__(self, *args, **kwargs):
        self._graph = dgl.graph(([0], [0]))
        self._df_nodes = pl.DataFrame()
        self._df_edges = pl.DataFrame()
        self._num_snapshots = 0
        super().__init__(*args, **kwargs)

    @property
    def graph(self):
        return self._graph

    @property
    def df_nodes(self) -> pl.DataFrame:
        return self._df_nodes

    @property
    def df_edges(self) -> pl.DataFrame:
        return self._df_edges

    @property
    def lf_nodes(self) -> pl.LazyFrame:
        return self._df_nodes.lazy()

    @property
    def lf_edges(self) -> pl.LazyFrame:
        return self._df_edges.lazy()

    @abstractmethod
    def __getitem__(self, idx) -> DGLGraph:
        pass

    def __len__(self):
        return self._num_snapshots

    def save(self):
        self.df_nodes.write_parquet(self.df_nodes_path)
        self.df_edges.write_parquet(self.df_edges_path)

    def _create_dgl_graph(self):
        src = self.df_edges["src_nid"].to_torch()
        dst = self.df_edges["dst_nid"].to_torch()
        num_nodes = self.df_nodes.select(pl.len()).item()

        graph = dgl.graph((src, dst), num_nodes=num_nodes)
        graph.ndata["feat"] = self.df_nodes["feat"].to_torch()
        graph.edata["feat"] = self.df_edges["feat"].to_torch()
        graph.edata["label"] = self.df_edges["label"].to_torch()

        return graph

    @property
    def raw_dir(self):
        return Path(self._raw_dir).absolute() / self.name

    @property
    def df_nodes_path(self):
        return self.raw_dir / "df_nodes.parquet"

    @property
    def df_edges_path(self):
        return self.raw_dir / "df_edges.parquet"

    def has_cache(self):
        return self.df_nodes_path.exists() and self.df_edges_path.exists()
