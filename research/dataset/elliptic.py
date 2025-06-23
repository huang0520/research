from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import override

import polars as pl
from polars import selectors as cs
from torch_geometric.data import Data, HeteroData

from research.dataset.base import BaseDataset
from research.transform.edge_life import edge_life
from research.utils import download_google


@dataclass
class GoogleFileID:
    tx_features: str = "19q09IFhfkOOBOXvn_dKhWjILJtjCcsjc"
    tx_classes: str = "1DiBxn8TXdbJqoSw58pYUeaqO3oOKhuQO"
    addr_features: str = "1mhrrobYdnaxYBIVK06EpKB2EwV-ERlce"
    addr_classes: str = "1ZaACVE4wSIx7r8Z9ze7ExQnJ0wzGrVkp"
    tx_tx_edges: str = "1Q2yG_CIDvfdGP-fKVPSw979EYgQukjz5"
    tx_addr_edges: str = "1SYpun0DMDt-h3sbI60G55jNsfwnFTHXg"
    addr_tx_edges: str = "16eWZTe-dsjEqgsOqFtg78m4fYhcxKlmZ"
    addr_addr_edges: str = "1x5JdNWX8fVM3FO8I0wOLDAeh2qaB53uV"


class EllipticTxTx(BaseDataset):
    @override
    def __init__(
        self,
        root: str = "./data/elliptic-txtx",
        force_reload: bool = False,
        incremental: bool = True,
        incremental_threshold: float = 0.2,
        only_edge: bool = False,
        **kwargs,
    ) -> None:
        self.raw_edge_file_name = "txs_edgelist.csv"
        self.raw_feat_file_name = "txs_features.csv"
        self.raw_label_file_name = "txs_classes.csv"
        super().__init__(
            root=root,
            force_reload=force_reload,
            incremental=incremental,
            incremental_threshold=incremental_threshold,
            only_edge=only_edge,
            **kwargs,
        )
        self.load(self.processed_paths[0])
        self._reset_cache()

    @override
    def process(self) -> None:
        raw_dir = Path(self.raw_dir)
        df_edges = pl.read_csv(raw_dir / self.raw_edge_file_name)
        df_feats = pl.read_csv(raw_dir / self.raw_feat_file_name)
        df_labels = pl.read_csv(raw_dir / self.raw_label_file_name)

        # Map ID
        id_map = {
            id: new_id
            for new_id, id in enumerate(df_feats["txId"].unique(maintain_order=True))
        }

        df_edges = df_edges.with_columns(pl.col("txId1", "txId2").replace(id_map))
        df_feats = df_feats.with_columns(pl.col("txId").replace(id_map))
        df_labels = df_labels.with_columns(pl.col("txId").replace(id_map))

        # Extract snapshot masks
        step_range = range(df_feats["Time step"].min(), df_feats["Time step"].max() + 1)  # type:ignore
        df_feats = df_feats.with_columns(**{
            f"mask_{step - 1}": pl.col("Time step") == step for step in step_range
        })

        id_masked = {
            f"mask_{step - 1}": df_feats.filter(f"mask_{step - 1}")["txId"]
            for step in step_range
        }
        df_edges = df_edges.with_columns(**{
            f"mask_{step - 1}": pl.col("txId1").is_in(id_masked[f"mask_{step - 1}"])
            & pl.col("txId2").is_in(id_masked[f"mask_{step - 1}"])
            for step in step_range
        })

        feats = (
            df_feats.select(cs.exclude(cs.starts_with("mask"), "txId", "Time step"))
            .fill_null(0)
            .to_torch(dtype=pl.Float32)
        )
        edges = df_edges.select(cs.starts_with("txId")).to_torch().T
        nmasks = df_feats.select(cs.starts_with("mask")).to_torch().T
        emasks = df_edges.select(cs.starts_with("mask")).to_torch().T

        data = HeteroData()
        data["n"].x = feats
        data["n"].mask = nmasks
        data["n", "e", "n"].edge_index = edges
        data["n", "e", "n"].mask = emasks

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    @override
    def download(self):
        download_google(
            GoogleFileID.tx_tx_edges, Path(self.raw_dir) / self.raw_edge_file_name
        )
        download_google(
            GoogleFileID.tx_features, Path(self.raw_dir) / self.raw_feat_file_name
        )
        download_google(
            GoogleFileID.tx_classes, Path(self.raw_dir) / self.raw_label_file_name
        )

    @override
    def raw_file_names(self) -> tuple[str, ...]:
        return (
            self.raw_edge_file_name,
            self.raw_feat_file_name,
            self.raw_label_file_name,
        )

    @override
    def processed_file_names(self) -> str:
        return "data.pt"


if __name__ == "__main__":
    pre_transform = partial(edge_life, life=7)
    EllipticTxTx(force_reload=True, pre_transform=(edge_life))
