from dataclasses import dataclass
from pathlib import Path
from typing import override

import dgl
import numpy as np
import pandas as pd
import polars as pl
from dgl.data.graph_serialize import load_graphs
from polars import selectors as cs
from torch_geometric.data import Data

from research.dataset.base import BaseDataset
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
    def __init__(
        self,
        root: str = "./data/elliptic-txtx",
        force_reload: bool = False,
        life: int = 2,
    ) -> None:
        self.raw_edge_file_name = "txs_edgelist.csv"
        self.raw_feat_file_name = "txs_features.csv"
        self.raw_label_file_name = "txs_classes.csv"
        self.life = life
        super().__init__(root=root, force_reload=force_reload)
        self.load(self.processed_paths[0])
        self._data = self._data.pin_memory()

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

        data_list = [
            # snapshot for snapshot in self.edge_life(feats, edges, nmask, emask)
            Data(x=feats, edge_index=edges, nmasks=nmasks, emasks=emasks)
        ]

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


class EllipticTxTx_(BaseDataset):
    def __init__(self, save_dir="./data", force_reload=False):
        self._edge_id = GoogleFileID.tx_tx_edges
        self._node_feature_id = GoogleFileID.tx_features
        self._node_label_id = GoogleFileID.tx_classes

        super().__init__(
            name="eplliptic-txtx", raw_dir=save_dir, force_reload=force_reload
        )

    def process(self):
        df_edges = pd.read_csv(self.raw_edge_path)
        df_node_features = pd.read_csv(self.raw_node_feature_path)
        df_node_label = pd.read_csv(self.raw_node_label_path)

        # Reset node id
        node_id2new_id = {
            node_id: i for i, node_id in enumerate(df_node_label["txId"].sort_values())
        }
        df_edges["txId1"] = df_edges["txId1"].map(node_id2new_id)
        df_edges["txId2"] = df_edges["txId2"].map(node_id2new_id)
        df_node_features["txId"] = df_node_features["txId"].map(node_id2new_id)
        df_node_label["txId"] = df_node_label["txId"].map(node_id2new_id)

        df_node_features = df_node_features.set_index("txId")
        df_node_label = df_node_label.set_index("txId")

        # Extract feature & label
        df_node_features = df_node_features.sort_index()
        node_features = torch.tensor(
            df_node_features.loc[:, "Local_feature_1":].to_numpy()
        )
        df_node_label = df_node_label.sort_index()
        node_label = torch.tensor(df_node_label.to_numpy())

        # Extract snapshot mask
        self._snapshot_masks = torch.tensor(
            np.array([
                np.array(df_node_features["Time step"] == step)
                for step in range(
                    df_node_features["Time step"].min(),  # type:ignore
                    df_node_features["Time step"].max() + 1,  # type:ignore
                )
            ])
        )

        # Create graph
        src, dst = df_edges.to_numpy().transpose()
        self._graph = dgl.graph((src, dst))
        self._graph.ndata["feature"] = node_features
        self._graph.ndata["label"] = node_label

    def download(self):
        download_google(self._edge_id, self.raw_edge_path)
        download_google(self._node_feature_id, self.raw_node_feature_path)
        download_google(self._node_label_id, self.raw_node_label_path)

    def load(self):
        self._snapshot_masks = torch.load(self.snapshot_masks_path, weights_only=True)
        glist, _ = load_graphs(str(self.save_path), [0])
        self._graph = glist[0]

    def _download(self):
        if (
            self.raw_edge_path.exists()
            and self.raw_node_feature_path.exists()
            and self.raw_node_label_path.exists()
        ):
            return

        self.raw_dir.mkdir(exist_ok=True)
        self.download()

    def __getitem__(self, idx):
        return dgl.node_subgraph(
            self.graph, self.snapshot_masks[idx], relabel_nodes=False
        )

    @property
    def raw_edge_path(self):
        return self.raw_dir / "txs_edgelist.csv"

    @property
    def raw_node_feature_path(self):
        return self.raw_dir / "txs_features.csv"

    @property
    def raw_node_label_path(self):
        return self.raw_dir / "txs_classes.csv"


if __name__ == "__main__":
    from dgl.dataloading import GraphDataLoader

    dataset = EllipticTxTx()
    dataloader = GraphDataLoader(dataset)
