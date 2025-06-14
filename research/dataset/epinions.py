from pathlib import Path
from typing import override
from zipfile import ZipFile

import polars as pl
import polars.selectors as cs
import torch as th
from torch_geometric.data import HeteroData

from research.dataset import BaseDataset
from research.utils.download import download_url

RAW_URL = "https://nrvis.com/download/data/dynamic/rec-epinions-user-ratings.zip"


class Epinions(BaseDataset):
    @override
    def __init__(
        self,
        root: str = "./data/epinions",
        force_reload: bool = False,
    ) -> None:
        self.raw_edge_name = "rec-epinions-user-ratings.csv"
        super().__init__(root=root, force_reload=force_reload)
        self.load(self.processed_paths[0])
        self._data = self._data.pin_memory()
        self._reset_cache()

    @override
    def process(self):
        df_edges = pl.read_csv(
            Path(self.raw_dir) / self.raw_edge_name,
            has_header=False,
            separator=" ",
            new_columns=["user", "product", "rating", "timestamp"],
        )

        n_users: int = df_edges.select("user").max().item()
        n_products: int = df_edges.select("product").max().item()

        # Extract edge snapshot masks
        timestamps: list[int] = df_edges["timestamp"].unique().sort().to_list()
        df_edges = df_edges.with_columns(**{
            f"mask_{i}": pl.col("timestamp") == stamp
            for i, stamp in enumerate(timestamps)
        })
        emasks = df_edges.select(cs.starts_with("mask")).to_torch().T

        # Extract node snapshot masks
        umasks = th.zeros((emasks.shape[0], n_users))
        pmasks = th.zeros((emasks.shape[0], n_products))

        for i in range(emasks.shape[0]):
            u, p = df_edges.filter(f"mask_{i}")["user", "product"]
            umasks[i][u.to_torch() - 1] = True
            pmasks[i][p.to_torch() - 1] = True

        edges = df_edges.select("user", "product").to_torch().T
        efeats = df_edges["rating"].to_torch()

        data = HeteroData()
        data["user"].x = th.randn((n_users, 2))
        data["user"].mask = umasks
        data["product"].x = th.randn((n_products, 2))
        data["product"].mask = pmasks
        data["user", "rating", "product"].edge_index = edges
        data["user", "rating", "product"].edge_attr = efeats
        data["user", "rating", "product"].mask = emasks

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    @override
    def download(self) -> None:
        zip_file = Path(self.raw_dir) / "raw.zip"
        download_url(RAW_URL, zip_file)

        with ZipFile(zip_file, "r") as zfile:
            zfile.extractall(self.raw_dir)

        # Transform file to csv
        with (
            Path(self.raw_dir) / f"{Path(self.raw_edge_name).stem}.edges"
        ).open() as f:
            lines = f.readlines()[1:]

        with (Path(self.raw_dir) / self.raw_edge_name).open("w") as f:
            f.writelines(lines)

    @override
    def raw_file_names(self):
        return self.raw_edge_name

    @override
    def processed_file_names(self) -> str:
        return "data.pt"


if __name__ == "__main__":
    Epinions(force_reload=True)
