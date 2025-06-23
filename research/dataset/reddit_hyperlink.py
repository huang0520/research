from functools import partial
from pathlib import Path
from typing import override

import polars as pl
import polars.selectors as cs
from torch_geometric.data.hetero_data import HeteroData

from research.dataset import BaseDataset
from research.transform import edge_life
from research.utils.download import download_url

EDGE_URL = "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"
NODE_URL = "https://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"


class RedditBody(BaseDataset):
    @override
    def __init__(
        self,
        root: str = "./data/reddit-body",
        force_reload: bool = False,
        incremental: bool = True,
        incremental_threshold: float = 0.2,
        only_edge: bool = False,
        **kwargs,
    ) -> None:
        self.raw_edge_file_name = "soc-redditHyperlinks-body.tsv"
        self.raw_node_file_name = "web-redditEmbeddings-subreddits.csv"
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
    def process(self):  # noqa: PLR0914
        raw_dir = Path(self.raw_dir)
        df_edges = pl.read_csv(
            raw_dir / self.raw_edge_file_name, separator="\t"
        ).with_columns(pl.col("TIMESTAMP").str.to_datetime("%Y-%m-%d %H:%M:%S"))
        df_nodes = pl.read_csv(
            raw_dir / self.raw_node_file_name,
            has_header=False,
            new_columns=["subreddit"],
        )

        # Extract used subreddit as categories
        subreddit_enum = pl.Enum(
            pl.concat(df_edges.select("SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"))
            .unique()
            .sort()
        )

        # Extract node feature
        # Using mean value as the missing embedding (from ROLAND)
        node_mean_feat: list = (
            df_nodes.select(pl.concat_list(cs.exclude("subreddit").mean()))
            .item()
            .to_list()
        )

        df_node_ir = (
            pl.DataFrame(subreddit_enum.categories, schema=["subreddit"])
            .join(
                df_nodes.select(
                    "subreddit",
                    pl.concat_list(cs.exclude("subreddit")).alias("feat"),
                ),
                on="subreddit",
                how="left",
            )
            .select(
                pl.col("subreddit").cast(subreddit_enum).to_physical().alias("nid"),
                pl.col("feat")
                .fill_null(node_mean_feat)
                .cast(pl.Array(pl.Float32, shape=len(node_mean_feat))),
            )
        )

        # Create intermediate dataframe
        # Transform edge feat
        df_edge_ir = df_edges.select(
            # Subreddit name to subreddit id
            pl.col("SOURCE_SUBREDDIT")
            .cast(subreddit_enum)
            .alias("src_nid")
            .to_physical(),
            pl.col("TARGET_SUBREDDIT")
            .cast(subreddit_enum)
            .alias("dst_nid")
            .to_physical(),
            pl.col("PROPERTIES").str.split(",").cast(pl.List(pl.Float32)).alias("feat"),
            pl.col("LINK_SENTIMENT").alias("label"),
            # Time stamp to time step (1 week interval)
            ((pl.col("TIMESTAMP") - pl.col("TIMESTAMP").min()) / pl.duration(weeks=1))
            .floor()
            .cast(pl.Int32)
            .alias("timestep"),
        ).with_row_index("eid")

        # List to Array
        edge_feat_size = df_edge_ir.select(pl.col("feat").list.len()).max().item()
        df_edge_ir = df_edge_ir.with_columns(
            pl.col("feat").cast(pl.Array(pl.Float32, shape=edge_feat_size))
        )

        # Create edge snapshot mask
        max_step = df_edge_ir.select("timestep").max().item()
        df_edge_ir = df_edge_ir.with_columns(
            (pl.col("timestep") == i).alias(f"mask_{i}") for i in range(max_step)
        )

        # Extract node indices of each snapshot
        node_indices = (
            df_edge_ir.filter(f"mask_{i}")
            .unpivot(("src_nid", "dst_nid"))
            .select(pl.col("value").unique())
            .to_series()
            .to_list()
            for i in range(max_step)
        )

        # Create node snapshot mask
        df_node_ir = df_node_ir.with_columns(
            pl.col("nid").is_in(indices).alias(f"mask_{i}")
            for i, indices in enumerate(node_indices)
        )

        nfeats = df_node_ir["feat"].to_torch()
        efeats = df_edge_ir["feat"].to_torch()
        edges = df_edge_ir.select(cs.ends_with("nid")).to_torch().T
        nmasks = df_node_ir.select(cs.starts_with("mask")).to_torch().T
        emasks = df_edge_ir.select(cs.starts_with("mask")).to_torch().T

        data = HeteroData()
        data["n"].x = nfeats
        data["n"].mask = nmasks
        data["n", "e", "n"].edge_index = edges
        data["n", "e", "n"].edge_attr = efeats
        data["n", "e", "n"].mask = emasks

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    @override
    def download(self):
        download_url(EDGE_URL, Path(self.raw_dir) / self.raw_edge_file_name)
        download_url(NODE_URL, Path(self.raw_dir) / self.raw_node_file_name)

    @override
    def raw_file_names(self) -> tuple[str, ...]:
        return (self.raw_edge_file_name, self.raw_node_file_name)

    @override
    def processed_file_names(self) -> str:
        return "data.pt"


if __name__ == "__main__":
    pre_transform = partial(edge_life, life=20)
    RedditBody(force_reload=True, pre_transform=pre_transform)
