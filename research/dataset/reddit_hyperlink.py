import dgl
import polars as pl
import polars.selectors as cs

from research.dataset import BaseDataset
from research.utils.download import download_url

EDGE_URL = "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"
NODE_URL = "https://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"


class RedditBodyDataset(BaseDataset):
    def __init__(self, save_dir="./data", force_reload=False):
        self._edge_url = EDGE_URL
        self._node_url = NODE_URL
        super().__init__(
            name="reddit-body", raw_dir=save_dir, force_reload=force_reload
        )

    def process(self):
        df_edges = pl.read_csv(
            self.raw_edge_path,
            separator="\t",
        ).with_columns(pl.col("TIMESTAMP").str.to_datetime("%Y-%m-%d %H:%M:%S"))
        df_nodes = pl.read_csv(
            self.raw_node_path, has_header=False, new_columns=["subreddit"]
        )

        # Extract used subreddit as categories
        subreddit_enum = pl.Enum(
            pl.concat(df_edges.select("SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"))
            .unique()
            .sort()
        )

        # Extract node feature
        # Using mean value as the missing embedding (from ROLAND)
        node_mean_feat = (
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

        self._df_nodes = df_node_ir
        self._df_edges = df_edge_ir.drop("timestep")
        self._num_snapshots = self._df_nodes.select(cs.starts_with("mask_")).width

        self._graph = self._create_dgl_graph()

    def __getitem__(self, idx):
        if idx >= self._num_snapshots:
            raise IndexError
        return dgl.edge_subgraph(self.graph, self.df_edges[f"mask_{idx}"].to_torch())

    def load(self):
        self._df_nodes = pl.read_parquet(self.df_nodes_path)
        self._df_edges = pl.read_parquet(self.df_edges_path)
        self._num_snapshots = self._df_nodes.select(cs.starts_with("mask_")).width
        self._graph = self._create_dgl_graph()

    def download(self):
        download_url(self._edge_url, self.raw_edge_path)
        download_url(self._node_url, self.raw_node_path)

    def _download(self):
        if self.raw_edge_path.exists() and self.raw_node_path.exists():
            return

        self.raw_dir.mkdir(exist_ok=True)
        self.download()

    @property
    def raw_edge_path(self):
        return self.raw_dir / "soc-redditHyperlinks-body.tsv"

    @property
    def raw_node_path(self):
        return self.raw_dir / "web-redditEmbeddings-subreddits.csv"


if __name__ == "__main__":
    dataset = RedditBodyDataset(force_reload=True)

    breakpoint()
