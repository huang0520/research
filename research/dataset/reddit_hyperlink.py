import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs

from research.dataset.base import BaseDataset
from research.utils.download import download_url


class RedditBodyDataset(BaseDataset):
    def __init__(self, save_dir="./data", force_reload=False):
        self._edge_url = "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"
        self._node_url = (
            "https://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"
        )
        super().__init__(
            name="reddit-body", raw_dir=save_dir, force_reload=force_reload
        )

    def process(self):
        df_edges = pd.read_csv(
            self.raw_edge_path,
            sep="\t",
            parse_dates=[3],  # Columns index of TIMESTAMP
            date_format="%Y-%m-%d %H:%M:%S",
        )
        df_nodes = pd.read_csv(self.raw_node_path, header=None, index_col=0)

        # Create subreddit id to node id dict
        subreddit_ids = pd.unique(
            df_edges.loc[:, ("SOURCE_SUBREDDIT", "TARGET_SUBREDDIT")].to_numpy().ravel()
        )
        subreddit_ids = np.sort(subreddit_ids)  # type:ignore
        self.subreddit_cat_type = pd.api.types.CategoricalDtype(
            subreddit_ids, ordered=True
        )

        df_edges["SOURCE_SUBREDDIT"] = df_edges["SOURCE_SUBREDDIT"].astype(
            self.subreddit_cat_type
        )
        df_edges["TARGET_SUBREDDIT"] = df_edges["TARGET_SUBREDDIT"].astype(
            self.subreddit_cat_type
        )

        # Extract node feature
        # Using mean value as the missing embedding (from ROLAND)
        node_features = torch.ones((len(self.subreddit_cat_type.categories), 300))
        node_features *= np.mean(df_nodes.to_numpy())
        for i, subreddit in enumerate(self.subreddit_cat_type.categories):
            if subreddit in df_nodes.index:
                node_features[i, :] = torch.Tensor(df_nodes.loc[subreddit].to_numpy())

        # Extract edge feature
        property_strs = df_edges["PROPERTIES"].to_numpy()
        edge_features = torch.Tensor(
            np.array([x.split(",") for x in property_strs]).astype("float64")
        )
        edge_label = torch.Tensor(df_edges["LINK_SENTIMENT"].to_numpy())

        # Create snapshot mask (1 week interval)
        time_start = df_edges["TIMESTAMP"].min()

        df_edges["TIMESTEP"] = (df_edges["TIMESTAMP"] - time_start) // pd.Timedelta(
            weeks=1
        )
        self._snapshot_masks = torch.Tensor(
            np.array([
                (df_edges["TIMESTEP"] == step).to_numpy()
                for step in range(
                    df_edges["TIMESTEP"].min(),  # type:ignore
                    df_edges["TIMESTEP"].max() + 1,
                )
            ])
        ).to(torch.bool)

        src = df_edges["SOURCE_SUBREDDIT"].cat.codes.to_numpy(copy=True)
        dst = df_edges["TARGET_SUBREDDIT"].cat.codes.to_numpy(copy=True)

        # Create graph
        self._graph = dgl.graph((src, dst))
        self._graph.ndata["feature"] = node_features
        self._graph.edata["feature"] = edge_features
        self._graph.edata["label"] = edge_label

    def __getitem__(self, idx):
        return dgl.edge_subgraph(
            self.graph, self.snapshot_masks[idx], relabel_nodes=False
        )

    def download(self):
        download_url(self._edge_url, self.raw_edge_path)
        download_url(self._node_url, self.raw_node_path)

    def load(self):
        self._snapshot_masks = torch.load(self.snapshot_masks_path, weights_only=True)
        glist, _ = load_graphs(str(self.save_path), [0])
        self._graph = glist[0]

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
    dataset = RedditBodyDataset()
    for snapshot in dataset:
        edge_record = set()
        for src, dst in zip(*snapshot.edges()):
            edge = (src.item(), dst.item())
            if edge in edge_record:
                breakpoint()
            else:
                edge_record.add(edge)

    breakpoint()
