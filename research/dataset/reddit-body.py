from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs, save_graphs

from research.utils.download import download_url


class RedditBodyDataset(DGLDataset):
    def __init__(self, save_dir="./data", force_reload=False):
        self._edge_url = "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"
        self._node_url = (
            "https://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"
        )
        super().__init__(
            name="reddit-body", raw_dir=save_dir, force_reload=force_reload
        )

    def process(self):
        rng = np.random.default_rng()

        df_edges = pd.read_csv(
            self.raw_edge_path,
            sep="\t",
            parse_dates=[3],  # Columns index of TIMESTAMP
            date_format="%Y-%m-%d %H:%M:%S",
        )
        df_nodes = pd.read_csv(self.raw_node_path, header=None)

        # Create subreddit id to node id dict
        subreddit_ids = set(df_edges["SOURCE_SUBREDDIT"].unique()) | set(
            df_edges["TARGET_SUBREDDIT"].unique()
        )
        subreddit_id2node_id: dict[str, int] = {
            subreddit_id: node_id for node_id, subreddit_id in enumerate(subreddit_ids)
        }
        self.node_id2subreddit_id = {
            node_id: subreddit_id
            for subreddit_id, node_id in subreddit_id2node_id.items()
        }

        # Change subreddit id to node id
        df_edges.loc[:, "SOURCE_SUBREDDIT"] = df_edges.loc[:, "SOURCE_SUBREDDIT"].map(
            subreddit_id2node_id
        )
        df_edges.loc[:, "TARGET_SUBREDDIT"] = df_edges.loc[:, "TARGET_SUBREDDIT"].map(
            subreddit_id2node_id
        )

        # Extract node feature
        df_nodes = df_nodes[df_nodes.iloc[:, 0].apply(lambda x: x in subreddit_ids)]
        df_nodes.iloc[:, 0] = df_nodes.iloc[:, 0].map(subreddit_id2node_id)
        df_nodes = df_nodes.set_index(0)

        embeddings_ = {
            idx: embedding.to_numpy() for idx, embedding in df_nodes.iterrows()
        }
        node_features_ = np.array([
            embeddings_.get(node_id, rng.random(300))
            for node_id in range(len(subreddit_id2node_id))
        ])
        node_features = torch.tensor(node_features_)

        # Extract edge feature
        edge_features_ = np.array(
            df_edges["PROPERTIES"].apply(lambda x: np.fromstring(x, sep=",")).to_list()
        )
        edge_features = torch.tensor(edge_features_)

        edge_label = torch.tensor(df_edges["LINK_SENTIMENT"].to_numpy())

        # Create snapshot index (1 week interval)
        time_start = df_edges["TIMESTAMP"].min()
        time_end = df_edges["TIMESTAMP"].max()

        time_indices = []
        time_curr = time_start
        while time_curr < time_end:
            time_indices.append(time_curr)
            time_curr += pd.Timedelta(weeks=1)
        time_indices.append(time_end + pd.Timedelta(days=1))

        snapshot_masks_ = []
        for i, time_indice in enumerate(time_indices[:-1]):
            snapshot_edge_indices = df_edges[
                (df_edges["TIMESTAMP"] >= time_indice)
                & (df_edges["TIMESTAMP"] < time_indices[i + 1])
            ].index
            snapshot_masks_.append(snapshot_edge_indices.to_numpy())
        snapshot_masks = torch.tensor(np.array(snapshot_masks_))

        # Create graph
        src, dst = (
            df_edges.loc[:, ("SOURCE_SUBREDDIT", "TARGET_SUBREDDIT")]
            .to_numpy(dtype="int64")
            .transpose()
        )
        self.graph = dgl.graph((src, dst))
        self.graph.ndata["feature"] = node_features
        self.graph.edata["weight"] = edge_features
        self.graph.edata["label"] = edge_label
        self.graph.edata["snapshot_mask"] = snapshot_masks

    def __getitem__(self, idx):
        return dgl.edge_subgraph(self.graph, self.graph.edata["snapshot_mask"][idx])

    def __len__(self):
        return super().__len__()

    def download(self):
        download_url(self._edge_url, self.raw_edge_path)
        download_url(self._node_url, self.raw_node_path)

    def save(self):
        save_graphs(str(self.save_path), self.graph)

    def load(self):
        glist, _ = load_graphs(str(self.save_path), [0])
        self.graph = glist[0]

    def has_cache(self):
        return self.save_path.exists()

    def _download(self):
        if self.raw_edge_path.exists() and self.raw_node_path.exists():
            return

        self.raw_dir.mkdir(exist_ok=True)
        self.download()

    @property
    def raw_dir(self):
        return Path(self._raw_dir).absolute() / self.name

    @property
    def raw_edge_path(self):
        return self.raw_dir / "soc-redditHyperlinks-body.tsv"

    @property
    def raw_node_path(self):
        return self.raw_dir / "web-redditEmbeddings-subreddits.csv"

    @property
    def save_path(self):
        return self.raw_dir / "reddit-body.bin"


if __name__ == "__main__":
    dataset = RedditBodyDataset(force_reload=True)
