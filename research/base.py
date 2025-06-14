from dataclasses import dataclass, field
from typing import Literal

import torch as th
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils.map import map_index

MainData = HeteroData


class SubData:
    def __init__(self, data: HeteroData):
        self._data = data


class SubData_(Data):
    x: Tensor
    edge_index: Tensor
    gnid: Tensor
    geid: Tensor

    def gid_to_lid(
        self,
        gid: Tensor,
        type: Literal["node", "edge"] = "node",
        inclusive: bool = False,
    ):
        index = self.gnid if type == "node" else self.geid
        lid, _ = map_index(gid, index, inclusive=inclusive)
        return lid

    def lid_to_gid(self, lid: Tensor, type: Literal["node", "edge"] = "node"):
        id = self.gnid if type == "node" else self.geid
        return id[lid]


@dataclass
class SnapshotMetadata:
    id: int
    gids: dict[NodeType | EdgeType, Tensor]
    masks: dict[NodeType | EdgeType, Tensor]
    ntype: list[NodeType]
    etype: list[EdgeType]


@dataclass(frozen=True)
class SnapshotDiffInfo:
    prev_id: int
    curr_id: int
    add_masks: dict[NodeType | EdgeType, Tensor]
    rm_masks: dict[NodeType | EdgeType, Tensor]
    keep_masks: dict[NodeType | EdgeType, Tensor]
    add_gids: dict[NodeType | EdgeType, Tensor]
    rm_gids: dict[NodeType | EdgeType, Tensor]
    keep_gids: dict[NodeType | EdgeType, Tensor]


@dataclass
class SnapshotContext:
    main_data: "MainData"
    metadata: dict[int, SnapshotMetadata] = field(default_factory=dict)
    device: str = "cuda"

    prev_id: int | None = None
    curr_id: int | None = None
    prev_data: SubData | None = None
    curr_data: SubData | None = None

    # Aggregation Cache
    # {layer_id: aggregation}
    prev_agg: dict[str, Tensor] = field(default_factory=dict)
    curr_agg: dict[str, Tensor] = field(default_factory=dict)

    _diff: SnapshotDiffInfo | None = None

    @property
    def diff(self) -> SnapshotDiffInfo:
        if self._diff is None:
            self._diff = self._compute_diff()

        return self._diff

    def _compute_diff(self) -> SnapshotDiffInfo:
        assert self.prev_id is not None and self.curr_id is not None
        # If no diff exist, compute the diff info
        src_meta = self.metadata[self.prev_id]
        dst_meta = self.metadata[self.curr_id]

        add_nmask = ~src_meta.nmask & dst_meta.nmask
        rm_nmask = src_meta.nmask & ~dst_meta.nmask
        keep_nmask = src_meta.nmask & dst_meta.nmask

        add_emask = ~src_meta.emask & dst_meta.emask
        rm_emask = src_meta.emask & ~dst_meta.emask
        keep_emask = src_meta.emask & dst_meta.emask

        return SnapshotDiffInfo(
            src_id=self.prev_id,
            dst_id=self.curr_id,
            add_nmask=add_nmask,
            rm_nmask=rm_nmask,
            keep_nmask=keep_nmask,
            add_emask=add_emask,
            rm_emask=rm_emask,
            keep_emask=keep_emask,
        )

    def step(self):
        assert self.curr_id is not None
        assert self.curr_data is not None
        self.prev_id = self.curr_id if self.curr_id is not None else None
        self.prev_data = self.curr_data.clone() if self.curr_data is not None else None
        self.prev_agg = self.curr_agg
        self.curr_agg = {}

        self._diff = None

    def reset_state(self):
        self.prev_id = None
        self.prev_data = None
        self.curr_id = None
        self.curr_data = None
        self.curr_agg = {}
        self.prev_agg = {}
        self._diff = None

    def __len__(self):
        return len(self.metadata)
