from dataclasses import dataclass, field
from typing import Literal

import torch as th
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils.map import map_index


class SubData(Data):
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


class MainData(Data):
    x: Tensor
    edge_index: Tensor
    nmasks: Tensor
    emasks: Tensor


@dataclass
class SnapshotMetadata:
    id: int
    gnid: Tensor
    geid: Tensor
    nmask: Tensor
    emask: Tensor


@dataclass
class SnapshotDiffInfo:
    src_id: int
    dst_id: int

    add_nmask: Tensor
    rm_nmask: Tensor
    keep_nmask: Tensor

    add_emask: Tensor
    rm_emask: Tensor
    keep_emask: Tensor

    _add_gnid: Tensor | None = None
    _rm_gnid: Tensor | None = None
    _keep_gnid: Tensor | None = None

    _add_geid: Tensor | None = None
    _rm_geid: Tensor | None = None
    _keep_geid: Tensor | None = None

    @property
    def add_gnid(self):
        if self._add_gnid is None:
            self._add_gnid = self.add_nmask.nonzero().view(-1)
        return self._add_gnid

    @property
    def rm_gnid(self):
        if self._rm_gnid is None:
            self._rm_gnid = self.rm_nmask.nonzero().view(-1)
        return self._rm_gnid

    @property
    def keep_gnid(self):
        if self._keep_gnid is None:
            self._keep_gnid = self.keep_nmask.nonzero().view(-1)
        return self._keep_gnid

    @property
    def add_geid(self):
        if self._add_geid is None:
            self._add_geid = self.add_emask.nonzero().view(-1)
        return self._add_geid

    @property
    def rm_geid(self):
        if self._rm_geid is None:
            self._rm_geid = self.rm_emask.nonzero().view(-1)
        return self._rm_geid

    @property
    def keep_geid(self):
        if self._keep_geid is None:
            self._keep_geid = self.keep_emask.nonzero().view(-1)
        return self._keep_geid


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
