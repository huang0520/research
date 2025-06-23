from dataclasses import dataclass, field
from typing import override

import torch as th
from torch import Tensor
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.typing import EdgeType, NodeOrEdgeType, NodeType
from torch_geometric.utils.mask import mask_to_index

from research.base import SnapshotDiffInfo, SnapshotMetadata


@dataclass(slots=True, frozen=True)
class ComputeInfo:
    compute_leids: Tensor = field(default=th.empty(0))
    keep_curr_lrid: Tensor = field(default=th.empty(0))
    keep_prev_lrid: Tensor = field(default=th.empty(0))
    use_cache: bool = False

    def to(self, device: str):
        return ComputeInfo(
            compute_leids=self.compute_leids.to(device, non_blocking=True),
            keep_curr_lrid=self.keep_curr_lrid.to(device, non_blocking=True),
            keep_prev_lrid=self.keep_prev_lrid.to(device, non_blocking=True),
            use_cache=self.use_cache,
        )


@dataclass(slots=True, frozen=True)
class TransferPackage:
    curr_id: int
    new_gids: dict[NodeOrEdgeType, Tensor]
    add_xs: dict[NodeType, Tensor]
    add_edge_indexes: dict[EdgeType, Tensor]
    add_edge_attrs: dict[EdgeType, Tensor]
    keep_prev_lids: dict[NodeOrEdgeType, Tensor]
    compute_info: ComputeInfo
    is_first: bool = False
    only_edge: bool = False

    def to(self, device: str):
        return TransferPackage(
            curr_id=self.curr_id,
            new_gids={
                k: v.to(device, non_blocking=True) for k, v in self.new_gids.items()
            },
            add_xs={k: v.to(device, non_blocking=True) for k, v in self.add_xs.items()},
            add_edge_indexes={
                k: v.to(device, non_blocking=True)
                for k, v in self.add_edge_indexes.items()
            },
            add_edge_attrs={
                k: v.to(device, non_blocking=True)
                for k, v in self.add_edge_attrs.items()
            },
            keep_prev_lids={
                k: v.to(device, non_blocking=True)
                for k, v in self.keep_prev_lids.items()
            },
            compute_info=self.compute_info.to(device),
            is_first=self.is_first,
            only_edge=self.only_edge,
        )


class BaseDataset(InMemoryDataset):
    _data: HeteroData
    _metadata: dict[int, SnapshotMetadata]
    _diffs: dict[int, SnapshotDiffInfo]
    _packages: dict[int, TransferPackage]
    _reordered_gids: dict[int, dict[NodeOrEdgeType, Tensor]]
    snapshot_ids: list[int]

    @override
    def __init__(
        self,
        incremental: bool,
        incremental_threshold: float = 0.2,
        only_edge: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._incremental = incremental
        self._incremental_threshold = incremental_threshold
        self._only_edge = only_edge

    def _reset_cache(self):
        assert hasattr(self, "_data")

        emasks: Tensor = self._data.edge_stores[0]["mask"]
        self.snapshot_ids = list(range(emasks.size(0)))
        self._metadata = {}
        self._diffs = {}
        self._reordered_gids = {}
        self._packages = {}
        for id in self.snapshot_ids:
            ntypes = self._data.node_types
            etypes = self._data.edge_types

            types = (*ntypes, *etypes)
            masks = {t: self._data[t]["mask"][id] for t in types}
            gids = {t: mask_to_index(masks[t]) for t in types}
            self._metadata[id] = SnapshotMetadata(id, gids, masks, ntypes, etypes)
            self._diffs[id] = self._compute_diff(id)

            if id == 0:
                self._reordered_gids[id] = {
                    t: self._metadata[id].gids[t]
                    for t in (*self._data.node_types, *self._data.edge_types)
                }
            else:
                add_gids = {t: self._diffs[id].add_gids[t] for t in types}
                keep_prev_lids = {
                    t: mask_to_index(~mask[self._reordered_gids[id - 1][t]])
                    for t, mask in self._diffs[id].rm_masks.items()
                }
                self._reordered_gids[id] = {
                    t: th.cat((
                        self._reordered_gids[id - 1][t][keep_prev_lids[t]],
                        add_gids[t],
                    )).contiguous()
                    for t in types
                }

    def __len__(self):
        assert hasattr(self, "snapshot_ids"), "Add `_reset_cache() to __init__()`"
        return len(self.snapshot_ids)

    def __getitem__(self, idx: int):
        if idx not in self._packages:
            self._create_package(idx)
        return self._packages[idx]

    def _create_package(self, idx: int):
        if not self._incremental:
            self._create_normal_package(idx)
            return self._packages[idx]

        if idx == 0:
            self._create_normal_package(idx)
        else:
            self._create_incremental_package(idx)

        return self._packages[idx]

    def _create_normal_package(self, idx: int):
        meta = self._metadata[idx]
        self._packages[idx] = TransferPackage(
            curr_id=0,
            new_gids=self._reordered_gids[idx],
            add_xs={t: self._data[t]["x"][meta.gids[t]] for t in self._data.node_types},
            add_edge_indexes={
                t: self._data[t]["edge_index"][:, meta.gids[t]]
                for t in self._data.edge_types
            },
            add_edge_attrs={
                t: self._data[t]["edge_attr"][meta.gids[t]]
                for t in self._data.edge_types
            }
            if "edge_attr" in self._data
            else {},
            keep_prev_lids={},
            compute_info=ComputeInfo(),
            is_first=True,
            only_edge=self._only_edge,
        )

    def _create_incremental_package(self, idx: int):
        diff = self._diffs[idx]

        # Keep part
        keep_prev_lids = {
            t: mask_to_index(~mask[self._reordered_gids[idx - 1][t]])
            for t, mask in diff.rm_masks.items()
        }

        # Add part
        add_edge_indexes = {
            t: self._data[t]["edge_index"][:, diff.add_gids[t]]
            for t in self._data.edge_types
        }
        if not self._only_edge:
            add_xs = {
                t: self._data[t]["x"][diff.add_gids[t]] for t in self._data.node_types
            }
            if "edge_attr" in self._data:
                add_edge_attrs = {
                    t: self._data[t]["edge_attr"][diff.add_gids[t]]
                    for t in self._data.edge_types
                }
            else:
                add_edge_attrs = {}
        else:
            add_xs = {
                t: self._data[t]["x"][self._reordered_gids[idx][t]]
                for t in self._data.node_types
            }
            if "edge_attr" in self._data:
                add_edge_attrs = {
                    t: self._data[t]["edge_attr"][self._reordered_gids[idx][t]]
                    for t in self._data.edge_types
                }
            else:
                add_edge_attrs = {}

        # Compute Info
        compute_leids, keep_curr_lrid, keep_prev_lrid = self._create_compute_info(idx)
        if (
            len(compute_leids)
            / len(self._reordered_gids[idx][self._data.edge_types[0]])
            >= self._incremental_threshold
        ):
            compute_info = ComputeInfo()
        else:
            compute_info = ComputeInfo(
                compute_leids=compute_leids,
                keep_curr_lrid=keep_curr_lrid,
                keep_prev_lrid=keep_prev_lrid,
                use_cache=True,
            )

        self._packages[idx] = TransferPackage(
            curr_id=idx,
            new_gids=self._reordered_gids[idx],
            add_xs=add_xs,
            add_edge_indexes=add_edge_indexes,
            add_edge_attrs=add_edge_attrs,
            keep_prev_lids=keep_prev_lids,
            compute_info=compute_info,
            only_edge=self._only_edge,
        )

    def _create_compute_info(self, idx: int):
        assert len(self._data.edge_types) == 1
        t = self._data.edge_types[0]

        diff = self._diffs[idx]
        change_emask = diff.add_masks[t] | diff.rm_masks[t]

        target_gnid: Tensor = self._data[t]["edge_index"][1, change_emask]

        curr_geid = self._reordered_gids[idx][t]
        curr_edge_index: Tensor = self._data[t]["edge_index"][:, curr_geid]

        # Find in edge of target dst
        lookup: Tensor = th.zeros(self._data.num_nodes, dtype=th.bool)  # type:ignore
        lookup[target_gnid] = True

        compute_leid = mask_to_index(lookup[curr_edge_index[1]])

        # Find non-computed lid of result
        curr_dst_gnid = self._reordered_gids[idx][t[2]]
        prev_dst_gnid = self._reordered_gids[idx - 1][t[2]]

        lookup = diff.keep_masks[t[2]].clone()
        lookup[curr_edge_index[1, compute_leid]] = False

        unaffected_curr_lrid = mask_to_index(lookup[curr_dst_gnid])
        unaffected_prev_lrid = mask_to_index(lookup[prev_dst_gnid])

        assert unaffected_curr_lrid.shape == unaffected_prev_lrid.shape
        return compute_leid, unaffected_curr_lrid, unaffected_prev_lrid

    def _compute_diff(self, idx: int) -> SnapshotDiffInfo:
        if idx == 0:
            return SnapshotDiffInfo(0, 0, {}, {}, {}, {}, {}, {})

        prev_meta = self._metadata[idx - 1]
        curr_meta = self._metadata[idx]

        types = (*self._data.node_types, *self._data.edge_types)
        add_masks = {t: ~prev_meta.masks[t] & curr_meta.masks[t] for t in types}
        rm_masks = {t: prev_meta.masks[t] & ~curr_meta.masks[t] for t in types}
        keep_masks = {t: prev_meta.masks[t] & curr_meta.masks[t] for t in types}

        add_gids = {t: m.nonzero().view(-1) for t, m in add_masks.items()}
        rm_gids = {t: m.nonzero().view(-1) for t, m in rm_masks.items()}
        keep_gids = {t: m.nonzero().view(-1) for t, m in keep_masks.items()}

        return SnapshotDiffInfo(
            idx - 1,
            idx,
            add_masks,
            rm_masks,
            keep_masks,
            add_gids,
            rm_gids,
            keep_gids,
        )
