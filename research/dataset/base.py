from dataclasses import dataclass

import torch as th
from torch import Tensor
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.typing import EdgeType, NodeOrEdgeType, NodeType
from torch_geometric.utils.mask import mask_to_index

from research.base import SnapshotDiffInfo, SnapshotMetadata


@dataclass(slots=True, frozen=True)
class TransferPackage:
    curr_id: int
    new_gids: dict[NodeOrEdgeType, Tensor]
    add_xs: dict[NodeType, Tensor]
    add_edge_indexes: dict[EdgeType, Tensor]
    keep_prev_lids: dict[NodeOrEdgeType, Tensor]
    add_edge_attrs: dict[EdgeType, Tensor]
    is_first: bool = False

    def to(self, device: "str"):
        return TransferPackage(
            self.curr_id,
            {k: v.to(device, non_blocking=True) for k, v in self.new_gids.items()},
            {k: v.to(device, non_blocking=True) for k, v in self.add_xs.items()},
            {
                k: v.to(device, non_blocking=True)
                for k, v in self.add_edge_indexes.items()
            },
            {
                k: v.to(device, non_blocking=True)
                for k, v in self.keep_prev_lids.items()
            },
            {
                k: v.to(device, non_blocking=True)
                for k, v in self.add_edge_attrs.items()
            },
            self.is_first,
        )


class BaseDataset(InMemoryDataset):
    _data: HeteroData
    _metadata: dict[int, SnapshotMetadata]
    _diffs: dict[int, SnapshotDiffInfo]
    _packages: dict[int, TransferPackage]
    _reordered_gids: dict[int, dict[NodeOrEdgeType, Tensor]]
    snapshot_ids: list[int]

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

    def __len__(self):
        assert hasattr(self, "snapshot_ids"), "Add `_reset_cache() to __init__()`"
        return len(self.snapshot_ids)

    def __getitem__(self, idx: int):
        if idx not in self._packages:
            self._create_package(idx)
        return self._packages[idx]

    def _create_package(self, idx: int):
        meta = self._metadata[idx]
        if idx == 0:
            self._reordered_gids[idx] = {
                t: meta.gids[t]
                for t in (*self._data.node_types, *self._data.edge_types)
            }
            self._packages[idx] = TransferPackage(
                0,
                self._reordered_gids[idx],
                {t: self._data[t]["x"][meta.gids[t]] for t in self._data.node_types},
                {
                    t: self._data[t]["edge_index"][:, meta.gids[t]]
                    for t in self._data.edge_types
                },
                {},
                {
                    t: self._data[t]["edge_attr"][meta.gids[t]]
                    for t in self._data.edge_types
                }
                if "edge_attr" in self._data
                else {},
                True,
            )
            return

        diff = self._diffs[idx]
        types = (*self._data.node_types, *self._data.edge_types)

        # Add part
        add_gids = {t: diff.add_gids[t] for t in types}
        add_xs = {
            t: self._data[t]["x"][diff.add_gids[t]] for t in self._data.node_types
        }
        add_edge_indexes = {
            t: self._data[t]["edge_index"][:, diff.add_gids[t]]
            for t in self._data.edge_types
        }
        if "edge_attr" in self._data:
            add_edge_attrs = {
                t: self._data[t]["edge_attr"][diff.add_gids[t]]
                for t in self._data.edge_types
            }
        else:
            add_edge_attrs = {}

        # Keep part
        if idx - 1 not in self._reordered_gids:
            self._create_package(idx - 1)

        keep_prev_lids = {
            t: mask_to_index(~mask[self._reordered_gids[idx - 1][t]])
            for t, mask in diff.rm_masks.items()
        }

        self._reordered_gids[idx] = {
            t: th.cat((
                self._reordered_gids[idx - 1][t][keep_prev_lids[t]],
                add_gids[t],
            )).contiguous()
            for t in types
        }
        self._packages[idx] = TransferPackage(
            idx,
            self._reordered_gids[idx],
            add_xs,
            add_edge_indexes,
            keep_prev_lids,
            add_edge_attrs,
        )

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
