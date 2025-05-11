import copy
from abc import ABC, abstractmethod

import torch as th
from torch import Tensor, nn
from torch_geometric.typing import Adj
from torch_geometric.utils.map import map_index

from research.data.loader import SnapshotManager


class CacheableMixin(ABC):
    @abstractmethod
    def compute_aggregate(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight=None,
        compute_eid: Tensor | None = None,
    ) -> Tensor:
        pass

    @abstractmethod
    def compute_update(self, x: Tensor) -> Tensor:
        pass


class CacheableModule(nn.Module, CacheableMixin):
    pass


class AggregationCache:
    def __init__(
        self, snapshot_manager: SnapshotManager, cacheable_type=(CacheableMixin,)
    ):
        self.snapshot_manager = snapshot_manager
        self.cacheable_type = cacheable_type
        self.registered_layer = {}
        self.cache = {}  # {snapshot_id: {layer_id: aggregation}}

    def get(self, layer_id: str, snapshot_id: int) -> Tensor | None:
        """Get the aggregation result of snapshot."""
        if snapshot_id not in self.cache or layer_id not in self.cache[snapshot_id]:
            return None
        return self.cache[snapshot_id][layer_id]

    def update(self, layer_id, snapshot_id: int, aggregation: Tensor):
        """Update cache."""
        if snapshot_id not in self.cache:
            self.cache[snapshot_id] = {}
        self.cache[snapshot_id][layer_id] = aggregation

    def register_model(self, model: nn.Module) -> "nn.Module | CachedLayer":
        cached_model = copy.deepcopy(model)
        if isinstance(model, self.cacheable_type):
            name = cached_model._get_name()
            cached_layer = CachedLayer(cached_model, name, self)  # type:ignore
            self.registered_layer[name] = cached_layer
            return cached_layer

        self._replace_cacheable_module(cached_model)
        return cached_model

    def _replace_cacheable_module(self, module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            path = f"{prefix}.{name}" if prefix else name

            if isinstance(child, self.cacheable_type):
                cached_layer = CachedLayer(child, name, self)
                self.registered_layer[path] = cached_layer
                setattr(module, name, cached_layer)  # type:ignore
            else:
                self._replace_cacheable_module(child, path)


class CachedLayer:
    """
    Wrapper to cache aggregation result of layer.
    Layer should inherient from CahceMixin.
    """

    def __init__(self, layer: CacheableModule, layer_id: str, cache: AggregationCache):
        self.layer = layer
        self.layer_id = layer_id
        self.cache = cache
        self.prev_snapshot_id: int | None = None

        if not isinstance(layer, CacheableMixin):
            raise TypeError(f"Layer {layer_id} does not support caching!!")

    def __call__(self, x: Tensor, edge_index, snapshot_id: int | None = None) -> Tensor:
        if snapshot_id is None:
            return self.layer(x, edge_index)

        if self.prev_snapshot_id is None:
            agg = self.layer.compute_aggregate(x, edge_index)
            self.cache.update(self.layer_id, snapshot_id, agg)
            self.prev_snapshot_id = snapshot_id
        else:
            # TODO: Update the aggregation cache
            prev_keep_gnids, prev_keep_lnids, compute_ldst, compute_eid = (
                self._get_compute_info(snapshot_id)
            )
            prev_agg = self.cache.get(self.layer_id, self.prev_snapshot_id)
            curr_agg = self.layer.compute_aggregate(
                x, edge_index, compute_eid=compute_eid
            )

            keep_agg = prev_agg[prev_keep_lnids]
            cover_idx, _ = map_index(
                prev_keep_gnids,
                self.cache.snapshot_manager.curr_cuda_data.gnid,
                inclusive=True,
            )

            curr_agg[cover_idx] = keep_agg
            agg = curr_agg

            self.cache.update(self.layer_id, snapshot_id, agg)
            self.prev_snapshot_id = snapshot_id

        return self.layer.compute_update(agg)

    def _get_compute_info(self, snapshot_id: int):
        assert self.prev_snapshot_id is not None

        # Get edges need to compute
        prev_emask = self.cache.snapshot_manager.snapshots[self.prev_snapshot_id].emask
        curr_emask = self.cache.snapshot_manager.snapshots[snapshot_id].emask
        compute_emask = prev_emask ^ curr_emask
        non_compute_emask = curr_emask & ~compute_emask

        # Get dst of compute edges
        compute_gdst = self.cache.snapshot_manager.main_data.edge_index[
            1, compute_emask
        ]
        non_compute_gdst = self.cache.snapshot_manager.main_data.edge_index[
            1, non_compute_emask
        ]
        compute_ldst, _ = map_index(
            compute_gdst.cuda(non_blocking=True),
            self.cache.snapshot_manager.curr_cuda_data.gnid,
        )

        # Find in edges of compute dst
        lookup: Tensor = compute_ldst.new_empty(
            (self.cache.snapshot_manager.curr_cuda_data.edge_index[1].max() + 1,),  # type:ignore
            dtype=th.bool,
        )
        lookup[compute_ldst] = True

        mask = lookup[self.cache.snapshot_manager.curr_cuda_data.edge_index[1]]
        compute_leids = mask.nonzero().view(-1)

        # Find global nid of keep aggregation result
        mask = th.isin(
            self.cache.snapshot_manager.snapshots[self.prev_snapshot_id].nid,
            non_compute_gdst,
        )
        prev_keep_gnids = (
            self.cache.snapshot_manager.snapshots[self.prev_snapshot_id]
            .nid[mask]
            .cuda(non_blocking=True)
        )
        prev_keep_lnids = mask.nonzero().view(-1).cuda(non_blocking=True)

        return prev_keep_gnids, prev_keep_lnids, compute_ldst, compute_leids
