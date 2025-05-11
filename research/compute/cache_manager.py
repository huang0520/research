import copy
from abc import ABC, abstractmethod
from typing import NewType, TypeGuard

import torch as th
from torch import Tensor, nn
from torch_geometric.typing import Adj
from torch_geometric.utils import index_to_mask, mask_to_index
from torch_geometric.utils.map import map_index

from research.base import SnapshotContext
from research.loader import SnapshotManager


def is_cacheable_module(obj: object) -> TypeGuard["CacheableModule"]:
    return isinstance(obj, (nn.Module, CacheableMixin))


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


class AggCacheManager:
    def __init__(self, context: SnapshotContext):
        self.context = context
        self.registered_layer = {}

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

    def register_model(self, model: nn.Module) -> "nn.Module | CachedModule":
        cached_model = copy.deepcopy(model)
        if isinstance(cached_model, CacheableModule):
            name = cached_model._get_name()
            cached_layer = CachedModule(cached_model, name, self)
            self.registered_layer[name] = cached_layer
            return cached_layer

        self._replace_cacheable_module(cached_model)
        return cached_model

    def _replace_cacheable_module(self, module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            path = f"{prefix}.{name}" if prefix else name

            if isinstance(child, CacheableModule):
                cached_layer = CachedModule(child, name, self)
                self.registered_layer[path] = cached_layer
                setattr(module, name, cached_layer)  # type:ignore
            else:
                self._replace_cacheable_module(child, path)


def create_cached_model(model: nn.Module, context: SnapshotContext) -> nn.Module:
    cached_model = copy.deepcopy(model)
    if is_cacheable_module(cached_model):
        name = cached_model._get_name()
        cached_layer = CachedModule(cached_model, name, context)
        return cached_layer

    _replace_cacheable_module(cached_model, context)
    return cached_model


def _replace_cacheable_module(
    module: nn.Module, context: SnapshotContext, prefix: str = ""
):
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name

        if is_cacheable_module(child):
            cached_layer = CachedModule(child, name, context)
            setattr(module, name, cached_layer)
        else:
            _replace_cacheable_module(child, context, path)


class CachedModule(nn.Module):
    """
    Wrapper to cache aggregation result of layer.
    Layer should inherient from CahceMixin.
    """

    def __init__(self, layer: CacheableModule, layer_id: str, context: SnapshotContext):
        super().__init__()

        self.layer = layer
        self.layer_id = layer_id
        self.context = context

        if not isinstance(layer, CacheableMixin):
            raise TypeError(f"Layer {layer_id} does not support caching!!")

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        if self.context.prev_agg.get(self.layer_id) is None:
            curr_agg = self.layer.compute_aggregate(x, edge_index)
        else:
            assert (
                self.context.curr_data is not None
                and self.context.prev_data is not None
            )

            affected_curr_leid, unaffected_gnid = self._get_compute_info()
            prev_agg = self.context.prev_agg[self.layer_id]
            curr_agg = self.layer.compute_aggregate(
                x, edge_index, compute_eid=affected_curr_leid
            )

            # Combine new aggregation and cache
            unaffected_prev_lnid = self.context.prev_data.gid_to_lid(
                unaffected_gnid, inclusive=True
            )
            unaffected_curr_lnid = self.context.curr_data.gid_to_lid(
                unaffected_gnid, inclusive=True
            )
            curr_agg[unaffected_curr_lnid] = prev_agg[unaffected_prev_lnid]

        self.context.curr_agg[self.layer_id] = curr_agg
        return self.layer.compute_update(curr_agg)

    def _get_compute_info(self):
        assert self.context.curr_data is not None and self.context.prev_data is not None
        diff = self.context.diff

        # Get edges need to compute
        compute_emask = diff.add_emask | diff.rm_emask

        # Get dst of compute edges
        affected_gnid = (
            self.context.main_data.edge_index[1, compute_emask]
            .unique(sorted=False)
            .to(self.context.device)
        )
        affected_curr_lnid = self.context.curr_data.gid_to_lid(affected_gnid)

        # Find in edges of compute dst
        lookup: Tensor = th.zeros(
            (self.context.curr_data.gnid.max() + 1),  # type: ignore
            dtype=th.bool,
            device=self.context.device,
        )
        lookup[affected_curr_lnid] = True

        mask = lookup[self.context.curr_data.edge_index[1]]
        affected_curr_leid = mask.nonzero().view(-1)

        # Update diff
        affected_nmask = index_to_mask(
            affected_gnid, size=self.context.main_data.num_nodes
        ).cpu()

        # Find global nid of keep aggregation result
        unaffected_nmask = diff.keep_nmask & ~affected_nmask
        unaffected_gnid = mask_to_index(unaffected_nmask).to(self.context.device)

        return affected_curr_leid, unaffected_gnid
