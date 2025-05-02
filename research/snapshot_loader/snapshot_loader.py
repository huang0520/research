import gc
from dataclasses import dataclass
from typing import Literal

import dgl
import torch as th
from dgl.heterograph import DGLGraph
from polars import selectors as cs

from research.dataset import BaseDataset


# @th.jit.script
class IDMapper:
    global_id_range: th.Tensor
    sort_id_range: th.Tensor
    sort_indices: th.Tensor
    device: th.device
    id_range_len: int

    def __init__(self, global_id_range: th.Tensor) -> None:
        # Local id range just from 0 ~ len(global_id_range) - 1, like following
        # self.local_id_range = range(global_id_range)
        self.global_id_range = global_id_range
        self.sort_id_range, self.sort_indices = th.sort(global_id_range)
        self.device = global_id_range.device
        self.id_range_len = self.sort_id_range.shape[0]

    def global_to_local(self, global_ids: th.Tensor, strict: bool = True) -> th.Tensor:
        indices = th.searchsorted(self.sort_id_range, global_ids)

        # strict == true: all input id should show in the id range
        # strict == false: input id not show in the id range will be ignore
        if strict:
            assert (self.sort_id_range[indices] == global_ids).all(), (
                "Some global IDs are not in the mapping range!!"
            )
            return self.sort_indices[indices]
        else:
            mask = indices < self.id_range_len
            valid_indices = indices[mask].contiguous()
            valid_global_ids = global_ids[mask].contiguous()

            matches = self.sort_id_range[valid_indices] == valid_global_ids
            rst = self.sort_indices[valid_indices[matches]]

        return rst

    def local_to_global(self, local_ids: th.Tensor) -> th.Tensor:
        return self.global_id_range[local_ids]

    def to(self, device: th.device) -> "IDMapper":
        self.global_id_range = self.global_id_range.to(device)
        self.sort_id_range = self.sort_id_range.to(device)
        self.sort_indices = self.sort_indices.to(device)
        self.device = device

        return self


class IDGetter:
    # TODO: Instead of just using mask overlap to find id, compare features to get more
    # overlap / differ ids
    def __init__(self, masks: th.Tensor):
        self._masks = masks

    def get_mask(
        self,
        idx1: int,
        idx2: int | None = None,
        type: Literal["self", "overlap", "differ", "1_only"] = "self",
    ) -> th.Tensor:
        match type:
            case "self":
                assert idx2 is None
                return self._masks[idx1]
            case "overlap":
                assert idx2 is not None
                return self._masks[idx1] & self._masks[idx2]
            case "differ":
                assert idx2 is not None
                return self._masks[idx1] ^ self._masks[idx2]
            case "1_only":
                assert idx2 is not None
                return self._masks[idx1] & ~self._masks[idx2]
            case _:
                raise TypeError(f"Not allow mask type: {type}")

    def get_id(
        self,
        idx1: int,
        idx2: int | None = None,
        type: Literal["self", "overlap", "differ", "1_only"] = "self",
    ) -> th.Tensor:
        return self.get_mask(idx1, idx2, type).nonzero(as_tuple=True)[0]


@dataclass(slots=True, frozen=True)
class GraphInfo:
    src_nids: th.Tensor
    dst_nids: th.Tensor
    nfeats: th.Tensor
    efeats: th.Tensor | None = None


@dataclass(slots=True)
class SnapshotIR:
    idx: int
    nids_mapper: IDMapper
    eids_mapper: IDMapper

    # Local nids/eids is just index of ids_global (0 ~ len(ids_global))
    # e.g. Local nids N <-> nids_global[N]
    nids_global: th.Tensor  # GPU, Int
    eids_global: th.Tensor  # GPU, Int

    src_nids_local: th.Tensor  # GPU, Int
    dst_nids_local: th.Tensor  # GPU, Int
    nfeats: th.Tensor  # GPU, Int
    efeats: th.Tensor | None = None  # GPU, Int
    nlabels: th.Tensor | None = None  # GPU, Int
    elabels: th.Tensor | None = None  # GPU, Int


def build_graph(ir: SnapshotIR) -> DGLGraph:
    snapshot = dgl.graph((ir.src_nids_local, ir.dst_nids_local))
    snapshot.ndata["feat"] = ir.nfeats
    snapshot.ndata[dgl.NID] = ir.nids_global
    snapshot.edata[dgl.EID] = ir.eids_global

    if ir.efeats is not None:
        snapshot.edata["feat"] = ir.efeats
    if ir.nlabels is not None:
        snapshot.ndata["label"] = ir.nlabels
    if ir.elabels is not None:
        snapshot.edata["label"] = ir.elabels

    return snapshot


def create_ir(
    idx: int,
    ginfo: GraphInfo,
    nids_getter: IDGetter,
    eids_getter: IDGetter,
) -> SnapshotIR:
    """
    Initialize the first snapshot from scratch.

    Returns:
        The initial DGL graph snapshot
    """
    # Get nodes and edges that exist in the current snapshot
    nids_global = nids_getter.get_id(idx)
    eids_global = eids_getter.get_id(idx)

    nids_mapper = IDMapper(nids_global)
    eids_mapper = IDMapper(eids_global)

    # Get information of edge
    src_nids_local = nids_mapper.global_to_local(ginfo.src_nids[eids_global]).cuda()
    dst_nids_local = nids_mapper.global_to_local(ginfo.dst_nids[eids_global]).cuda()
    nfeats = ginfo.nfeats[nids_global].cuda()
    efeats = ginfo.efeats[eids_global].cuda() if ginfo.efeats is not None else None

    return SnapshotIR(
        idx=idx,
        nids_mapper=nids_mapper.to(th.device("cuda")),
        eids_mapper=eids_mapper.to(th.device("cuda")),
        nids_global=nids_global.cuda(),
        eids_global=eids_global.cuda(),
        src_nids_local=src_nids_local,
        dst_nids_local=dst_nids_local,
        nfeats=nfeats,
        efeats=efeats,
    )


def update_ir(
    idx: int,
    ginfo: GraphInfo,
    nid_getter: IDGetter,
    eid_getter: IDGetter,
    prev_ir: SnapshotIR,
):
    """
    Update the snapshot based on differences from the previous snapshot.
    This is more efficient than transfering each snapshot.

    Returns:
        The updated DGL graph snapshot
    """
    # Identify nodes/edges to keep (present in both previous and current)
    # TODO: Consider the overlap of features
    keep_nid_global = nid_getter.get_id(prev_ir.idx, idx, "overlap")
    keep_eid_global = eid_getter.get_id(prev_ir.idx, idx, "overlap")

    # Identify nodes/edges to add (not in previous but present in current)
    # TODO: Consider the overlap of features
    add_nid_global = nid_getter.get_id(idx, prev_ir.idx, "1_only")
    add_eid_global = eid_getter.get_id(idx, prev_ir.idx, "1_only")

    # # Move necessary informantion to GPU for the operation need to perform on GPU
    keep_nid_global_cuda = keep_nid_global.cuda()
    keep_eid_global_cuda = keep_eid_global.cuda()
    add_nid_global_cuda = add_nid_global.cuda()
    add_eid_global_cuda = add_eid_global.cuda()

    # # Get global IDs and features for nodes/edges to add
    add_src_nids_global_cuda = ginfo.src_nids[add_eid_global].cuda()
    add_dst_nids_global_cuda = ginfo.dst_nids[add_eid_global].cuda()
    add_nfeat_cuda = ginfo.nfeats[add_nid_global].cuda()
    add_efeat_cuda = (
        ginfo.efeats[add_eid_global].cuda() if ginfo.efeats is not None else None
    )

    # Find indices in previous snapshot for nodes/edges to keep
    keep_nid_local_prev_cuda = prev_ir.nids_mapper.global_to_local(keep_nid_global_cuda)
    keep_eid_local_prev_cuda = prev_ir.eids_mapper.global_to_local(keep_eid_global_cuda)

    # Get source/destination and features for edges/nodes to keep
    keep_src_nids_global_cuda = prev_ir.nids_mapper.local_to_global(
        prev_ir.src_nids_local[keep_eid_local_prev_cuda]
    )
    keep_dst_nids_global_cuda = prev_ir.nids_mapper.local_to_global(
        prev_ir.dst_nids_local[keep_eid_local_prev_cuda]
    )
    keep_nfeat_cuda = prev_ir.nfeats[keep_nid_local_prev_cuda]
    keep_efeat_cuda = (
        prev_ir.efeats[keep_eid_local_prev_cuda] if prev_ir.efeats is not None else None
    )

    nids_global_cuda = th.cat((keep_nid_global_cuda, add_nid_global_cuda))
    eids_global_cuda = th.cat((keep_eid_global_cuda, add_eid_global_cuda))
    nids_mapper = IDMapper(nids_global_cuda)
    eids_mapper = IDMapper(eids_global_cuda)

    src_nids_global_cuda = th.hstack((
        keep_src_nids_global_cuda,
        add_src_nids_global_cuda,
    ))
    dst_nids_global_cuda = th.hstack((
        keep_dst_nids_global_cuda,
        add_dst_nids_global_cuda,
    ))

    src_nids_local_cuda = nids_mapper.global_to_local(src_nids_global_cuda)
    dst_nids_local_cuda = nids_mapper.global_to_local(dst_nids_global_cuda)
    nfeats_cuda = th.cat((keep_nfeat_cuda, add_nfeat_cuda))
    efeats_cuda = (
        th.cat((keep_efeat_cuda, add_efeat_cuda))
        if keep_efeat_cuda is not None and add_efeat_cuda is not None
        else None
    )

    return SnapshotIR(
        idx=idx,
        nids_mapper=nids_mapper,
        eids_mapper=eids_mapper,
        nids_global=nids_global_cuda,
        eids_global=eids_global_cuda,
        src_nids_local=src_nids_local_cuda,
        dst_nids_local=dst_nids_local_cuda,
        nfeats=nfeats_cuda,
        efeats=efeats_cuda,
    )


def find_compute_eids(
    prev_idx: int,
    curr_idx: int,
    ginfo: GraphInfo,
    eid_getter: IDGetter,
    curr_ir: SnapshotIR,
):
    # Find the dst node of differ edges. The idx of nodes represent results that need
    # to be compute.
    compute_dst_nids_global = ginfo.dst_nids[
        eid_getter.get_id(prev_idx, curr_idx, "differ")
    ].cuda()

    compute_dst_nids_local = curr_ir.nids_mapper.global_to_local(
        compute_dst_nids_global, strict=False
    ).unique(sorted=False)

    lookup: th.Tensor = th.zeros(
        curr_ir.dst_nids_local.max() + 1,  # type:ignore
        dtype=th.bool,
        device="cuda",
    )
    lookup[compute_dst_nids_local] = True

    mask = lookup[curr_ir.dst_nids_local]
    indices = mask.nonzero(as_tuple=True)[0]

    return indices


class SnapshotLoader:
    def __init__(
        self, dataset: BaseDataset, start: int = 0, end: int | None = None
    ) -> None:
        """
        Initialize the snapshot iterator with dataset and range settings.

        Args:
            dataset: The graph dataset containing temporal information
            start: The starting index of snapshots (default: 0)
            end: The ending index of snapshots (default: None)
        """
        # Extract graph structure from the dataset
        src_nids, dst_nids = dataset.graph.edges()
        nfeats: th.Tensor = dataset.graph.ndata["feat"]  # type: ignore
        efeats: th.Tensor | None = dataset.graph.edata.get("feat")
        self.ginfo = GraphInfo(
            src_nids=src_nids, dst_nids=dst_nids, nfeats=nfeats, efeats=efeats
        )

        # Load all node/edge masks for the range of snapshots
        # Each mask indicates whether a node/edge exists in a specific snapshot
        self.nid_getter = IDGetter(
            dataset.df_nodes.select(cs.starts_with("mask_")).to_torch().T
        )
        self.eid_getter = IDGetter(
            dataset.df_edges.select(cs.starts_with("mask_")).to_torch().T
        )

        # Initialize state tracking variables
        self.prev_ir: SnapshotIR

        # Set up iteration range
        self.curr_index = start
        self.end = end if end else len(dataset)

    def __next__(self):
        if self.curr_index >= self.end:
            raise StopIteration

        if not hasattr(self, "prev_ir"):
            ir = create_ir(
                self.curr_index, self.ginfo, self.nid_getter, self.eid_getter
            )
            compute_eids = None
        else:
            ir = update_ir(
                self.curr_index,
                self.ginfo,
                self.nid_getter,
                self.eid_getter,
                self.prev_ir,
            )
            compute_eids = find_compute_eids(
                self.prev_ir.idx, self.curr_index, self.ginfo, self.eid_getter, ir
            )

        snapshot = build_graph(ir)
        self.prev_ir = ir
        self.curr_index += 1
        return snapshot, compute_eids

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()

    def __del__(self):
        attrs_to_clear = [
            "prev_sorted_ids",
            "prev_nfeats",
            "prev_efeats",
            "prev_src",
            "prev_dst",
            "prev_nids",
            "prev_eids",
            "node_masks",
            "edge_masks",
        ]

        for attr in attrs_to_clear:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, None)

        gc.collect()
        th.cuda.empty_cache()
