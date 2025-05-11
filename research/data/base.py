from dataclasses import dataclass

from torch import Tensor
from torch_geometric.data import Data


@dataclass
class SnapshotInfo:
    nmask: Tensor
    emask: Tensor
    nid: Tensor
    eid: Tensor


class SubData(Data):
    x: Tensor
    edge_index: Tensor
    gnid: Tensor
    geid: Tensor


class MainData(Data):
    x: Tensor
    edge_index: Tensor
    nmasks: Tensor
    emasks: Tensor
