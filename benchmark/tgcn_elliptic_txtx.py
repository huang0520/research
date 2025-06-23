import time

import torch as th
from torch.utils.data.dataloader import DataLoader

from benchmark.base import BenchmarkConfig, ModelConfig
from benchmark.elliptic_txtx import no_reduce, reduce_both, reduce_edge
from research.loader import AsyncPipeline, PackageProcessor
from research.model import TGCN


def run_baseline_fn(model: th.nn.Module, dataloader: DataLoader):
    model.eval()
    compose_fn = PackageProcessor()

    start = time.perf_counter()
    hn = None
    with th.no_grad():
        for pack in dataloader:
            pack_ = pack.to("cuda")
            _, data = compose_fn(pack_)
            _, hn = model(data["n"]["x"], data["n", "e", "n"]["edge_index"], hn)
    th.cuda.synchronize()
    end = time.perf_counter()
    return end - start


def run_pipeline_no_cache_fn(model: th.nn.Module, dataloader: DataLoader):
    model.eval()
    pipeline = AsyncPipeline(dataloader, "cuda")

    start = time.perf_counter()
    hn = None
    with th.cuda.stream(pipeline.compute_stream):  # type:ignore
        with th.no_grad():
            for _, data, _ in pipeline:
                _, hn = model(data["n"]["x"], data["n", "e", "n"]["edge_index"], hn)
    th.cuda.synchronize()
    end = time.perf_counter()
    return end - start


def run_pipeline_fn(model: th.nn.Module, dataloader: DataLoader):
    model.eval()
    pipeline = AsyncPipeline(dataloader, "cuda")

    start = time.perf_counter()
    hn = None
    with th.cuda.stream(pipeline.compute_stream):  # type:ignore
        with th.no_grad():
            for _, data, compute_info in pipeline:
                _, hn = model(
                    data["n"]["x"], data["n", "e", "n"]["edge_index"], hn, compute_info
                )
    th.cuda.synchronize()
    end = time.perf_counter()
    return end - start


normal = ModelConfig(
    name="TGCN_Normal",
    model_fn=lambda _: TGCN(182, 32, 100, gcn_norm=False, cache_gconv=False).to("cuda"),
)

cached = ModelConfig(
    name="TGCN_Cached",
    model_fn=lambda max_nodes: TGCN(
        182, 32, 100, gcn_norm=False, cache_gconv=True, max_nodes=max_nodes
    ).to("cuda"),
)

baseline_config = BenchmarkConfig(
    name="TGCN_EllipticTxTx_Baseline",
    dataset_config=no_reduce,
    model_config=normal,
    run_fn=run_baseline_fn,
)

previous_config = BenchmarkConfig(
    name="TGCN_EllipticTxTx_Previous",
    dataset_config=reduce_edge,
    model_config=normal,
    run_fn=run_pipeline_no_cache_fn,
)

ours_no_cache_config = BenchmarkConfig(
    name="TGCN_EllipticTxTx_Ours_NoCache",
    dataset_config=reduce_both,
    model_config=normal,
    run_fn=run_pipeline_no_cache_fn,
)

ours_config = BenchmarkConfig(
    name="TGCN_EllipticTxTx_Ours",
    dataset_config=reduce_both,
    model_config=cached,
    run_fn=run_pipeline_fn,
)
