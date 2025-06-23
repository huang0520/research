from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import torch as th
from loguru import logger
from torch.utils.data import DataLoader

from research.dataset import BaseDataset


@dataclass
class DatasetConfig:
    name: str
    dataset_fn: Callable[[], BaseDataset]


@dataclass
class ModelConfig:
    name: str
    model_fn: Callable[[int], th.nn.Module]


@dataclass
class BenchmarkConfig:
    name: str
    dataset_config: DatasetConfig
    model_config: ModelConfig
    run_fn: Callable[[th.nn.Module, DataLoader], float]
    loader_fn: Callable[[BaseDataset], DataLoader] = field(
        default=lambda dataset: DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            collate_fn=lambda batch: batch[0],
            pin_memory=True,
        )
    )


@dataclass
class BenchmarkResult:
    name: str
    model_name: str
    dataset_name: str
    times: list[float]
    mean_time: float
    std_time: float
    median_time: float
    min_time: float
    max_time: float
    memory_peak: float
    memory_allocated: float


class Bench:
    def __init__(self, result_dir: str = "./benchmark_result"):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)

    @contextmanager
    def cleanup(self):  # noqa: PLR6301
        try:
            # Clear cache before run
            th.cuda.empty_cache()
            if th.cuda.is_available():
                th.cuda.reset_peak_memory_stats()
            yield

        finally:
            # Clear cache after run
            th.cuda.empty_cache()

    def run_single_model(
        self, config: BenchmarkConfig, n_runs: int = 10, warmup_runs: int = 2
    ):
        logger.info(f"=== Benchmarking {config.name} ===")

        times = []
        memory_peaks = []
        memory_allocated = []
        for run_idx in range(n_runs + warmup_runs):
            with self.cleanup():
                dataset = config.dataset_config.dataset_fn()

                max_nodes = max(
                    len(meta.gids[dataset._data.edge_types[0][2]])
                    for meta in dataset._metadata.values()
                )

                model = config.model_config.model_fn(max_nodes)
                dataloader = config.loader_fn(dataset)

                # Run the model
                run_time = config.run_fn(model, dataloader)
                peak_mem = th.cuda.max_memory_allocated() / 1024**3
                alloc_mem = th.cuda.memory_allocated() / 1024**3

                if run_idx >= warmup_runs:
                    times.append(run_time)
                    memory_peaks.append(peak_mem)
                    memory_allocated.append(alloc_mem)
                    logger.info(
                        f"Run {run_idx - warmup_runs + 1}/{n_runs}: {run_time:.4f}s"
                    )
                else:
                    logger.info(f"Warmup {run_idx + 1}/{warmup_runs}: {run_time:.4f}s")

                del model, dataloader

        result = self._create_result(config, times, memory_peaks, memory_allocated)
        self.save_result(result)

        return result

    @staticmethod
    def _create_result(
        config: BenchmarkConfig,
        times: list[float],
        memory_peaks: list[float],
        memory_allocated: list[float],
    ):
        rst = pl.DataFrame(
            {
                "name": config.name,
                "model_name": config.model_config.name,
                "dataset_name": config.dataset_config.name,
            },
        )

        times_ = pl.Series("times", times)
        memory_peaks_ = pl.Series("memory_peaks", memory_peaks)
        memory_allocated_ = pl.Series("memory_allocated", memory_allocated)

        return rst.with_columns(
            times=pl.lit(times),
            mean_time=times_.mean(),
            std_time=times_.std(),
            median_time=times_.median(),
            min_time=times_.min(),
            max_time=times_.max(),
            memory_peak=memory_peaks_.mean(),
            memory_allocated=memory_allocated_.mean(),
        )

    def compare_models(
        self, configs: Sequence[BenchmarkConfig], n_runs: int = 10, warmup_runs: int = 2
    ) -> pl.DataFrame:
        """Compare multiple models by running them sequentially."""
        results = pl.DataFrame()

        for config in configs:
            result = self.run_single_model(config, n_runs, warmup_runs)
            results = results.vstack(result)

        self.print_comparison(results)
        self.save_result(results)

        return results

    @staticmethod
    def print_comparison(results: pl.DataFrame):
        """Print comparison table of results."""

        # Sort by mean time for easier comparison
        baseline_time_expr = pl.col("mean_time").first()
        rst_df = results.with_columns(
            speedup=baseline_time_expr / pl.col("mean_time")
        ).select(
            "name",
            "mean_time",
            "std_time",
            "median_time",
            "min_time",
            "max_time",
            "memory_peak",
            "speedup",
        )

        logger.info("=" * 40)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("=" * 40)
        logger.info(rst_df)

    def save_result(self, result: pl.DataFrame):
        if len(result) == 1:
            filename = self.result_dir / f"{result['name'][0]}.json"
        else:
            filename = self.result_dir / "comparison_result.json"
        result.write_json(filename)
