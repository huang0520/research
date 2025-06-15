import cProfile
from copy import deepcopy
import time

import torch as th
from torch.autograd import profiler
from torch.cuda import synchronize
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data.dataloader import DataLoader

from research.dataset import EllipticTxTx, Epinions, RedditBody
from research.loader import AsyncPipeline, PackageProcessor
from research.model import TGCN
from research.transform import edge_life

th.backends.cudnn.enabled = True
th.backends.cudnn.benchmark = True


class Profiler:
    def __init__(
        self,
        baseline_model,
        our_no_cache_model,
        our_model,
        baseline_dataloader,
        our_dataloader,
    ) -> None:
        self.baseline = baseline_model
        self.our_no_cache = our_no_cache_model
        self.our = our_model
        self.baseline_loader = baseline_dataloader
        self.our_loader = our_dataloader
        self.pipeline = AsyncPipeline(our_dataloader, "cuda")

    def run_baseline(self, warmup=True):
        compose_fn = PackageProcessor()
        if warmup:
            for _ in range(2):
                hn = None
                with th.no_grad():
                    for pack in self.baseline_loader:
                        pack_ = pack.to("cuda")
                        _, data = compose_fn(pack_)
                        _, hn = self.baseline(
                            data["n"]["x"], data["n", "e", "n"]["edge_index"], hn
                        )
            synchronize()

        start = time.perf_counter()
        hn = None
        with th.no_grad():
            for pack in self.baseline_loader:
                pack_ = pack.to("cuda")
                _, data = compose_fn(pack_)
                _, hn = self.baseline(
                    data["n"]["x"], data["n", "e", "n"]["edge_index"], hn
                )
        synchronize()
        end = time.perf_counter()
        return end - start

    def run_our_no_cache(self, warmup=True):
        if warmup:  # noqa: PLR1702
            for _ in range(2):
                hn = None
                with th.cuda.stream(self.pipeline.compute_stream):  # type:ignore
                    # with th.amp.autocast("cuda"):  # type:ignore
                    with th.no_grad():
                        for _, data, _ in self.pipeline:
                            _, hn = self.our_no_cache(
                                data["n"]["x"],
                                data["n", "e", "n"]["edge_index"],
                                hn,
                            )
            synchronize()

        start = time.perf_counter()
        hn = None
        with th.cuda.stream(self.pipeline.compute_stream):  # type:ignore
            # with th.amp.autocast("cuda"):  # type:ignore
            with th.no_grad():
                for _, data, _ in self.pipeline:
                    _, hn = self.our_no_cache(
                        data["n"]["x"],
                        data["n", "e", "n"]["edge_index"],
                        hn,
                    )
        synchronize()
        end = time.perf_counter()
        return end - start

    def run_our(self, warmup=True):
        if warmup:  # noqa: PLR1702
            for _ in range(2):
                hn = None
                with th.cuda.stream(self.pipeline.compute_stream):  # type:ignore
                    # with th.amp.autocast("cuda"):  # type:ignore
                    with th.no_grad():
                        for _, data, compute_info in self.pipeline:
                            _, hn = self.our(
                                data["n"]["x"],
                                data["n", "e", "n"]["edge_index"],
                                hn,
                                compute_info,
                            )
            synchronize()

        start = time.perf_counter()
        hn = None
        with th.cuda.stream(self.pipeline.compute_stream):  # type:ignore
            # with th.amp.autocast("cuda"):  # type:ignore
            with th.no_grad():
                for _, data, compute_info in self.pipeline:
                    _, hn = self.our(
                        data["n"]["x"],
                        data["n", "e", "n"]["edge_index"],
                        hn,
                        compute_info,
                    )
        synchronize()
        end = time.perf_counter()
        return end - start

    def compare_performance(self, n_runs=10):
        """Compare overall performance with statistical significance"""
        print("\n=== Performance Comparison ===")

        baseline_times = []
        our_no_cache_times = []
        our_times = []

        for i in range(n_runs):
            th.cuda.empty_cache()
            baseline_times.append(self.run_baseline())

            th.cuda.empty_cache()
            our_no_cache_times.append(self.run_our_no_cache())

            th.cuda.empty_cache()
            our_times.append(self.run_our())

            print(
                f"Run {i + 1}/{n_runs}: "
                f"Baseline={baseline_times[-1]:.4f}s, "
                f"Ours w/o cache = {our_no_cache_times[-1]:.4f}s, "
                f"Ours={our_times[-1]:.4f}s"
            )

        baseline_mean = sum(baseline_times) / len(baseline_times)
        our_no_cache_means = sum(our_no_cache_times) / len(our_no_cache_times)
        our_means = sum(our_times) / len(our_times)
        baseline_std = th.std(th.tensor(baseline_times)).item()
        our_no_cache_std = th.std(th.tensor(our_no_cache_times)).item()
        our_std = th.std(th.tensor(our_times)).item()

        print()
        print(f"Baseline: {baseline_mean:.4f}s ± {baseline_std:.4f}s")
        print(f"Our w/o cache: {our_no_cache_means:.4f}s ± {our_no_cache_std:.4f}s")
        print(f"Our: {our_means:.4f}s ± {our_std:.4f}s")
        print(f"Speedup: {baseline_mean / our_means:.2f}x")


def main():
    # Your existing setup code here...
    our_dataset = EllipticTxTx(incremental=True)
    our_dataset = edge_life(our_dataset, life=5)
    our_dataloader = DataLoader(
        our_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        collate_fn=lambda batch: batch[0],
        pin_memory=True,
    )
    baseline_dataset = EllipticTxTx(incremental=False)
    baseline_dataset = edge_life(baseline_dataset, life=7)
    baseline_dataloader = DataLoader(
        baseline_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        collate_fn=lambda batch: batch[0],
        pin_memory=True,
    )
    max_nodes = max(
        len(meta.gids[our_dataset._data.edge_types[0][2]])
        for meta in our_dataset._metadata.values()
    )
    baseline = TGCN(182, 3, 100, True, cache_gconv=False).to("cuda")
    our_no_cache = TGCN(182, 3, 100, True, cache_gconv=False).to("cuda")
    our = TGCN(182, 3, 100, gcn_norm=True, cache_gconv=True, max_nodes=max_nodes).to(
        "cuda"
    )

    pprofiler = Profiler(
        baseline,
        our_no_cache,
        our,
        baseline_dataloader,
        our_dataloader,
    )
    pprofiler.compare_performance()


if __name__ == "__main__":
    main()
