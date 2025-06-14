import cProfile
import time
from collections import defaultdict

import pandas as pd
import torch as th
from torch.autograd import profiler
from torch.cuda import synchronize
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data.dataloader import DataLoader
from torch_geometric.nn.conv.gcn_conv import GCNConv

from research.compute.cache_manager import (
    create_cached_model,
)
from research.dataset import EllipticTxTx, Epinions, RedditBody
from research.loader import AsyncPipeline
from research.model import TGCN
from research.transform import edge_life
from research.utils import edge_subgraph

th.backends.cudnn.enabled = True
th.backends.cudnn.benchmark = True


class DetailedProfiler:
    def __init__(self, model, cached_model, manager, dataset):
        self.model = model
        self.cached_model = cached_model
        self.manager = manager
        self.dataset = dataset
        self.results = defaultdict(list)

    def profile_memory_transfer(self):
        """Profile just the data transfer overhead"""
        print("=== Memory Transfer Profiling ===")

        # Baseline: Full data transfer
        times_baseline = []
        for _ in range(10):
            th.cuda.empty_cache()
            synchronize()
            start = time.perf_counter()

            for meta in self.manager.context.metadata.values():
                snapshot = edge_subgraph(self.dataset._data, meta.geid)
                _ = snapshot.to("cuda", non_blocking=True)
                synchronize()

            end = time.perf_counter()
            times_baseline.append(end - start)

        # Incremental: Data transfer only
        times_incremental = []
        for _ in range(10):
            th.cuda.empty_cache()
            synchronize()
            start = time.perf_counter()

            for _, snapshot in self.manager.get_generator():
                # Just accessing the data, no computation
                _ = snapshot.x.shape
                _ = snapshot.edge_index.shape

            end = time.perf_counter()
            times_incremental.append(end - start)

        print(
            f"Baseline transfer: {sum(times_baseline) / len(times_baseline):.4f}s ± {th.std(th.tensor(times_baseline)):.4f}s"
        )
        print(
            f"Incremental transfer: {sum(times_incremental) / len(times_incremental):.4f}s ± {th.std(th.tensor(times_incremental)):.4f}s"
        )

        return times_baseline, times_incremental

    def profile_computation_breakdown(self):
        """Detailed breakdown of computation phases"""
        print("\n=== Computation Breakdown ===")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with record_function("incremental_full_run"):
                hn = None
                for i, snapshot in self.manager.get_generator():
                    with record_function(f"snapshot_{i}"):
                        with record_function("model_forward"):
                            _, hn = self.cached_model(
                                snapshot.x, snapshot.edge_index, hn
                            )

        # Print detailed breakdown
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        # Export for further analysis
        prof.export_chrome_trace("incremental_trace.json")

        return prof

    def profile_memory_fragmentation(self):
        """Check memory fragmentation effects"""
        print("\n=== Memory Fragmentation Analysis ===")

        # Profile memory usage patterns
        for method_name, run_method in [
            ("Baseline", self.run_baseline),
            ("Incremental", self.run_incremental),
        ]:
            th.cuda.empty_cache()
            th.cuda.reset_peak_memory_stats()

            th.cuda.memory._record_memory_history(max_entries=1000000)

            start_mem = th.cuda.memory_allocated()
            run_method(warmup=True)
            peak_mem = th.cuda.max_memory_allocated()

            try:
                th.cuda.memory._dump_snapshot(f"tmp/{method_name}.pickle")
            except Exception as e:
                print(f"Failed to capture memory snapshot {e}")

            th.cuda.memory._record_memory_history(enabled=None)

            print(f"{method_name}:")
            print(f"  Start memory: {start_mem / 1e6:.2f} MB")
            print(f"  Peak memory: {peak_mem / 1e6:.2f} MB")
            print(f"  Memory growth: {(peak_mem - start_mem) / 1e6:.2f} MB")

    def profile_tensor_operations(self):
        """Profile individual tensor operations in incremental updates"""
        print("\n=== Tensor Operations Profiling ===")

        # Hook into the incremental update methods
        original_merge = self.manager._merge_new
        original_remove = self.manager._remove_old
        original_compute_info = (
            self.cached_model.gconv._get_compute_info
            if hasattr(self.cached_model, "gconv")
            else None
        )

        merge_times = []
        remove_times = []
        compute_info_times = []

        def timed_merge_new(context, diff_info):
            start = time.perf_counter()
            result = original_merge(context, diff_info)
            merge_times.append(time.perf_counter() - start)
            return result

        def timed_remove_old(context, diff_info):
            start = time.perf_counter()
            result = original_remove(context, diff_info)
            remove_times.append(time.perf_counter() - start)
            return result

        # Monkey patch for timing
        self.manager._merge_new = timed_merge_new
        self.manager._remove_old = timed_remove_old

        # Run incremental method
        self.run_incremental(warmup=False)

        # Restore original methods
        self.manager._merge_new = original_merge
        self.manager._remove_old = original_remove

        if merge_times:
            print(
                f"Merge operations: {len(merge_times)} calls, avg {sum(merge_times) / len(merge_times) * 1000:.2f}ms"
            )
        if remove_times:
            print(
                f"Remove operations: {len(remove_times)} calls, avg {sum(remove_times) / len(remove_times) * 1000:.2f}ms"
            )

    def profile_batch_sizes(self):
        """Analyze the sizes of batches being processed"""
        print("\n=== Batch Size Analysis ===")

        batch_info = []

        # Hook into the cached module to capture batch sizes
        if hasattr(self.cached_model, "gconv"):
            original_forward = self.cached_model.gconv.forward

            def capture_batch_forward(x, edge_index):
                batch_info.append({
                    "nodes": x.shape[0],
                    "edges": edge_index.shape[1],
                    "compute_ratio": 1.0,  # Will be updated if partial computation
                })
                return original_forward(x, edge_index)

            self.cached_model.gconv.forward = capture_batch_forward

            # Run incremental method
            self.run_incremental(warmup=False)

            # Restore
            self.cached_model.gconv.forward = original_forward

            if batch_info:
                df = pd.DataFrame(batch_info)
                print(f"Average nodes per snapshot: {df['nodes'].mean():.1f}")
                print(f"Average edges per snapshot: {df['edges'].mean():.1f}")
                print(f"Node count std: {df['nodes'].std():.1f}")
                print(f"Edge count std: {df['edges'].std():.1f}")

                return df

    def run_baseline(self, warmup=True):
        """Run baseline method"""
        if warmup:
            # Warmup
            for _ in range(2):
                hn = None
                with th.no_grad():
                    for meta in self.manager.context.metadata.values():
                        snapshot = edge_subgraph(self.dataset._data, meta.geid).cuda(
                            non_blocking=True
                        )
                        _, hn = self.model(snapshot.x, snapshot.edge_index, hn)

        synchronize()
        start = time.perf_counter()
        hn = None
        with th.no_grad():
            for meta in self.manager.context.metadata.values():
                snapshot = edge_subgraph(self.dataset._data, meta.geid).cuda(
                    non_blocking=True
                )
                _, hn = self.model(snapshot.x, snapshot.edge_index, hn)
        synchronize()
        end = time.perf_counter()
        return end - start

    def run_incremental(self, warmup=True):
        """Run incremental method"""
        if warmup:
            # Warmup
            for _ in range(2):
                hn = None
                with th.no_grad():
                    for _, snapshot in self.manager.get_generator():
                        # _, hn = self.cached_model(snapshot.x, snapshot.edge_index, hn)
                        _, hn = self.model(snapshot.x, snapshot.edge_index, hn)

        synchronize()
        start = time.perf_counter()
        hn = None
        with th.no_grad():
            for _, snapshot in self.manager.get_generator():
                _, hn = self.cached_model(snapshot.x, snapshot.edge_index, hn)
                # _, hn = self.model(snapshot.x, snapshot.edge_index, hn)
        synchronize()
        end = time.perf_counter()
        return end - start

    def compare_performance(self, n_runs=10):
        """Compare overall performance with statistical significance"""
        print("\n=== Performance Comparison ===")

        baseline_times = []
        incremental_times = []

        for i in range(n_runs):
            th.cuda.empty_cache()
            baseline_times.append(self.run_baseline())

            th.cuda.empty_cache()
            incremental_times.append(self.run_incremental())

            print(
                f"Run {i + 1}/{n_runs}: Baseline={baseline_times[-1]:.4f}s, Incremental={incremental_times[-1]:.4f}s"
            )

        baseline_mean = sum(baseline_times) / len(baseline_times)
        incremental_mean = sum(incremental_times) / len(incremental_times)
        baseline_std = th.std(th.tensor(baseline_times)).item()
        incremental_std = th.std(th.tensor(incremental_times)).item()

        print(f"\nBaseline: {baseline_mean:.4f}s ± {baseline_std:.4f}s")
        print(f"Incremental: {incremental_mean:.4f}s ± {incremental_std:.4f}s")
        print(f"Speedup: {baseline_mean / incremental_mean:.2f}x")

        return baseline_times, incremental_times


def collect_fn(batch):
    assert len(batch) == 1
    return batch[0]


def main():
    # Your existing setup code here...
    dataset = EllipticTxTx()
    dataset = edge_life(dataset, life=2)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        collate_fn=collect_fn,
        pin_memory=True,
    )
    pipeline = AsyncPipeline(dataloader, "cuda")

    model = TGCN(182, 3, 100, gcn_norm=False).to("cuda")
    layer = GCNConv(182, 3, normalize=False).to("cuda")

    for id, data in pipeline:
        print(id)
        hidden = None
        with th.cuda.stream(pipeline.compute_stream):  # type:ignore
            with th.no_grad():
                _, hidden = model(
                    data["node"]["x"],
                    data["node", "edge", "node"]["edge_index"],
                    hidden,
                )

    breakpoint()


if __name__ == "__main__":
    run_detailed_profiling()
