from benchmark import tgcn_elliptic_txtx, tgcn_redditbody
from benchmark.base import Bench, BenchmarkConfig

TGCN_ELLIPTIC_TXTX: tuple[BenchmarkConfig, ...] = (
    tgcn_elliptic_txtx.baseline_config,
    tgcn_elliptic_txtx.previous_config,
    tgcn_elliptic_txtx.ours_no_cache_config,
    tgcn_elliptic_txtx.ours_config,
)

TGCN_REDDITBODY: tuple[BenchmarkConfig, ...] = (
    tgcn_redditbody.baseline_config,
    tgcn_redditbody.previous_config,
    tgcn_redditbody.ours_no_cache_config,
    tgcn_redditbody.ours_config,
)
