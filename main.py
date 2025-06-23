import sys

import torch as th
from loguru import logger

from benchmark import TGCN_ELLIPTIC_TXTX, TGCN_REDDITBODY, Bench

th.backends.cudnn.enabled = True
th.backends.cudnn.benchmark = False


def main():
    clean_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )
    logger.remove()
    logger.add(sys.stdout, level="INFO", format=clean_format)

    # bench_configs = TGCN_ELLIPTIC_TXTX
    bench_configs = TGCN_REDDITBODY
    bench = Bench()
    bench.compare_models(bench_configs, n_runs=10)


if __name__ == "__main__":
    main()
