# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from math import ceil


def _estimate_bench_iter(f, tuple_of_args):
    warmup_iter_guess = 5
    min_round_time_ms = 100
    rounds = 5
    warmup_rounds = 1

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(warmup_iter_guess):
        f(*tuple_of_args)
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / warmup_iter_guess

    main_iter = ceil(min_round_time_ms / elapsed)

    return warmup_rounds, main_iter, rounds


def _time_ms(f, tuple_of_args, warmup: int, iters: int, rounds: int) -> float:
    for _ in range(warmup):
        f(*tuple_of_args)

    run_iters = max(iters, rounds)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(run_iters):
        f(*tuple_of_args)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end)
    return ms / max(1, run_iters)


def report_benchmark(f, tuple_of_args) -> dict[str, float]:
    warmup_rounds, iterations, rounds = _estimate_bench_iter(f, tuple_of_args)
    mean_time_ms = _time_ms(f, tuple_of_args, warmup_rounds, iterations, rounds)
    return {"mean_time_ms": mean_time_ms}