# Memory benchmarking

Environment: Python 3.14.5, NumPy 2.4.6, macOS-26.5.2-arm64-arm-64bit-Mach-O.
Each setting uses 3 repetitions; graphs show medians and min/max bands.
Fixed parameters: S=K=100, T=1 year, sigma=0.2, r=0.05; both calls and puts.

## Method

tracemalloc measures peak allocations during each pricing call after warmup and garbage collection. Imports, dataset storage, plotting, and result storage are outside the measurement. A calibration allocation checks that NumPy data buffers are visible. This is not process RSS; allocator caches and untraced native allocations are excluded. Recorded runtimes include tracing overhead and must not replace the existing runtime benchmarks.

## Observations

- Call tree: 25 to 1,000 steps (40.0x effort), 2.40 to 33.72 KiB (14.06x memory).
  Linear fit: 32.88 bytes/step plus 1594 bytes fixed overhead, R-squared=1.0000; consistent with linear O(N) growth over this range (descriptive check, not a statistical test).
- Call Monte Carlo, at least two full batches: 20,000–1,000,000 paths, 472.25–472.25 KiB peak memory.
  Peak-memory max/min ratio: 1.000; consistent with a fixed-batch O(1) plateau (20% descriptive tolerance, not a statistical test).
- Put tree: 25 to 1,000 steps (40.0x effort), 2.38 to 33.70 KiB (14.19x memory).
  Linear fit: 32.88 bytes/step plus 1570 bytes fixed overhead, R-squared=1.0000; consistent with linear O(N) growth over this range (descriptive check, not a statistical test).
- Put Monte Carlo, at least two full batches: 20,000–1,000,000 paths, 472.25–472.25 KiB peak memory.
  Peak-memory max/min ratio: 1.000; consistent with a fixed-batch O(1) plateau (20% descriptive tolerance, not a statistical test).

## Interpretation

The tree keeps one list of N+1 payoffs and updates it in place: O(N) auxiliary space. A roughly proportional rise in memory supports that prediction; fixed overhead and list capacity mean the measured ratios need not match exactly.

Monte Carlo holds arrays for at most 10,000 paths at once. Its space is O(min(M, B)) for M paths and batch cap B, hence O(1) in M with B fixed. Memory can rise below the cap and on the transition to multiple batches because the previous batch can coexist with new temporaries. Beyond that transition, a plateau supports constant auxiliary space. Measurements support or challenge these expectations, but do not prove asymptotic bounds.

## Presentation

Use binomial_memory.png and monte_carlo_memory.png for the space-complexity segment. Pair them with the existing historical convergence and runtime figures. Demo the menu, explain the Black–Scholes reference and test results, then discuss accuracy, runtime, memory, and measurement limitations. Coordinate speakers and recording separately.
