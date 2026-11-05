"""Measure peak traced allocations, not whole-process resident memory."""

import argparse
import gc
import platform
import time
import tracemalloc
from numbers import Integral
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from options import EuropeanCall, EuropeanPut
from PricingModels.monte_carlo import BATCH_SIZE

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TREE_STEPS = (25, 50, 100, 200, 500, 1000)
MC_PATHS = (1000, 5000, 10000, 20000, 100000, 500000, 1000000)


def validate_count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def verify_numpy_tracking():
    """Refuse misleading measurements if NumPy buffers are not traced."""
    tracemalloc.start()
    try:
        buffer = np.empty(100000, dtype=np.float64)
        _, peak = tracemalloc.get_traced_memory()
        if peak < buffer.nbytes:
            raise RuntimeError("This NumPy build does not expose buffers to tracemalloc.")
    finally:
        tracemalloc.stop()


def measure_peak(option, method, effort, seed):
    kwargs = {"steps": effort} if method == "binomial" else {"n_paths": effort, "seed": seed}
    gc.collect()
    tracemalloc.start()
    try:
        start = time.perf_counter()
        price = option.price(method=method, **kwargs)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return float(price), peak, elapsed


def run_benchmark(tree_steps=TREE_STEPS, mc_paths=MC_PATHS, repeats=3,
                  seed=42, output_dir=RESULTS_DIR):
    """Benchmark a controlled call/put pair while varying only effort."""
    if tracemalloc.is_tracing():
        raise RuntimeError("Stop existing tracemalloc tracing before benchmarking.")
    validate_count(repeats, "repeats")
    for effort in (*tree_steps, *mc_paths):
        validate_count(effort, "effort")
    if not tree_steps or not mc_paths:
        raise ValueError("Both effort ranges must be nonempty.")
    verify_numpy_tracking()
    rows = []
    for klass in (EuropeanCall, EuropeanPut):
        option = klass(100.0, 100.0, 1.0, 0.2, 0.05)
        option.price(method="binomial", steps=25)
        option.price(method="monte_carlo", n_paths=BATCH_SIZE, seed=seed)
        for method, efforts in (("binomial", tree_steps), ("monte_carlo", mc_paths)):
            for effort in sorted(set(efforts)):
                for repeat in range(1, repeats + 1):
                    price, peak, elapsed = measure_peak(option, method, effort, seed)
                    rows.append(dict(method=method, option_type=option.option_type,
                                     effort=effort, repeat=repeat, peak_bytes=peak,
                                     traced_runtime_s=elapsed, price=price, seed=seed,
                                     batch_size=BATCH_SIZE, S=100, K=100, T=1,
                                     sigma=0.2, r=0.05))
                print(f"{option.option_type}: {method}, effort={effort:,}", flush=True)
    details = pd.DataFrame(rows)
    summary = details.groupby(["method", "option_type", "effort"], as_index=False).agg(
        median_peak_bytes=("peak_bytes", "median"),
        min_peak_bytes=("peak_bytes", "min"), max_peak_bytes=("peak_bytes", "max"),
        median_traced_runtime_s=("traced_runtime_s", "median"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "memory_details.csv", index=False)
    summary.to_csv(output_dir / "memory_summary.csv", index=False)
    make_graphs(summary, output_dir)
    write_analysis(summary, output_dir, repeats)
    print(f"Saved memory results to {output_dir}")
    return details, summary


def make_graphs(summary, output_dir):
    for method, title, xlabel in (
        ("binomial", "Binomial Tree: Peak Allocated Memory", "Tree steps (N)"),
        ("monte_carlo", "Monte Carlo: Peak Allocated Memory", "Simulated paths (M)"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for kind in ("call", "put"):
            rows = summary[(summary.method == method) & (summary.option_type == kind)]
            axis.plot(rows.effort, rows.median_peak_bytes / 1024, "o-", label=kind.title())
            axis.fill_between(rows.effort, rows.min_peak_bytes / 1024,
                              rows.max_peak_bytes / 1024, alpha=0.12)
        if method == "monte_carlo":
            axis.axvline(BATCH_SIZE, color="gray", linestyle="--", label=f"Batch cap: {BATCH_SIZE:,}")
        axis.set(xscale="linear" if method == "binomial" else "log", xlabel=xlabel,
                 ylabel="Peak traced allocations (KiB)", title=title)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{method}_memory.png", dpi=150)
        plt.close(figure)


def write_analysis(summary, output_dir, repeats):
    lines = ["# Memory benchmarking", "",
             f"Environment: Python {platform.python_version()}, NumPy {np.__version__}, {platform.platform()}.",
             f"Each setting uses {repeats} repetitions; graphs show medians and min/max bands.",
             "Fixed parameters: S=K=100, T=1 year, sigma=0.2, r=0.05; both calls and puts.", "",
             "## Method", "",
             "tracemalloc measures peak allocations during each pricing call after warmup and garbage collection. "
             "Imports, dataset storage, plotting, and result storage are outside the measurement. "
             "A calibration allocation checks that NumPy data buffers are visible. "
             "This is not process RSS; allocator caches and untraced native allocations are excluded. "
             "Recorded runtimes include tracing overhead and must not replace the existing runtime benchmarks.", "",
             "## Observations", ""]
    for kind in ("call", "put"):
        tree = summary[(summary.method == "binomial") & (summary.option_type == kind)].sort_values("effort")
        first, last = tree.iloc[0], tree.iloc[-1]
        lines.append(f"- {kind.title()} tree: {int(first.effort):,} to {int(last.effort):,} steps "
                     f"({last.effort / first.effort:.1f}x effort), "
                     f"{first.median_peak_bytes / 1024:.2f} to {last.median_peak_bytes / 1024:.2f} KiB "
                     f"({last.median_peak_bytes / first.median_peak_bytes:.2f}x memory).")
        if len(tree) >= 2:
            slope, overhead = np.polyfit(tree.effort, tree.median_peak_bytes, 1)
            residual = tree.median_peak_bytes - (slope * tree.effort + overhead)
            variation = ((tree.median_peak_bytes - tree.median_peak_bytes.mean()) ** 2).sum()
            fit = 1 - float((residual ** 2).sum() / variation) if variation else 0.0
            verdict = "consistent with" if slope > 0 and fit >= 0.95 else "not clearly consistent with"
            lines.append(f"  Linear fit: {slope:.2f} bytes/step plus {overhead:.0f} bytes fixed overhead, "
                         f"R-squared={fit:.4f}; {verdict} linear O(N) growth "
                         "over this range (descriptive check, not a statistical test).")
        plateau = summary[(summary.method == "monte_carlo") &
                          (summary.option_type == kind) & (summary.effort >= 2 * BATCH_SIZE)]
        if len(plateau) >= 2:
            lines.append(f"- {kind.title()} Monte Carlo, at least two full batches: "
                         f"{int(plateau.effort.min()):,}–{int(plateau.effort.max()):,} paths, "
                         f"{plateau.median_peak_bytes.min() / 1024:.2f}–"
                         f"{plateau.median_peak_bytes.max() / 1024:.2f} KiB peak memory.")
            spread = plateau.median_peak_bytes.max() / plateau.median_peak_bytes.min()
            verdict = "consistent with a fixed-batch O(1) plateau" if spread <= 1.2 else "not flat over this range"
            lines.append(f"  Peak-memory max/min ratio: {spread:.3f}; {verdict} "
                         "(20% descriptive tolerance, not a statistical test).")
        else:
            lines.append(f"- {kind.title()} Monte Carlo: more settings above two full batches "
                         "are needed to evaluate the plateau.")
    lines.extend(["", "## Interpretation", "",
                  "The tree keeps one list of N+1 payoffs and updates it in place: O(N) auxiliary space. "
                  "A roughly proportional rise in memory supports that prediction; fixed overhead and list capacity "
                  "mean the measured ratios need not match exactly.", "",
                  f"Monte Carlo holds arrays for at most {BATCH_SIZE:,} paths at once. "
                  "Its space is O(min(M, B)) for M paths and batch cap B, hence O(1) in M with B fixed. "
                  "Memory can rise below the cap and on the transition to multiple batches because the previous "
                  "batch can coexist with new temporaries. Beyond that transition, a plateau supports constant "
                  "auxiliary space. Measurements support or challenge these expectations, but do not prove asymptotic bounds.", "",
                  "## Presentation", "",
                  "Use binomial_memory.png and monte_carlo_memory.png for the space-complexity segment. "
                  "Pair them with the existing historical convergence and runtime figures. "
                  "Demo the menu, explain the Black–Scholes reference and test results, then discuss accuracy, "
                  "runtime, memory, and measurement limitations. Coordinate speakers and recording separately.", ""])
    (output_dir / "memory_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=TREE_STEPS)
    parser.add_argument("--paths", type=int, nargs="+", default=MC_PATHS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    if args.seed < 0 or args.repeats < 1 or any(value < 1 for value in (*args.steps, *args.paths)):
        parser.error("Efforts and repeats must be positive; seed must be nonnegative.")
    run_benchmark(args.steps, args.paths, args.repeats, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
