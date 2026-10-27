"""
Convergence analysis & performance benchmarking across pricing models.

Two questions this answers:

1. ACCURACY -- as we throw more computational effort at Binomial
   (more steps) and Monte Carlo (more simulated paths), how much
   closer do they get to the Black-Scholes reference price? This is
   "convergence": both methods are approximations that should
   approach the exact answer as effort increases.

2. PERFORMANCE -- how much does that extra accuracy cost in runtime?
   Binomial and Monte Carlo have very different cost/accuracy
   tradeoffs, and this makes that tradeoff visible instead of
   theoretical.

Black-Scholes itself has no "convergence" dimension -- it's an exact
closed-form formula, so it's used here only as the reference price
everything else is measured against, and as a performance baseline
(since it's just one calculation, it should be dramatically faster
than either approximation method).
"""

import time
import csv
from options import EuropeanCall


def _time_price(option, method, **kwargs):
    """
    Price an option and measure how long it took, using
    time.perf_counter() -- the standard tool for timing short-running
    code in Python (higher resolution and more reliable than
    time.time() for this purpose).
    """
    start = time.perf_counter()
    price = option.price(method=method, **kwargs)
    elapsed = time.perf_counter() - start
    return price, elapsed


def run_benchmark(option, steps_list=(10, 25, 50, 100, 200, 500, 1000),
                   paths_list=(1_000, 5_000, 10_000, 50_000, 100_000, 500_000),
                   seed=42):
    """
    Run Black-Scholes (as the reference), then Binomial across
    `steps_list` and Monte Carlo across `paths_list`, recording price,
    absolute error vs. the Black-Scholes reference, and time taken
    for each run.

    Returns a list of dicts, one per pricing run:
        {"method", "param", "price", "error", "time"}
    "param" is the step count / path count used (None for Black-Scholes).
    """
    results = []

    # Black-Scholes is our ground truth for European options -- every
    # other method's accuracy is measured as distance from this price.
    reference_price, bs_time = _time_price(option, method="black_scholes")
    results.append({
        "method": "black_scholes", "param": None,
        "price": reference_price, "error": 0.0, "time": bs_time,
    })

    # Binomial: re-price at increasing step counts. More steps means
    # a finer-grained tree (each node closer to its neighbors), which
    # should mean the price gets closer to Black-Scholes.
    for steps in steps_list:
        price, elapsed = _time_price(option, method="binomial", steps=steps)
        results.append({
            "method": "binomial", "param": steps,
            "price": price, "error": abs(price - reference_price), "time": elapsed,
        })

    # Monte Carlo: re-price at increasing path counts. More simulated
    # paths means the average payoff is a better estimate of the true
    # expected payoff (law of large numbers), so error should shrink
    # -- but noisily, unlike Binomial's smooth convergence, since
    # each run still involves randomness.
    for paths in paths_list:
        price, elapsed = _time_price(option, method="monte_carlo", n_paths=paths, seed=seed)
        results.append({
            "method": "monte_carlo", "param": paths,
            "price": price, "error": abs(price - reference_price), "time": elapsed,
        })

    return results


def print_table(results):
    """
    Print the benchmark results as a simple aligned text table --
    no extra dependency needed beyond the standard library.
    """
    header = f"{'Method':<15}{'Steps/Paths':>12}{'Price':>12}{'Error':>14}{'Time (s)':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        param = r["param"] if r["param"] is not None else "-"
        print(f"{r['method']:<15}{str(param):>12}{r['price']:>12.4f}{r['error']:>14.6f}{r['time']:>12.6f}")


def save_csv(results, path="benchmark_results.csv"):
    """
    Save the raw results to CSV so they can be reopened later,
    dropped into a spreadsheet, or re-plotted without rerunning
    the (slow) Monte Carlo simulations.
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "param", "price", "error", "time"])
        writer.writeheader()
        writer.writerows(results)


def plot_convergence(results, save_path="convergence.png"):
    """
    Plot absolute error vs. steps/paths for Binomial and Monte Carlo,
    both axes log-scaled -- convergence behavior is usually easiest
    to read as roughly straight lines on a log-log plot.
    """
    import matplotlib.pyplot as plt

    binomial = [r for r in results if r["method"] == "binomial"]
    monte_carlo = [r for r in results if r["method"] == "monte_carlo"]

    plt.figure()
    plt.plot([r["param"] for r in binomial], [r["error"] for r in binomial],
              marker="o", label="Binomial (steps)")
    plt.plot([r["param"] for r in monte_carlo], [r["error"] for r in monte_carlo],
              marker="o", label="Monte Carlo (paths)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Steps / Paths (log scale)")
    plt.ylabel("Absolute error vs Black-Scholes (log scale)")
    plt.title("Convergence: pricing error vs. computational effort")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_performance(results, save_path="performance.png"):
    """
    Plot time taken vs. steps/paths -- shows the actual runtime cost
    behind the accuracy gains in the convergence plot.
    """
    import matplotlib.pyplot as plt

    binomial = [r for r in results if r["method"] == "binomial"]
    monte_carlo = [r for r in results if r["method"] == "monte_carlo"]

    plt.figure()
    plt.plot([r["param"] for r in binomial], [r["time"] for r in binomial],
              marker="o", label="Binomial (steps)")
    plt.plot([r["param"] for r in monte_carlo], [r["time"] for r in monte_carlo],
              marker="o", label="Monte Carlo (paths)")
    plt.xscale("log")
    plt.xlabel("Steps / Paths (log scale)")
    plt.ylabel("Time (seconds)")
    plt.title("Computational performance vs. steps/paths")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # A representative example option to benchmark. Change these
    # values to benchmark a different option.
    option = EuropeanCall(S=100, K=105, T=1, sigma=0.2, r=0.05)

    results = run_benchmark(option)
    print_table(results)
    save_csv(results)
    plot_convergence(results)
    plot_performance(results)
    print("\nSaved: benchmark_results.csv, convergence.png, performance.png")