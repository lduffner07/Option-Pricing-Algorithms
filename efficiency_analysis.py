"""
Historical convergence and efficiency comparison.

Uses the pricing scenarios made from Yahoo Finance and FRED data.
The same scenarios are tested with Binomial Tree and Monte Carlo.
Black-Scholes is the reference price.

Run a small test first:  python efficiency_analysis.py --quick
Run the full experiment: python efficiency_analysis.py
"""

import argparse
import time
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd

from options import EuropeanCall, EuropeanPut


# Experiment settings. Each method is tested over a RANGE of efforts.
TREE_STEPS = [10, 25, 50, 100, 200, 500, 1000]
MC_PATHS = [1000, 5000, 10000, 50000, 100000, 500000]
MC_SEEDS = 3                  # Three independent Monte Carlo runs per scenario.
DATES_PER_GROUP = 2          # 40 stocks x 2 types x 3 strikes x 3 maturities x 2 = 1440.
RANDOM_SEED = 42
TARGET_ERRORS = [0.10, 0.05, 0.02, 0.01]  # Average dollar error targets.

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "pricing_scenarios.csv"
RESULTS_DIR = BASE_DIR / "results"


def select_scenarios(quick=False):
    """Choose a balanced sample across stocks, calls/puts, strikes, maturities."""
    data = pd.read_csv(DATA_FILE)
    needed = ["scenario_id", "ticker", "date", "S", "K", "T", "sigma", "r", "option_type"]
    for name in needed:
        if name not in data.columns:
            raise ValueError("Missing column in CSV: " + name)

    # Lindsey's data uses strike ratios 0.90, 1.00, 1.10, and 30/90/180 days.
    data["strike_ratio"] = (data["K"] / data["S"]).round(2)
    data["expiry_days"] = (data["T"] * 365).round().astype(int)

    sample_size = DATES_PER_GROUP
    if quick:
        # A small test to check that everything runs before the full experiment.
        first_two_stocks = sorted(data["ticker"].unique())[:2]
        data = data[data["ticker"].isin(first_two_stocks)]
        sample_size = 1

    groups = ["ticker", "option_type", "strike_ratio", "expiry_days"]
    if (data.groupby(groups).size() < sample_size).any():
        raise ValueError("Some groups have too few scenarios for this sample size.")

    # Random sampling with a fixed seed makes the experiment reproducible.
    selected = data.groupby(groups, group_keys=False).sample(
        n=sample_size, random_state=RANDOM_SEED
    )
    return selected.sort_values("scenario_id").reset_index(drop=True)


def timed_price(option, method, steps_or_paths, seed=None):
    """Return (price, runtime in milliseconds) for one pricing call."""
    start = time.perf_counter()
    if method == "binomial":
        price = option.price(method="binomial", steps=steps_or_paths)
    else:
        price = option.price(
            method="monte_carlo", n_paths=steps_or_paths, seed=seed
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return float(price), elapsed_ms


def run_experiment(scenarios, quick=False):
    """Price every selected scenario at several tree depths and path counts."""
    tree_steps = TREE_STEPS
    mc_paths = MC_PATHS
    if quick:
        tree_steps = [25, 100, 200]
        mc_paths = [1000, 10000, 50000]

    details = []
    total = len(scenarios)

    # Warm up both algorithms once before recording measurements.
    first = scenarios.iloc[0]
    warmup = EuropeanCall(first["S"], first["K"], first["T"], first["sigma"], first["r"])
    warmup.price(method="black_scholes")
    warmup.price(method="binomial", steps=50)
    warmup.price(method="monte_carlo", n_paths=1000, seed=RANDOM_SEED)

    for i, row in scenarios.iterrows():
        if row["option_type"] == "call":
            option = EuropeanCall(row["S"], row["K"], row["T"], row["sigma"], row["r"])
        else:
            option = EuropeanPut(row["S"], row["K"], row["T"], row["sigma"], row["r"])

        bs_price = float(option.price(method="black_scholes"))
        scenario_id = int(row["scenario_id"])

        # Binomial pricing is deterministic, so one result is enough for error.
        for steps in tree_steps:
            price, runtime = timed_price(option, "binomial", steps)
            error = abs(price - bs_price)
            details.append({
                "scenario_id": scenario_id,
                "ticker": row["ticker"],
                "option_type": row["option_type"],
                "strike_ratio": row["strike_ratio"],
                "expiry_days": row["expiry_days"],
                "method": "binomial",
                "steps_or_paths": steps,
                "black_scholes_price": bs_price,
                "model_price": price,
                "absolute_error": error,
                "error_pct_of_stock": 100 * error / row["S"],
                "runtime_ms": runtime,
            })

        # Monte Carlo is random: average ABSOLUTE errors from several seeds.
        for paths in mc_paths:
            prices = []
            errors = []
            runtimes = []
            for trial in range(MC_SEEDS):
                # A different seed for each scenario and trial, but repeatable.
                seed = RANDOM_SEED + scenario_id * 10 + trial
                price, runtime = timed_price(option, "monte_carlo", paths, seed)
                prices.append(price)
                errors.append(abs(price - bs_price))
                runtimes.append(runtime)

            average_error = mean(errors)
            details.append({
                "scenario_id": scenario_id,
                "ticker": row["ticker"],
                "option_type": row["option_type"],
                "strike_ratio": row["strike_ratio"],
                "expiry_days": row["expiry_days"],
                "method": "monte_carlo",
                "steps_or_paths": paths,
                "black_scholes_price": bs_price,
                "model_price": mean(prices),
                "absolute_error": average_error,
                "error_pct_of_stock": 100 * average_error / row["S"],
                # Mean time for ONE Monte Carlo run, not all three combined.
                "runtime_ms": mean(runtimes),
            })

        if (i + 1) % 100 == 0 or i + 1 == total:
            print("Completed", i + 1, "of", total, "pricing scenarios.", flush=True)

    return pd.DataFrame(details)


def make_summary(details):
    """One summary row for each algorithm and step/path count."""
    summary = details.groupby(["method", "steps_or_paths"]).agg(
        scenarios=("scenario_id", "count"),
        mean_absolute_error=("absolute_error", "mean"),
        mean_error_pct_stock=("error_pct_of_stock", "mean"),
        mean_runtime_ms=("runtime_ms", "mean"),
        median_runtime_ms=("runtime_ms", "median"),
    ).reset_index()
    return summary.sort_values(["method", "steps_or_paths"])


def make_accuracy_targets(summary):
    """Find the fastest tested configuration using median runtime."""
    rows = []
    for target in TARGET_ERRORS:
        for method in ["binomial", "monte_carlo"]:
            choices = summary[
                (summary["method"] == method) &
                (summary["mean_absolute_error"] <= target)
            ].sort_values("median_runtime_ms")

            if len(choices) == 0:
                rows.append({
                    "target_mae_dollars": target, "method": method,
                    "steps_or_paths": "not reached", "median_runtime_ms": "",
                    "mean_absolute_error": "",
                })
            else:
                best = choices.iloc[0]
                rows.append({
                    "target_mae_dollars": target, "method": method,
                    "steps_or_paths": int(best["steps_or_paths"]),
                    "median_runtime_ms": best["median_runtime_ms"],
                    "mean_absolute_error": best["mean_absolute_error"],
                })
    return pd.DataFrame(rows)


def save_plot(x, y, title, xlabel, ylabel, filename):
    """Save one simple line graph."""
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xscale("log")
    if min(y) > 0:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close()


def make_graphs(summary):
    """Convergence, runtime, and direct efficiency comparisons."""
    for method, name, unit in [
        ("binomial", "Binomial Tree", "tree steps (N)"),
        ("monte_carlo", "Monte Carlo", "simulation paths (M)"),
    ]:
        rows = summary[summary["method"] == method]
        save_plot(
            rows["steps_or_paths"], rows["mean_absolute_error"],
            name + ": Historical Convergence",
            "Number of " + unit, "Mean absolute error vs. Black-Scholes ($)",
            method + "_historical_convergence.png",
        )
        save_plot(
            rows["steps_or_paths"], rows["median_runtime_ms"],
            name + ": Historical Runtime",
            "Number of " + unit, "Median runtime per option (ms)",
            method + "_historical_runtime.png",
        )

    # Use MEDIAN measured runtime to reduce the effect of slow timing outliers.
    # Both methods now use the SAME x-axis: measured runtime per option.
    plt.figure(figsize=(8, 5))
    for method, label in [("binomial", "Binomial Tree"),
                          ("monte_carlo", "Monte Carlo")]:
        rows = summary[summary["method"] == method]
        plt.plot(rows["median_runtime_ms"], rows["mean_absolute_error"],
                 marker="o", label=label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Median runtime per option (ms)")
    plt.ylabel("Mean absolute error vs. Black-Scholes ($)")
    plt.title("Historical Scenarios: Accuracy vs. Runtime")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "historical_accuracy_vs_runtime.png", dpi=150)
    plt.close()

    # Percentage of stock price helps when stocks have very different prices.
    plt.figure(figsize=(8, 5))
    for method, label in [("binomial", "Binomial Tree"),
                          ("monte_carlo", "Monte Carlo")]:
        rows = summary[summary["method"] == method]
        plt.plot(rows["median_runtime_ms"], rows["mean_error_pct_stock"],
                 marker="o", label=label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Median runtime per option (ms)")
    plt.ylabel("Mean absolute error as % of stock price")
    plt.title("Historical Scenarios: Price-Normalized Error vs. Runtime")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "historical_normalized_accuracy_vs_runtime.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Historical pricing efficiency experiment")
    parser.add_argument("--quick", action="store_true", help="Small test with 2 stocks")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_scenarios(quick=args.quick)
    print("Selected", len(selected), "scenarios from",
          selected["ticker"].nunique(), "companies.", flush=True)
    selected.to_csv(RESULTS_DIR / "historical_selected_scenarios.csv", index=False)

    details = run_experiment(selected, quick=args.quick)
    summary = make_summary(details)
    targets = make_accuracy_targets(summary)

    details.to_csv(RESULTS_DIR / "historical_efficiency_details.csv", index=False)
    summary.to_csv(RESULTS_DIR / "historical_efficiency_summary.csv", index=False)
    targets.to_csv(RESULTS_DIR / "historical_accuracy_targets.csv", index=False)
    make_graphs(summary)

    print("\nSUMMARY (same historical scenarios for both algorithms):")
    print(summary.round(5).to_string(index=False))
    print("\nTARGET ERRORS (fastest tested setting reaching each target):")
    print(targets.to_string(index=False))
    print("\nSaved CSV files and six graphs in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
