import argparse
import math
import time
from pathlib import Path

import pandas as pd

from options import EuropeanCall, EuropeanPut

BASE_DIR = Path(__file__).resolve().parent
REQUIRED_COLUMNS = (
    "scenario_id", "ticker", "date", "S", "K", "T", "sigma", "r", "option_type"
)


def load_scenarios(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Scenario file not found: {path}")
    df = pd.read_csv(path, nrows=limit)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("The scenario CSV is empty.")
    if df["scenario_id"].isna().any() or df["scenario_id"].duplicated().any():
        raise ValueError("Scenario IDs must be present and unique.")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required columns must not contain missing values.")
    for col in ("S", "K", "T", "sigma", "r"):
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not df[col].map(math.isfinite).all():
            raise ValueError(f"{col} has non-finite values.")
    for col in ("S", "K", "T", "sigma"):
        if (df[col] <= 0).any():
            raise ValueError(f"{col} must be positive.")
    if not df["option_type"].isin(["call", "put"]).all():
        raise ValueError("option_type must be 'call' or 'put' (lowercase).")
    return df


def moneyness_label(option_type: str, ratio: float) -> str:
    # K/S = 1 is ATM. Low strikes are ITM calls but OTM puts.
    if math.isclose(ratio, 1.0, rel_tol=0, abs_tol=1e-8):
        return "ATM"
    if option_type == "call":
        return "ITM" if ratio < 1.0 else "OTM"
    return "ITM" if ratio > 1.0 else "OTM"


def timed_price(option, method: str, **kwargs) -> tuple[float, float]:
    """Return (price, elapsed seconds), timing only the pricing call."""
    start_ns = time.perf_counter_ns()
    price = float(option.price(method=method, **kwargs))
    elapsed = (time.perf_counter_ns() - start_ns) / 1e9
    return price, elapsed


def run_batch(scenarios: pd.DataFrame, steps: int, paths: int, seed: int,
              progress_every: int = 1000) -> tuple[pd.DataFrame, float]:
    rows = []
    start = time.perf_counter()
    total = len(scenarios)

    for index, row in enumerate(scenarios.itertuples(index=False), start=1):
        kind = str(row.option_type)
        klass = EuropeanCall if kind == "call" else EuropeanPut
        option = klass(S=float(row.S), K=float(row.K), T=float(row.T),
                       sigma=float(row.sigma), r=float(row.r))

        try:
            bs, bs_s = timed_price(option, "black_scholes")
            bt, bt_s = timed_price(option, "binomial", steps=steps)
            mc, mc_s = timed_price(option, "monte_carlo", n_paths=paths,
                                   seed=seed + int(row.scenario_id))
        except Exception as exc:
            raise RuntimeError(f"Pricing failed at scenario_id={row.scenario_id}") from exc

        b_error = abs(bt - bs)
        m_error = abs(mc - bs)
        ratio = float(row.K) / float(row.S)
        rows.append({
            "scenario_id": row.scenario_id,
            "ticker": row.ticker,
            "date": row.date,
            "S": float(row.S),
            "K": float(row.K),
            "T": float(row.T),
            "sigma": float(row.sigma),
            "r": float(row.r),
            "option_type": kind,
            "moneyness": moneyness_label(kind, ratio),
            "maturity_days": round(float(row.T) * 365),
            "bs_price": bs,
            "binomial_price": bt,
            "mc_price": mc,
            "binomial_error": b_error,
            "mc_error": m_error,
            "binomial_error_pct_spot": b_error / float(row.S) * 100,
            "mc_error_pct_spot": m_error / float(row.S) * 100,
            "bs_time_s": bs_s,
            "binomial_time_s": bt_s,
            "mc_time_s": mc_s,
        })
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(f"Processed {index:,} / {total:,} scenarios", flush=True)

    return pd.DataFrame(rows), time.perf_counter() - start


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Long format makes it easy to calculate equivalent metrics for each method.
    shared = results[["scenario_id", "option_type", "moneyness", "maturity_days"]]
    method_frames = []
    for method, prefix in (("binomial", "binomial"), ("monte_carlo", "mc")):
        subset = shared.copy()
        subset["method"] = method
        subset["absolute_error"] = results[f"{prefix}_error"]
        subset["error_pct_spot"] = results[f"{prefix}_error_pct_spot"]
        subset["squared_error"] = subset["absolute_error"] ** 2
        subset["time_s"] = results[f"{prefix}_time_s"]
        method_frames.append(subset)
    long = pd.concat(method_frames, ignore_index=True)

    def aggregate(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        grouped = frame.groupby(columns, as_index=False, sort=False).agg(
            scenarios=("scenario_id", "count"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            mean_squared_error=("squared_error", "mean"),
            mean_error_pct_spot=("error_pct_spot", "mean"),
            mean_time_s=("time_s", "mean"),
            median_time_s=("time_s", "median"),
            total_pricing_time_s=("time_s", "sum"),
        )
        grouped["rmse"] = grouped.pop("mean_squared_error").pow(0.5)
        return grouped

    overall = aggregate(long, ["method"])
    by_group = aggregate(long, ["method", "option_type", "moneyness", "maturity_days"])
    return overall, by_group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/pricing_scenarios.csv",
        help="Scenario CSV, relative to the project root if not absolute"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Read only the first N scenarios (e.g. 1000 for a pilot)")
    parser.add_argument("--steps", type=int, default=100, help="Binomial tree steps")
    parser.add_argument("--paths", type=int, default=10_000, help="Monte Carlo paths per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Base Monte Carlo random seed")
    parser.add_argument("--tag", default=None, help="Prefix of output file names")
    parser.add_argument("--output-dir", default="results", help="Output folder")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.steps < 1 or args.paths < 1 or args.seed < 0:
        parser.error("--steps and --paths must be positive; --seed cannot be negative")

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(input_path, args.limit)
    tag = args.tag or f"{input_path.stem}_{len(scenarios)}"
    print(f"Loaded {len(scenarios):,} scenarios from {input_path}")
    print(f"Binomial steps: {args.steps}; Monte Carlo paths: {args.paths}")

    results, wall_time_s = run_batch(scenarios, args.steps, args.paths, args.seed)
    summary, breakdown = summarize(results)
    paths = {
        "pricing results": output_dir / f"{tag}_pricing_results.csv",
        "overall summary": output_dir / f"{tag}_summary.csv",
        "group summary": output_dir / f"{tag}_by_group.csv",
    }
    results.to_csv(paths["pricing results"], index=False)
    summary.to_csv(paths["overall summary"], index=False)
    breakdown.to_csv(paths["group summary"], index=False)

    print("\nOverall benchmark (price error in dollars):")
    print(summary[["method", "scenarios", "mean_absolute_error", "rmse",
                   "median_time_s", "total_pricing_time_s"]].to_string(index=False,
                                                                         float_format=lambda x: f"{x:.6f}"))
    print(f"\nTotal wall-clock time (including processing): {wall_time_s:.3f} seconds")
    print("\nSaved:")
    for title, path in paths.items():
        print(f"  {title}: {path}")
    print("\nNote: timings are per pricing call, with one run per scenario. "
          "For more reliable runtime comparisons, see efficiency_analysis.py.")
if __name__ == "__main__":
    main()
