"""Interactive interface to the existing European option pricing tools."""

import math
from pathlib import Path

from batch_benchmark import BASE_DIR, load_scenarios, run_batch, summarize, timed_price
from options import EuropeanCall, EuropeanPut


def read_number(prompt, default, integer=False, minimum=None):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        try:
            value = (int if integer else float)(raw or str(default))
            if not math.isfinite(value) or (minimum is not None and value < minimum):
                raise ValueError
            return value
        except ValueError:
            bound = f", at least {minimum}" if minimum is not None else ""
            print(f"Enter a finite {'integer' if integer else 'number'}{bound}.")


def read_option(scenarios):
    if scenarios is not None:
        source = input("Use a loaded scenario? [y/N]: ").strip().lower()
        if source == "y":
            while True:
                scenario_id = read_number("Scenario ID", int(scenarios.iloc[0].scenario_id), True, 1)
                rows = scenarios[scenarios.scenario_id == scenario_id]
                if not rows.empty:
                    row = rows.iloc[0]
                    klass = EuropeanCall if row.option_type == "call" else EuropeanPut
                    print(f"Using {row.ticker}, {row.date}, {row.option_type}")
                    return klass(float(row["S"]), float(row["K"]), float(row["T"]),
                                 float(row["sigma"]), float(row["r"]))
                print("That scenario ID is not in the loaded dataset.")
    while True:
        kind = input("Option type (call/put) [call]: ").strip().lower() or "call"
        if kind in ("call", "put"):
            break
        print("Enter call or put.")
    spot = read_number("Spot price", 100)
    strike = read_number("Strike price", 100)
    maturity = read_number("Time to expiry (years)", 1, minimum=0)
    volatility = read_number("Annual volatility (decimal, e.g. 0.2)", 0.2, minimum=0)
    rate = read_number("Annual risk-free rate (decimal)", 0.05)
    klass = EuropeanCall if kind == "call" else EuropeanPut
    return klass(spot, strike, maturity, volatility, rate)


def pricing_settings():
    steps = read_number("Tree steps", 100, True, 1)
    paths = read_number("Monte Carlo paths", 10000, True, 1)
    seed = read_number("Random seed", 42, True, 0)
    return steps, paths, seed


def compare(option):
    steps, paths, seed = pricing_settings()
    reference = float(option.price(method="black_scholes"))
    print(f"{'Method':<16} {'Price':>12} {'Absolute error':>16} {'Runtime (ms)':>16}")
    for method, kwargs in (("black_scholes", {}), ("binomial", {"steps": steps}),
                           ("monte_carlo", {"n_paths": paths, "seed": seed})):
        price, elapsed = timed_price(option, method, **kwargs)
        print(f"{method:<16} {price:12.6f} {abs(price - reference):16.6f} {elapsed * 1000:16.3f}")
    print("Error is relative to Black–Scholes, not observed market prices; timings are single runs.")


def performance_benchmark(scenarios):
    kind = input("Benchmark type (runtime/memory) [runtime]: ").strip().lower() or "runtime"
    if kind == "memory":
        from memory_benchmark import run_benchmark

        repeats = read_number("Memory measurement repetitions", 3, True, 1)
        run_benchmark(repeats=repeats)
        return
    if kind != "runtime":
        raise ValueError("Choose runtime or memory.")
    if scenarios is None:
        raise ValueError("Load the historical dataset first (menu option 1).")
    limit = read_number("Number of loaded scenarios to benchmark", min(20, len(scenarios)), True, 1)
    if limit > len(scenarios):
        raise ValueError(f"Only {len(scenarios):,} scenarios are loaded.")
    steps, paths, seed = pricing_settings()
    results, elapsed = run_batch(scenarios.iloc[:limit], steps, paths, seed)
    summary, groups = summarize(results)
    output_dir = BASE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("pricing_results", results), ("summary", summary), ("by_group", groups)):
        frame.to_csv(output_dir / f"menu_{name}.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wall time: {elapsed:.3f} s. Saved results/menu_*.csv (replaced on each run).")


def main():
    scenarios = None
    while True:
        print("\nEuropean Option Pricing\n1. Load Historical Dataset\n2. Run Binomial Tree"
              "\n3. Run Monte Carlo\n4. Compare Algorithms\n5. Performance Benchmark\n6. Exit")
        try:
            choice = input("Select an option: ").strip()
            if choice == "6":
                print("Goodbye!")
                return
            if choice == "1":
                raw = input("Dataset CSV [data/pricing_scenarios.csv]: ").strip()
                path = Path(raw or "data/pricing_scenarios.csv").expanduser()
                if not path.is_absolute():
                    path = BASE_DIR / path
                limit = read_number("Rows to load (0 = all)", 0, True, 0)
                loaded = load_scenarios(path, limit or None)
                scenarios = loaded
                print(f"Loaded {len(scenarios):,} scenarios from {path}")
                print(scenarios.head().to_string(index=False))
            elif choice in ("2", "3", "4"):
                option = read_option(scenarios)
                if choice == "4":
                    compare(option)
                else:
                    method = "binomial" if choice == "2" else "monte_carlo"
                    if choice == "2":
                        kwargs = {"steps": read_number("Tree steps", 100, True, 1)}
                    else:
                        kwargs = {"n_paths": read_number("Monte Carlo paths", 10000, True, 1),
                                  "seed": read_number("Random seed", 42, True, 0)}
                    price, elapsed = timed_price(option, method, **kwargs)
                    print(f"{method} {option.option_type}: ${price:.6f} ({elapsed * 1000:.3f} ms)")
            elif choice == "5":
                performance_benchmark(scenarios)
            else:
                print("Choose an option from 1 to 6.")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        except (ValueError, OSError, RuntimeError, OverflowError) as exc:
            print(f"Unable to complete request: {exc}")


if __name__ == "__main__":
    main()
