# Comparing Binomial Tree and Monte Carlo Algorithms for Options Pricing

**COP3530 Project 2 — Google Shallowmind**  
**Team Members:** Kainoa Mauch, Lindsey Duffner, Ido Ben David  
[GitHub Repository](https://github.com/lduffner07/Option-Pricing-Algorithms) | [Video Demonstration](https://youtu.be/wivhKHtSZQw)

## Overview

We implemented two option-pricing algorithms from scratch in Python: the **Cox–Ross–Rubinstein Binomial Tree** and **Monte Carlo simulation**. Both price European call and put options. We compare their numerical accuracy, execution time, and memory usage using the **Black–Scholes formula** as the theoretical benchmark.

The project includes an interactive menu, historical financial data, batch benchmarking, convergence and efficiency graphs, memory analysis, and automated tests.

## Requirements and Setup

Install **Python 3.10 or newer** and the required libraries. In a terminal opened in the main project folder, run:

```bash
python -m pip install numpy pandas scipy matplotlib yfinance pandas-datareader
```

The `yfinance` and `pandas-datareader` packages are only needed to regenerate historical data; the provided CSVs can be used directly.

## Run the Program

```bash
python main.py
```

The interactive menu contains six options:

1. **Load Historical Dataset:** Load `data/pricing_scenarios.csv` (all 100,080 scenarios, or a smaller number of rows).
2. **Run Binomial Tree:** Price a European call or put using manual inputs or a loaded historical scenario; choose the tree depth.
3. **Run Monte Carlo:** Price a European call or put; choose the number of simulated paths and random seed.
4. **Compare Algorithms:** Display Black–Scholes, Binomial Tree, and Monte Carlo prices, absolute errors, and runtimes for the same option.
5. **Performance Benchmark:** Run a runtime benchmark on loaded scenarios or run the separate memory benchmark. Runtime results are saved to `results/menu_*.csv`.
6. **Exit:** Close the program.

Numbers in brackets are **default values**. Press Enter to use the default, or type a replacement.

**Example:** Select option `2`, choose `call`, and enter `S=100`, `K=90`, `T=1`, `sigma=0.2`, `r=0.05`, and `100` tree steps. The program displays the option price and runtime.

## Historical Data

We generated **100,080 hypothetical European option scenarios** using stock prices from Yahoo Finance and Treasury yields from FRED:

- 40 companies across 11 GICS sectors and 139 selected historical dates per company.
- Calls and puts, strike prices at 90%, 100%, or 110% of the stock price, and expirations of 30, 90, or 180 days.
- Each row includes `scenario_id`, `ticker`, `date`, `S`, `K`, `T`, `sigma`, `r`, and `option_type`.

Volatility is estimated from historical returns, and Treasury rates are matched to maturity. These are **hypothetical option scenarios, not observed market option prices**; accuracy is measured against Black–Scholes. The dataset is already provided, so downloading market data again is unnecessary.

## Reproduce the Experiments

Run these commands from the repository's main folder.

### Full Batch Benchmark (100,080 scenarios)

```bash
python batch_benchmark.py --tag full
```

Runs each scenario with **100 tree steps** and **10,000 Monte Carlo paths** by default. Outputs:

- `results/full_pricing_results.csv` — scenario-level prices, errors, and runtimes.
- `results/full_summary.csv` — overall accuracy and timing summary.
- `results/full_by_group.csv` — breakdown by call/put, moneyness, and maturity.

For a quick demonstration without processing the entire dataset:

```bash
python batch_benchmark.py --limit 20 --tag demo
```

### Historical Convergence and Efficiency (1,440 scenarios)

```bash
python efficiency_analysis.py
```

Uses a balanced random sample: two historical scenarios per combination of **40 stocks × 2 option types × 3 strike ratios × 3 expirations**. It tests seven tree depths (`10` to `1000` steps) and six Monte Carlo path counts (`1000` to `500000`), averaging Monte Carlo absolute errors over three seeds.

Outputs include `historical_efficiency_details.csv`, `historical_efficiency_summary.csv`, `historical_accuracy_targets.csv`, the selected scenarios CSV, and six graphs in `results/`:

- `binomial_historical_convergence.png` and `monte_carlo_historical_convergence.png`
- `binomial_historical_runtime.png` and `monte_carlo_historical_runtime.png`
- `historical_accuracy_vs_runtime.png` and `historical_normalized_accuracy_vs_runtime.png`

**Note:** The current `--quick` option overwrites the same output filenames as a full run. Avoid running `python efficiency_analysis.py --quick` after generating your final results unless you intend to regenerate the full experiment afterward.

### Memory Benchmark

```bash
python memory_benchmark.py
```

Measures peak traced memory allocations for European calls and puts over increasing tree steps and simulated paths. Outputs include `memory_details.csv`, `memory_summary.csv`, `binomial_memory.png`, `monte_carlo_memory.png`, and `memory_analysis.md` in `results/`. Memory is measured with `tracemalloc`, **not total process RAM usage**.

### Automated Tests

```bash
python -m unittest -v test_pricing
```

Tests numerical accuracy, edge cases, invalid inputs, Monte Carlo random-seed reproducibility and batching, put–call parity, menu integration, and memory output generation. The file `data/pricing_scenarios_sample.csv` is needed for a menu integration test.

## Algorithm Complexities

| Algorithm | Time | Auxiliary space |
| --- | --- | --- |
| Binomial Tree | `O(N²)` | `O(N)` |
| Monte Carlo | `O(M)` | `O(1)` relative to total paths M with fixed-size batches |
| Black–Scholes | `O(1)` | `O(1)` |

`N` is the number of tree steps; `M` is the number of simulated paths. The Binomial Tree updates one list of `N + 1` payoff values in place. Monte Carlo processes at most **10,000 paths per batch**, adding payoffs to a running sum instead of retaining every path.

## Main Results

Across the **100,080-scenario baseline benchmark**, Binomial Tree had a mean absolute error of approximately **$0.01109**, compared with **$0.08352** for Monte Carlo (at 100 steps and 10,000 paths, respectively). In the separate **1,440-scenario efficiency experiment**, Binomial Tree achieved lower average error at comparable measured runtimes over the configurations tested. Memory and runtime trends were consistent with the theoretical complexities above.

These findings concern the tested European option settings under Black–Scholes assumptions; they do not establish that one algorithm is always better for every pricing problem.

## Project Structure

```text
Option-Pricing-Algorithms/
├── PricingModels/
│   ├── black_scholes.py
│   ├── binomial.py
│   └── monte_carlo.py
├── data/
│   ├── pricing_scenarios.csv
│   └── pricing_scenarios_sample.csv
├── results/                  # Experiment CSVs and graphs
├── options.py                # EuropeanCall and EuropeanPut classes
├── main.py                   # Interactive menu
├── batch_benchmark.py        # 100,080-scenario experiment
├── efficiency_analysis.py    # Historical convergence and efficiency
├── memory_benchmark.py       # Space-complexity measurements
├── test_pricing.py           # Automated tests
└── README.md
```

The historical data collection script and additional raw-data CSVs may also be included in the repository.

## Team Contributions

- **Kainoa Mauch:** Pricing algorithms, option classes, full batch benchmarking, and historical convergence/efficiency analysis.
- **Lindsey Duffner:** Historical data collection, cleaning, volatility and Treasury-rate processing, and scenario generation.
- **Ido Ben David:** Menu interface, automated unit/integration tests, memory benchmarking, and memory graphs.

## Data Sources

[Yahoo Finance](https://finance.yahoo.com/) and FRED Treasury series [DGS1MO](https://fred.stlouisfed.org/series/DGS1MO), [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO), and [DGS6MO](https://fred.stlouisfed.org/series/DGS6MO).
