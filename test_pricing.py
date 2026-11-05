"""Run with python -m unittest -v test_pricing."""

import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from PricingModels.binomial import binomial_call, binomial_put
from PricingModels.black_scholes import black_scholes_call, black_scholes_put
from PricingModels.monte_carlo import BATCH_SIZE, monte_carlo_call, monte_carlo_put
from options import EuropeanCall, EuropeanPut

FUNCTIONS = (binomial_call, binomial_put, monte_carlo_call, monte_carlo_put,
             black_scholes_call, black_scholes_put)
SCENARIOS = ((100, 100, 1, 0.2, 0.05), (90, 100, 0.5, 0.3, 0.02),
             (120, 100, 2, 0.25, -0.01))


class PricingTests(unittest.TestCase):
    def test_black_scholes_known_values(self):
        self.assertAlmostEqual(black_scholes_call(*SCENARIOS[0]), 10.450583572185565, places=10)
        self.assertAlmostEqual(black_scholes_put(*SCENARIOS[0]), 5.573526022256971, places=10)

    def test_binomial_accuracy(self):
        for parameters in SCENARIOS:
            for tree, reference in ((binomial_call, black_scholes_call),
                                    (binomial_put, black_scholes_put)):
                with self.subTest(parameters=parameters, kind=tree.__name__):
                    self.assertAlmostEqual(tree(*parameters, steps=1000),
                                           reference(*parameters), delta=0.025)

    def test_monte_carlo_accuracy(self):
        for parameters in SCENARIOS:
            for simulation, reference in ((monte_carlo_call, black_scholes_call),
                                          (monte_carlo_put, black_scholes_put)):
                with self.subTest(parameters=parameters, kind=simulation.__name__):
                    self.assertAlmostEqual(simulation(*parameters, n_paths=500000, seed=42),
                                           reference(*parameters), delta=0.25)

    def test_binomial_convergence(self):
        for tree, reference in ((binomial_call, black_scholes_call),
                                (binomial_put, black_scholes_put)):
            expected = reference(*SCENARIOS[0])
            self.assertLess(abs(tree(*SCENARIOS[0], steps=500) - expected),
                            abs(tree(*SCENARIOS[0], steps=25) - expected))

    def test_expiration_payoffs(self):
        for spot in (80, 100, 120):
            for function in FUNCTIONS:
                with self.subTest(spot=spot, function=function.__name__):
                    expected = max(spot - 100, 0) if function.__name__.endswith("call") else max(100 - spot, 0)
                    self.assertEqual(function(spot, 100, 0, 0.2, 0.05), expected)

    def test_zero_volatility(self):
        for rate in (-0.05, 0, 0.05):
            for spot in (80, 100, 120):
                for function in FUNCTIONS:
                    with self.subTest(rate=rate, spot=spot, function=function.__name__):
                        difference = spot - 100 * math.exp(-rate)
                        expected = max(difference if function.__name__.endswith("call") else -difference, 0)
                        self.assertAlmostEqual(function(spot, 100, 1, 0, rate), expected)

    def test_zero_volatility_at_expiration(self):
        for function in FUNCTIONS:
            expected = 10 if function.__name__.endswith("call") else 0
            self.assertEqual(function(110, 100, 0, 0, 0.05), expected)

    def test_invalid_parameters(self):
        invalid_values = {0: (0, -1, True, "100", None), 1: (0, -1),
                          2: (-1,), 3: (-0.2,)}
        for index in range(5):
            for invalid in (*invalid_values.get(index, ()), math.nan, math.inf, -math.inf):
                parameters = list(SCENARIOS[0])
                parameters[index] = invalid
                for function in (*FUNCTIONS, EuropeanCall, EuropeanPut):
                    with self.subTest(function=function.__name__, index=index, invalid=invalid):
                        with self.assertRaises(ValueError):
                            function(*parameters)

    def test_invalid_effort(self):
        for function in (binomial_call, binomial_put, monte_carlo_call, monte_carlo_put):
            for effort in (0, -1, 1.5, True, "100", None):
                with self.subTest(function=function.__name__, effort=effort):
                    with self.assertRaises(ValueError):
                        function(*SCENARIOS[0], effort)

    def test_numpy_integer_effort(self):
        self.assertGreater(binomial_call(*SCENARIOS[0], steps=np.int64(10)), 0)
        self.assertGreater(monte_carlo_put(*SCENARIOS[0], n_paths=np.int64(10), seed=42), 0)

    def test_invalid_seed(self):
        for function in (monte_carlo_call, monte_carlo_put):
            for seed in (-1, 1.5, True, "42"):
                with self.subTest(function=function.__name__, seed=seed):
                    with self.assertRaises(ValueError):
                        function(*SCENARIOS[0], seed=seed)

    def test_invalid_tree_probability(self):
        with self.assertRaisesRegex(ValueError, "increase"):
            binomial_call(100, 100, 1, 0.01, 0.5, steps=1)

    def test_seed_reproducibility_across_batches(self):
        for function in (monte_carlo_call, monte_carlo_put):
            for paths in (1, BATCH_SIZE - 1, BATCH_SIZE, 2 * BATCH_SIZE + 7):
                with self.subTest(function=function.__name__, paths=paths):
                    first = function(*SCENARIOS[0], n_paths=paths, seed=123)
                    np.random.seed(987)
                    self.assertEqual(first, function(*SCENARIOS[0], n_paths=paths, seed=123))
            self.assertNotEqual(function(*SCENARIOS[0], seed=123),
                                function(*SCENARIOS[0], seed=456))

    def test_partial_batch_matches_one_random_stream(self):
        paths = 2 * BATCH_SIZE + 7
        spot, strike, maturity, volatility, rate = SCENARIOS[0]
        draws = np.random.default_rng(42).standard_normal(paths)
        terminal = spot * np.exp((rate - volatility ** 2 / 2) * maturity
                                 + volatility * math.sqrt(maturity) * draws)
        for function, payoffs in ((monte_carlo_call, np.maximum(terminal - strike, 0)),
                                  (monte_carlo_put, np.maximum(strike - terminal, 0))):
            expected = math.exp(-rate * maturity) * float(payoffs.mean())
            self.assertAlmostEqual(function(*SCENARIOS[0], n_paths=paths, seed=42), expected, places=12)

    def test_simulation_batches_are_bounded(self):
        import PricingModels.monte_carlo as simulation

        for function in (monte_carlo_call, monte_carlo_put):
            with patch.object(simulation, "_simulate_terminal_prices",
                              wraps=simulation._simulate_terminal_prices) as generate:
                function(*SCENARIOS[0], n_paths=3 * BATCH_SIZE + 7, seed=42)
            self.assertEqual([call.args[4] for call in generate.call_args_list],
                             [BATCH_SIZE, BATCH_SIZE, BATCH_SIZE, 7])

    def test_put_call_parity(self):
        for parameters in SCENARIOS:
            spot, strike, maturity, _, rate = parameters
            expected = spot - strike * math.exp(-rate * maturity)
            self.assertAlmostEqual(black_scholes_call(*parameters) - black_scholes_put(*parameters),
                                   expected, places=10)
            self.assertAlmostEqual(binomial_call(*parameters, steps=200)
                                   - binomial_put(*parameters, steps=200), expected, places=9)

    def test_option_dispatch(self):
        for klass, functions in ((EuropeanCall, (black_scholes_call, binomial_call, monte_carlo_call)),
                                  (EuropeanPut, (black_scholes_put, binomial_put, monte_carlo_put))):
            option = klass(*SCENARIOS[0])
            for method, function in zip(("black_scholes", "binomial", "monte_carlo"), functions):
                kwargs = {"seed": 42} if method == "monte_carlo" else {}
                self.assertEqual(option.price(method=method, **kwargs), function(*SCENARIOS[0], **kwargs))
            with self.assertRaises(ValueError):
                option.price(method="unknown")
            self.assertEqual(klass(110, 100, 0, 0, 0.05).price(), 10 if klass is EuropeanCall else 0)


class IntegrationTests(unittest.TestCase):
    def test_menu_pricing_loading_comparison_and_runtime(self):
        import main as interface

        dataset = Path(__file__).resolve().parent / "data" / "pricing_scenarios_sample.csv"
        answers = ["invalid", "5", "runtime", "1", str(dataset), "2",
                   "2", "y", "1", "25", "3", "y", "1", "1000", "42",
                   "4", "y", "1", "25", "1000", "42",
                   "5", "runtime", "2", "25", "1000", "42", "6"]
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch("builtins.input", side_effect=answers), patch("sys.stdout", output), \
                    patch.object(interface, "BASE_DIR", Path(directory)):
                interface.main()
            self.assertTrue((Path(directory) / "results/menu_summary.csv").is_file())
            self.assertIn("Loaded 2 scenarios", output.getvalue())
            self.assertIn("Absolute error", output.getvalue())
            self.assertIn("Load the historical dataset first", output.getvalue())
            self.assertIn("Goodbye!", output.getvalue())

    def test_menu_manual_pricing_and_eof(self):
        import main as interface

        answers = ["2", "put", "100", "110", "0", "0", "0.05", "100", "6"]
        output = io.StringIO()
        with patch("builtins.input", side_effect=answers), patch("sys.stdout", output):
            interface.main()
        self.assertIn("$10.000000", output.getvalue())
        with patch("builtins.input", side_effect=EOFError), patch("sys.stdout", io.StringIO()):
            interface.main()

    def test_numeric_input_retries(self):
        from main import read_number

        with patch("builtins.input", side_effect=["nan", "inf", "-1", "abc", "3"]), \
                patch("sys.stdout", io.StringIO()):
            self.assertEqual(read_number("Count", 1, integer=True, minimum=1), 3)

    def test_memory_outputs_and_menu_dispatch(self):
        import main as interface
        from memory_benchmark import run_benchmark

        with tempfile.TemporaryDirectory() as directory, patch("sys.stdout", io.StringIO()):
            details, summary = run_benchmark(tree_steps=(5, 10), mc_paths=(1000, 20000, 30000),
                                             repeats=1, output_dir=directory)
            self.assertEqual(len(details), 10)
            self.assertEqual(len(summary), 10)
            self.assertTrue((summary.median_peak_bytes > 0).all())
            for filename in ("memory_details.csv", "memory_summary.csv", "memory_analysis.md",
                             "binomial_memory.png", "monte_carlo_memory.png"):
                self.assertGreater((Path(directory) / filename).stat().st_size, 0)
        with patch("builtins.input", side_effect=["memory", "1"]), \
                patch("memory_benchmark.run_benchmark") as benchmark:
            interface.performance_benchmark(None)
        benchmark.assert_called_once_with(repeats=1)


if __name__ == "__main__":
    unittest.main()
