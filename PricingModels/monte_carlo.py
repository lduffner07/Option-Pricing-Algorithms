"""
Monte Carlo pricing for European call and put options.

Optimization: simulate terminal stock prices in small, fixed-size batches
instead of holding every simulated price and payoff in memory at once.
Time complexity: O(n_paths)
Auxiliary space complexity: O(1) with respect to n_paths, because the
maximum batch size is fixed at 10,000 paths.
"""

import numpy as np


# Maximum number of simulated prices held in memory at one time.
BATCH_SIZE = 10_000


def _simulate_terminal_prices(S, T, sigma, r, count, rng):
    """Generate ONE batch of terminal stock prices under risk-neutral GBM."""
    Z = rng.standard_normal(count)

    # Exact terminal-price formula for geometric Brownian motion.
    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z
    return S * np.exp(drift + diffusion)


def _monte_carlo_price(S, K, T, sigma, r, n_paths, seed, option_type):
    """Average payoffs across batches, without saving all paths."""
    if not isinstance(n_paths, (int, np.integer)) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    # Create the random-number generator ONCE so batches continue the
    # same reproducible sequence of draws when a seed is supplied.
    rng = np.random.default_rng(seed)
    total_payoff = 0.0

    for start in range(0, n_paths, BATCH_SIZE):
        # The final batch may be smaller than BATCH_SIZE.
        count = min(BATCH_SIZE, n_paths - start)
        terminal_prices = _simulate_terminal_prices(S, T, sigma, r, count, rng)

        if option_type == "call":
            payoffs = np.maximum(terminal_prices - K, 0.0)
        else:
            payoffs = np.maximum(K - terminal_prices, 0.0)

        # Add the batch's payoff total. Do not keep the individual paths.
        total_payoff += float(payoffs.sum())

    average_payoff = total_payoff / n_paths
    return float(np.exp(-r * T) * average_payoff)


def monte_carlo_call(S, K, T, sigma, r, n_paths=10000, seed=None):
    """Price a European call; existing arguments remain unchanged."""
    return _monte_carlo_price(S, K, T, sigma, r, n_paths, seed, "call")


def monte_carlo_put(S, K, T, sigma, r, n_paths=10000, seed=None):
    """Price a European put; existing arguments remain unchanged."""
    return _monte_carlo_price(S, K, T, sigma, r, n_paths, seed, "put")