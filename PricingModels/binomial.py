"""
Cox-Ross-Rubinstein (CRR) binomial tree pricing.

Optimization: use one list of option values and update it in place.
Time complexity: O(steps^2)
Auxiliary space complexity: O(steps)
"""

import math

# Prices European call/put options
def _binomial_price(S, K, T, sigma, r, steps, option_type):
    """Price an option by working backward through a binomial tree."""
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    # One interval of the binomial tree.
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))   # factor for an upward stock move
    d = 1 / u                             # factor for a downward move
    p = (math.exp(r * dt) - d) / (u - d)   # risk-neutral up probability
    discount = math.exp(-r * dt)

    # Calculate the option payoff at expiry.
    # We keep ONLY option values, rather than a separate list of stock prices.
    values = []
    for up_moves in range(steps + 1):
        down_moves = steps - up_moves
        stock_price = S * (u ** up_moves) * (d ** down_moves)

        if option_type == "call":
            payoff = max(stock_price - K, 0.0)
        else:
            payoff = max(K - stock_price, 0.0)

        values.append(payoff)

    # Work backward from expiry to the present.
    # Each value is replaced in the SAME list; no new list per time step.
    for time_step in range(steps - 1, -1, -1):
        for up_moves in range(time_step + 1):
            down_value = values[up_moves]
            up_value = values[up_moves + 1]
            continuation_value = discount * (p * up_value + (1 - p) * down_value)


            # European options cannot be exercised early.
            values[up_moves] = continuation_value

    return values[0]

# Pricing a European call option
def binomial_call(S, K, T, sigma, r, steps=100):
    return _binomial_price(S, K, T, sigma, r, steps, "call")


# Pricing a European put option
def binomial_put(S, K, T, sigma, r, steps=100):
    return _binomial_price(S, K, T, sigma, r, steps, "put")