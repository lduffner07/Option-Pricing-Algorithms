from PricingModels.black_scholes import black_scholes_call
from PricingModels.black_scholes import black_scholes_put
from PricingModels.binomial import binomial_call
from PricingModels.binomial import binomial_put
from PricingModels.monte_carlo import monte_carlo_call
from PricingModels.monte_carlo import monte_carlo_put



class Option:
    """
    Base class for all options. Stores the five core parameters every
    pricing model needs, and validates them on creation.
    """

    def __init__(self, S, K, T, sigma, r):
        if S <= 0:
            raise ValueError("Spot price S must be positive.")
        if K <= 0:
            raise ValueError("Strike price K must be positive.")
        if T <= 0:
            raise ValueError("Time to expiry T must be positive.")
        if sigma <= 0:
            raise ValueError("Volatility sigma must be positive.")

        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.r = r

    def price(self, method="black_scholes", **kwargs):
        raise NotImplementedError("Subclasses must implement price().")

class EuropeanCall(Option): # inherits from Option class
    """
    A European call option: the right (not obligation) to BUY the
    underlying at strike K, exercisable only at expiry T.
    """
    option_type = "call"

    def price(self, method="black_scholes", **kwargs):
        if method == "black_scholes":
            return black_scholes_call(self.S, self.K, self.T, self.sigma, self.r)
        elif method == "binomial":
            return binomial_call(self.S, self.K, self.T, self.sigma, self.r, steps=kwargs.get("steps", 100))
        elif method == "monte_carlo":
            return monte_carlo_call(self.S, self.K, self.T, self.sigma, self.r, n_paths=kwargs.get("n_paths", 10000), seed=kwargs.get("seed", None))
        else:
            raise ValueError(f"Unknown pricing method: {method}")

class EuropeanPut(Option):
    """
    A European put option: the right to SELL the underlying
    at strike K, exercisable only at expiration.
    """
    option_type = "put"
    def price(self, method = "black_scholes", **kwargs):
        if method == "black_scholes":
            return black_scholes_put(self.S, self.K, self.T, self.sigma, self.r)
        elif method == "binomial":
            return binomial_put(self.S, self.K, self.T, self.sigma, self.r, steps=kwargs.get("steps", 100))
        elif method == "monte_carlo":
            return monte_carlo_put(self.S, self.K, self.T, self.sigma, self.r, n_paths=kwargs.get("n_paths", 10000), seed=kwargs.get("seed", None))
        else:
            raise ValueError(f"Unknown pricing method: {method}")

