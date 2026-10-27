from scipy.stats import norm
import math

"""
    Price a European call option using the Black-Scholes formula.

    Parameters:
    S : Current price of the underlying asset
    K : Strike price
    T : Time to expiration, in years
    sigma : Volatility of the underlying (annualized, as a decimal)
    r : Risk-free interest rate (annualized, as a decimal)
"""

def _d1_d2(S, K, T, sigma, r):
    """
    Compute the d1 and d2 terms used throughout the Black-Scholes formula.

    d1 = [ln(S/K) + (r + sigma^2 / 2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes_call(S, K, T, sigma, r):
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, T, sigma, r):

    d1, d2 = _d1_d2(S, K, T, sigma, r)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price