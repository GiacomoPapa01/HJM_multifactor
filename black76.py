"""
Black-76 pricing primitives.

The Black-76 formula prices European options on futures. It's the standard
adaptation of Black-Scholes when the underlying has no drift under the
risk-neutral measure (which is the case for futures, by the cost-of-carry
argument).

Convention used throughout this codebase
----------------------------------------
The `sigma` argument is always the **aggregated** volatility

    sigma_hat = sqrt( int_0^T Sigma^2(u) du )

NOT the standard annualised vol. In the simplest constant-vol case this is
just `sigma_ann * sqrt(T)`. We make this choice because the HJM pricers
naturally produce sigma_hat as their output, and it removes the need to
re-divide by sqrt(T) every time we evaluate Black-76.

If you have market-quoted annualised vols (matrix indexed by T x K), call
`market_prices_from_iv` which handles the conversion in one place.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black76(forward: float, strike: float, discount: float, sigma_hat: float) -> float:
    """
    Black-76 call price for a single option.

    Parameters
    ----------
    forward    : current forward price F(0, T)
    strike     : strike K
    discount   : discount factor for maturity T (i.e. P(0,T))
    sigma_hat  : aggregated volatility = sigma * sqrt(T)

    Returns
    -------
    call price (scalar)
    """
    # d1, d2 written in terms of sigma_hat: cancels the sqrt(T) you'd otherwise
    # see in the textbook formula.
    d1 = (np.log(forward / strike) + 0.5 * sigma_hat ** 2) / sigma_hat
    d2 = d1 - sigma_hat
    return discount * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def black76_grid(forward: float,
                 strikes: np.ndarray,
                 discounts: np.ndarray,
                 sigma_hat: np.ndarray) -> np.ndarray:
    """
    Vectorised Black-76 on a (maturity, strike) grid.

    Parameters
    ----------
    forward   : scalar, F(0, T) (same for all rows since we're on one future)
    strikes   : (n_strikes,)
    discounts : (n_maturities,)
    sigma_hat : (n_maturities,) — aggregated vol at each maturity

    Returns
    -------
    prices : (n_maturities, n_strikes) matrix of call prices
    """
    strikes = np.asarray(strikes).reshape(1, -1)        # (1, K)
    discounts = np.asarray(discounts).reshape(-1, 1)    # (T, 1)
    sigma_hat = np.asarray(sigma_hat).reshape(-1, 1)    # (T, 1)

    d1 = (np.log(forward / strikes) + 0.5 * sigma_hat ** 2) / sigma_hat
    d2 = d1 - sigma_hat
    return discounts * (forward * norm.cdf(d1) - strikes * norm.cdf(d2))


def market_prices_from_iv(forward: float,
                          strikes: np.ndarray,
                          maturities: np.ndarray,
                          discounts: np.ndarray,
                          iv_matrix: np.ndarray) -> np.ndarray:
    """
    Convert an annualised-IV grid into market prices via Black-76.

    The Excel file gives standard-quote implied vols (sigma, not sigma_hat),
    so here we multiply by sqrt(T) row-wise before pricing.

    Parameters
    ----------
    forward    : scalar
    strikes    : (n_strikes,)
    maturities : (n_maturities,) — in years
    discounts  : (n_maturities,)
    iv_matrix  : (n_maturities, n_strikes) — annualised IVs

    Returns
    -------
    prices : (n_maturities, n_strikes)
    """
    sigma_hat = iv_matrix * np.sqrt(np.asarray(maturities).reshape(-1, 1))
    # Same grid call but with row-wise sigma_hat — easier to just inline d1/d2
    strikes_r = np.asarray(strikes).reshape(1, -1)
    discounts_c = np.asarray(discounts).reshape(-1, 1)

    d1 = (np.log(forward / strikes_r) + 0.5 * sigma_hat ** 2) / sigma_hat
    d2 = d1 - sigma_hat
    return discounts_c * (forward * norm.cdf(d1) - strikes_r * norm.cdf(d2))


def mse(observed: np.ndarray, model: np.ndarray) -> float:
    """Mean squared error over the full price matrix."""
    if observed.shape != model.shape:
        raise ValueError(f"Shape mismatch: {observed.shape} vs {model.shape}")
    return float(np.mean((observed - model) ** 2))
