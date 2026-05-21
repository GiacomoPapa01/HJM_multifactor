"""
HJM pricers for the three calibration setups.

Each function takes the model parameters and returns
    (prices_matrix, sigma_hat_per_maturity)
where prices_matrix has shape (n_maturities, n_strikes) and sigma_hat is the
aggregated volatility used for Black-76 at each maturity.

The three setups share the same skeleton: build sigma_hat(T), call Black-76.
What differs is how sigma_hat is constructed from the parameters.

Mathematical recap
------------------
The HJM aggregated variance at maturity T is
    sigma_hat^2(T) = int_0^T Sigma_1^2(u) du  +  int_0^T Sigma_2^2(u) du

Case 1 — constant volatilities (Σ₁, Σ₂ scalars):
    sigma_hat^2(T) = (Σ₁^2 + Σ₂^2) · T

Case 2 — Σ₁ time-dependent, Σ₂ constant:
    sigma_hat^2(T_n) = sum_{k≤n} I1_k  +  Σ₂^2 · T_n
    where I1_k = ∫_{T_{k-1}}^{T_k} Σ₁²(u) du (parameter)

Case 3 — both time-dependent:
    sigma_hat^2(T_n) = sum_{k≤n} I1_k + sum_{k≤n} I2_k

We parametrise the *increments* I_k rather than the volatility curve itself
because (a) they're what enters the pricing formula, (b) non-negativity of
I_k automatically gives a monotonically increasing sigma_hat^2, which is
what no-arbitrage requires.
"""

from __future__ import annotations

import numpy as np

from black76 import black76_grid


def price_constant_vol(sigma1: float,
                       sigma2: float,
                       forward: float,
                       strikes: np.ndarray,
                       maturities: np.ndarray,
                       discounts: np.ndarray):
    """
    Pricer for the constant-volatility model (Exercise 3).
    Parameters Σ₁ and Σ₂ are scalars; sigma_hat scales as sqrt(T).
    """
    T = np.asarray(maturities)
    sigma_hat = np.sqrt((sigma1 ** 2 + sigma2 ** 2) * T)
    prices = black76_grid(forward, strikes, discounts, sigma_hat)
    return prices, sigma_hat


def price_one_time_dependent(vol1_increments: np.ndarray,
                             sigma2_const: float,
                             forward: float,
                             strikes: np.ndarray,
                             maturities: np.ndarray,
                             discounts: np.ndarray):
    """
    Pricer for the Σ₁(t)-time-dependent / Σ₂-constant model (Exercise 4).

    Parameters
    ----------
    vol1_increments : array of length n_maturities — the I1_k = ∫_{T_{k-1}}^{T_k} Σ₁²(u) du
    sigma2_const    : scalar — the constant Σ₂
    """
    T = np.asarray(maturities)
    sigma_hat = np.sqrt(np.cumsum(vol1_increments) + sigma2_const ** 2 * T)
    prices = black76_grid(forward, strikes, discounts, sigma_hat)
    return prices, sigma_hat


def price_two_time_dependent(vol1_increments: np.ndarray,
                             vol2_increments: np.ndarray,
                             forward: float,
                             strikes: np.ndarray,
                             maturities: np.ndarray,
                             discounts: np.ndarray):
    """
    Pricer for the fully time-dependent model (Exercise 5).

    Both vol1_increments and vol2_increments have length n_maturities and
    represent the integral of Σ_k^2 over each successive maturity interval.
    """
    sigma_hat = np.sqrt(np.cumsum(vol1_increments) + np.cumsum(vol2_increments))
    prices = black76_grid(forward, strikes, discounts, sigma_hat)
    return prices, sigma_hat
