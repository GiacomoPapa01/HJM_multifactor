"""
Calibration of the three HJM volatility models to market option prices.

All three exercises share the same skeleton:
    1. Define an MSE objective in terms of the model parameters
    2. Minimize via SLSQP with non-negativity bounds
    3. Return the calibrated parameters

The original MATLAB code uses random starting points; we keep that behavior
but expose a seed so results are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from black76 import mse
from pricing_models import (
    price_constant_vol,
    price_one_time_dependent,
    price_two_time_dependent,
)


@dataclass
class ConstantVolResult:
    sigma1: float
    sigma2: float
    mse: float


@dataclass
class OneTimeDepResult:
    vol1_increments: np.ndarray   # (n_maturities,)
    sigma2_const: float
    mse: float
    sigma_hat: np.ndarray         # (n_maturities,) calibrated aggregated vol


@dataclass
class TwoTimeDepResult:
    vol1_increments: np.ndarray
    vol2_increments: np.ndarray
    mse: float
    sigma_hat: np.ndarray


# -----------------------------------------------------------------------------
# Exercise 3 — constant Σ₁, constant Σ₂
# -----------------------------------------------------------------------------

def calibrate_constant_vol(market_prices: np.ndarray,
                           forward: float,
                           strikes: np.ndarray,
                           maturities: np.ndarray,
                           discount_factors: np.ndarray,
                           max_vol: float = 2.0,
                           seed: int = 0,
                           n_restarts: int = 10) -> ConstantVolResult:
    """
    Two-parameter calibration (Σ₁, Σ₂) — Exercise 3.

    Uses multi-start because SLSQP from a random init occasionally lands in
    the trivial near-zero local minimum (where σ̂ ≈ 0 makes Black-76 ≈ 0 and
    the gradient becomes uninformative). We try a handful of starting points
    and keep the best solution.
    """
    rng = np.random.default_rng(seed)

    def objective(params):
        s1, s2 = params
        prices, _ = price_constant_vol(s1, s2, forward, strikes,
                                       maturities, discount_factors)
        return mse(market_prices, prices)

    best = None
    for _ in range(n_restarts):
        x0 = max_vol * rng.random(2)
        try:
            res = minimize(objective, x0, method="SLSQP",
                           bounds=[(1e-6, None), (1e-6, None)],
                           options={"ftol": 1e-10, "maxiter": 1000})
        except (FloatingPointError, ValueError):
            continue
        if best is None or res.fun < best.fun:
            best = res

    return ConstantVolResult(sigma1=best.x[0], sigma2=best.x[1], mse=best.fun)


# -----------------------------------------------------------------------------
# Exercise 4 — Σ₁(t) time-dependent (8 increments), Σ₂ constant
# -----------------------------------------------------------------------------

def calibrate_one_time_dependent(market_prices: np.ndarray,
                                 forward: float,
                                 strikes: np.ndarray,
                                 maturities: np.ndarray,
                                 discount_factors: np.ndarray,
                                 max_vol: float = 2.0,
                                 seed: int = 0,
                                 n_restarts: int = 5) -> OneTimeDepResult:
    """
    Calibration of Σ₁ as 8 integral increments plus a single constant Σ₂.

    Decision vector layout:
        [I1_1, I1_2, ..., I1_n, sigma2_const]

    Same multi-start trick as Exercise 3 to avoid the degenerate σ̂ ≈ 0 minimum.
    """
    rng = np.random.default_rng(seed)
    n = len(maturities)

    def objective(params):
        vol1_inc = params[:-1]
        s2 = params[-1]
        prices, _ = price_one_time_dependent(vol1_inc, s2, forward, strikes,
                                             maturities, discount_factors)
        return mse(market_prices, prices)

    best = None
    for _ in range(n_restarts):
        x0 = max_vol * rng.random(n + 1)
        try:
            res = minimize(objective, x0, method="SLSQP",
                           bounds=[(1e-8, None)] * (n + 1),
                           options={"ftol": 1e-10, "maxiter": 2000})
        except (FloatingPointError, ValueError):
            continue
        if best is None or res.fun < best.fun:
            best = res

    vol1_inc = best.x[:-1]
    s2 = best.x[-1]
    _, sigma_hat = price_one_time_dependent(vol1_inc, s2, forward, strikes,
                                            maturities, discount_factors)
    return OneTimeDepResult(vol1_increments=vol1_inc,
                            sigma2_const=s2,
                            mse=best.fun,
                            sigma_hat=sigma_hat)


# -----------------------------------------------------------------------------
# Exercise 5 — Both Σ₁(t) and Σ₂(t) time-dependent
# -----------------------------------------------------------------------------

def calibrate_two_time_dependent(market_prices: np.ndarray,
                                 forward: float,
                                 strikes: np.ndarray,
                                 maturities: np.ndarray,
                                 discount_factors: np.ndarray,
                                 max_vol: float = 2.0,
                                 seed: int = 0,
                                 n_restarts: int = 5) -> TwoTimeDepResult:
    """
    Calibration of both Σ₁ and Σ₂ as 8 integral increments each.

    Decision vector is laid out as the concatenation of the two increment
    arrays — 2 * n_maturities parameters in total.
    """
    rng = np.random.default_rng(seed)
    n = len(maturities)

    def objective(params):
        v1 = params[:n]
        v2 = params[n:]
        prices, _ = price_two_time_dependent(v1, v2, forward, strikes,
                                             maturities, discount_factors)
        return mse(market_prices, prices)

    best = None
    for _ in range(n_restarts):
        x0 = max_vol * rng.random(2 * n)
        try:
            res = minimize(objective, x0, method="SLSQP",
                           bounds=[(1e-8, None)] * (2 * n),
                           options={"ftol": 1e-10, "maxiter": 3000})
        except (FloatingPointError, ValueError):
            continue
        if best is None or res.fun < best.fun:
            best = res

    v1 = best.x[:n]
    v2 = best.x[n:]
    _, sigma_hat = price_two_time_dependent(v1, v2, forward, strikes,
                                            maturities, discount_factors)
    return TwoTimeDepResult(vol1_increments=v1,
                            vol2_increments=v2,
                            mse=best.fun,
                            sigma_hat=sigma_hat)
