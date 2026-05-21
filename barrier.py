"""
Exercise 6 — Pricing a down-and-in call option.

The barrier is L = 450, strike K = 500, maturity 6 months.
We price it three times using the three calibrated volatility models, and
also use the closed-form Bjork formula (Theorem 18.8 in *Arbitrage Theory in
Continuous Time*) as a benchmark for the constant-vol case.

A note on monitoring: the Monte Carlo simulator only checks the barrier at
the option grid maturities (which is what the original MATLAB code did). The
closed-form formula assumes *continuous* monitoring. This is the main source
of the gap between the two prices for the constant-vol model — continuous
monitoring catches more barrier crossings, hence a higher down-and-in price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from black76 import black76


@dataclass
class BarrierResults:
    mc_constant: float
    mc_one_time_dep: float
    mc_two_time_dep: float
    closed_form_constant: float


def simulate_paths(variance_increments: np.ndarray,
                   F0: float,
                   n_paths: int = 100_000,
                   seed: int = 0) -> np.ndarray:
    """
    Simulate forward-price paths under the lognormal HJM dynamics.

    Each step has its own *variance increment* dV_k = ∫_{T_{k-1}}^{T_k} Σ²(u) du,
    so the log-return over step k is N(-0.5 * dV_k, dV_k). This is the discrete
    version of the martingale-corrected lognormal dynamics.

    Parameters
    ----------
    variance_increments : (n_steps,) — total Σ² integrated over each step
    F0    : initial forward price
    n_paths : number of Monte Carlo paths
    seed  : RNG seed for reproducibility

    Returns
    -------
    paths : (n_paths, n_steps + 1) — paths including F0 at column 0
    """
    rng = np.random.default_rng(seed)
    n_steps = len(variance_increments)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = F0
    for k, dV in enumerate(variance_increments):
        drift = -0.5 * dV
        shock = rng.standard_normal(n_paths) * np.sqrt(dV)
        paths[:, k + 1] = paths[:, k] * np.exp(drift + shock)
    return paths


def _down_and_in_payoff(paths: np.ndarray,
                       barrier: float,
                       strike: float,
                       discount: float) -> float:
    """
    Standard down-and-in call payoff: only paths that touched the barrier pay.
    Barrier is monitored at every column of `paths`.
    """
    touched = (paths < barrier).any(axis=1)
    terminal = paths[:, -1]
    payoff = np.maximum(terminal - strike, 0.0) * touched
    return discount * payoff.mean()


def price_constant_vol_barrier(sigma1: float, sigma2: float,
                               maturities: np.ndarray,
                               F0: float, discount_T: float,
                               barrier: float, strike: float,
                               n_paths: int = 100_000, seed: int = 0) -> float:
    """Down-and-in price under the constant-vol model."""
    # Variance per step: (Σ₁² + Σ₂²) * dT_k
    dT = np.diff(np.concatenate([[0.0], maturities]))
    variance_increments = (sigma1 ** 2 + sigma2 ** 2) * dT
    paths = simulate_paths(variance_increments, F0, n_paths, seed)
    return _down_and_in_payoff(paths, barrier, strike, discount_T)


def price_one_time_dep_barrier(vol1_increments: np.ndarray, sigma2_const: float,
                               maturities: np.ndarray,
                               F0: float, discount_T: float,
                               barrier: float, strike: float,
                               n_paths: int = 100_000, seed: int = 0) -> float:
    """Down-and-in price under the Σ₁(t) / Σ₂ constant model."""
    dT = np.diff(np.concatenate([[0.0], maturities]))
    variance_increments = vol1_increments + (sigma2_const ** 2) * dT
    paths = simulate_paths(variance_increments, F0, n_paths, seed)
    return _down_and_in_payoff(paths, barrier, strike, discount_T)


def price_two_time_dep_barrier(vol1_increments: np.ndarray,
                               vol2_increments: np.ndarray,
                               maturities: np.ndarray,
                               F0: float, discount_T: float,
                               barrier: float, strike: float,
                               n_paths: int = 100_000, seed: int = 0) -> float:
    """Down-and-in price under the fully time-dependent model."""
    variance_increments = vol1_increments + vol2_increments
    paths = simulate_paths(variance_increments, F0, n_paths, seed)
    return _down_and_in_payoff(paths, barrier, strike, discount_T)


def closed_form_down_and_in(sigma1: float, sigma2: float,
                            F0: float, K: float, L: float, T: float,
                            discount_T: float) -> float:
    """
    Closed-form down-and-in call under constant lognormal volatility (Bjork 18.8).

    Uses the reflection-principle pricing identity for barrier options on a
    GBM. Only valid when L < K (which is our case: L = 450 < K = 500).
    Assumes continuous monitoring.

    Note on the formula
    -------------------
    The MATLAB version of this project computes the exponent as
        2 * (r - 0.5 * sigma_hat^2) / sigma_hat^2
    where sigma_hat^2 = (Σ₁² + Σ₂²) * T is the *total* variance. This is
    dimensionally off — it gives the textbook exponent times an extra 1/T
    factor. We use the correct form below: the exponent is

        2 * r_annual / sigma_annual^2 − 1

    which equals the MATLAB version only when T = 1. Numbers will therefore
    differ slightly from the MATLAB report; the MC benchmarks (which are
    dimensionless) still line up.
    """
    # Implied annualised "risk-free rate" from the discount factor at maturity
    r = -np.log(discount_T) / T

    # Annualised variance and aggregated vol
    sigma_sq_ann = sigma1 ** 2 + sigma2 ** 2
    sigma_hat = np.sqrt(sigma_sq_ann * T)

    # Bjork 18.8: down-and-in call with barrier L < K
    exponent = 2.0 * r / sigma_sq_ann - 1.0
    vanilla = black76(L ** 2 / F0, K, discount_T, sigma_hat)
    return (L / F0) ** exponent * vanilla


def price_all_models(constant_result, one_tdep_result, two_tdep_result,
                    market_data,
                    barrier: float = 450.0,
                    strike: float = 500.0,
                    maturity: float = 0.5,
                    n_paths: int = 100_000,
                    seed: int = 0) -> BarrierResults:
    """
    Run all four pricings (3 Monte Carlo + 1 closed-form) and return them.

    `market_data` provides F0, the maturity grid, and the discount curve.
    """
    F0 = market_data.forward
    maturities = market_data.maturities
    # Discount factor at the *barrier maturity*, interpolated from the grid
    discount_T = float(np.interp(maturity, maturities, market_data.discount_factors))

    mc_const = price_constant_vol_barrier(
        constant_result.sigma1, constant_result.sigma2,
        maturities, F0, discount_T, barrier, strike, n_paths, seed)

    mc_one = price_one_time_dep_barrier(
        one_tdep_result.vol1_increments, one_tdep_result.sigma2_const,
        maturities, F0, discount_T, barrier, strike, n_paths, seed + 1)

    mc_two = price_two_time_dep_barrier(
        two_tdep_result.vol1_increments, two_tdep_result.vol2_increments,
        maturities, F0, discount_T, barrier, strike, n_paths, seed + 2)

    cf = closed_form_down_and_in(
        constant_result.sigma1, constant_result.sigma2,
        F0, strike, barrier, maturity, discount_T)

    return BarrierResults(mc_constant=mc_const,
                          mc_one_time_dep=mc_one,
                          mc_two_time_dep=mc_two,
                          closed_form_constant=cf)
