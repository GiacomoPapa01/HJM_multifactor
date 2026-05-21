"""
HJM Multifactor Calibration & Option Pricing — German Power Market.

Driver script that ties everything together:
    - load and clean the market data
    - run the three calibration exercises
    - price the down-and-in barrier option
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from barrier import price_all_models
from calibration import (
    calibrate_constant_vol,
    calibrate_one_time_dependent,
    calibrate_two_time_dependent,
)
from data_loading import filter_strikes, load_market_data
from plotting import plot_comparison, plot_market_surface
from pricing_models import (
    price_constant_vol,
    price_one_time_dependent,
    price_two_time_dependent,
)


DATA_FILE = "data/DATA_DEEEX.xlsx"

# Strike filter — wings are noisy, so we restrict to the central portion
K_LOWER, K_UPPER = 410, 540

# Barrier option spec
BARRIER = 450.0
BARRIER_STRIKE = 500.0
BARRIER_MATURITY = 0.5
N_MC_PATHS = 100_000


def _divider(title: str):
    bar = "=" * max(60, len(title) + 6)
    print(f"\n{bar}\n  {title}\n{bar}")


def main(make_plots: bool = True):
    # -------------------------------------------------------------------
    # 1. Load and inspect the market data
    # -------------------------------------------------------------------
    _divider("Loading market data")
    raw = load_market_data(DATA_FILE)
    print(f"Forward price F(0, 4Q25)   : {raw.forward:.2f}")
    print(f"# strikes (full grid)      : {len(raw.strikes)}")
    print(f"# maturities               : {len(raw.maturities)}")
    print(f"Maturity range (years)     : {raw.maturities.min():.2f} – {raw.maturities.max():.2f}")
    print(f"IV range                   : {raw.implied_vols.min():.3f} – {raw.implied_vols.max():.3f}")

    if make_plots:
        plot_market_surface(raw.strikes, raw.maturities, raw.market_prices,
                            "Market Option Prices (4Q25)", "Option price")
        plot_market_surface(raw.strikes, raw.maturities, raw.implied_vols,
                            "Market Implied Volatilities (4Q25)", "IV",
                            cmap="turbo", view=(20, -20))

    # Trim the wings before calibration
    data = filter_strikes(raw, K_LOWER, K_UPPER)
    max_vol = float(data.implied_vols.max())   # used to scale the random init
    print(f"\nAfter strike filter [{K_LOWER}, {K_UPPER}]: {len(data.strikes)} strikes kept.")

    # -------------------------------------------------------------------
    # 2. Exercise 3 — constant volatility calibration
    # -------------------------------------------------------------------
    _divider("Exercise 3 — Constant volatility calibration")
    const = calibrate_constant_vol(
        data.market_prices, data.forward, data.strikes,
        data.maturities, data.discount_factors, max_vol)
    print(f"  Sigma1 = {const.sigma1:.4f}")
    print(f"  Sigma2 = {const.sigma2:.4f}")
    print(f"  MSE    = {const.mse:.4f}")

    if make_plots:
        const_prices, _ = price_constant_vol(
            const.sigma1, const.sigma2, data.forward, data.strikes,
            data.maturities, data.discount_factors)
        plot_comparison(data.strikes, data.maturities,
                        data.market_prices, const_prices,
                        "Constant Vol: Market vs Model Prices", "Price")

    # -------------------------------------------------------------------
    # 3. Exercise 4 — Σ₁(t) time-dependent, Σ₂ constant
    # -------------------------------------------------------------------
    _divider("Exercise 4 — One time-dependent volatility")
    one_tdep = calibrate_one_time_dependent(
        data.market_prices, data.forward, data.strikes,
        data.maturities, data.discount_factors, max_vol)
    print(f"  Constant Sigma2 = {one_tdep.sigma2_const:.4f}")
    print(f"  MSE             = {one_tdep.mse:.4f}")
    print("\n  Sigma1 increments per maturity interval:")
    df = pd.DataFrame({
        "Maturity":     data.maturities,
        "Sigma1 incr.": one_tdep.vol1_increments,
        "Sigma2 incr.": one_tdep.sigma2_const ** 2 * np.diff(np.concatenate([[0], data.maturities])),
        "Sigma hat":    one_tdep.sigma_hat,
    })
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if make_plots:
        one_prices, _ = price_one_time_dependent(
            one_tdep.vol1_increments, one_tdep.sigma2_const,
            data.forward, data.strikes, data.maturities, data.discount_factors)
        plot_comparison(data.strikes, data.maturities,
                        data.market_prices, one_prices,
                        "1 Time-Dep Vol: Market vs Model Prices", "Price")

    # -------------------------------------------------------------------
    # 4. Exercise 5 — Both time-dependent
    # -------------------------------------------------------------------
    _divider("Exercise 5 — Two time-dependent volatilities")
    two_tdep = calibrate_two_time_dependent(
        data.market_prices, data.forward, data.strikes,
        data.maturities, data.discount_factors, max_vol)
    print(f"  MSE = {two_tdep.mse:.4f}")
    print("\n  Increments per maturity interval:")
    df = pd.DataFrame({
        "Maturity":     data.maturities,
        "Sigma1 incr.": two_tdep.vol1_increments,
        "Sigma2 incr.": two_tdep.vol2_increments,
        "Sigma hat":    two_tdep.sigma_hat,
    })
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if make_plots:
        two_prices, _ = price_two_time_dependent(
            two_tdep.vol1_increments, two_tdep.vol2_increments,
            data.forward, data.strikes, data.maturities, data.discount_factors)
        plot_comparison(data.strikes, data.maturities,
                        data.market_prices, two_prices,
                        "2 Time-Dep Vols: Market vs Model Prices", "Price")

    # -------------------------------------------------------------------
    # 5. Exercise 6 — Down-and-in call pricing
    # -------------------------------------------------------------------
    _divider("Exercise 6 — Down-and-in call")
    print(f"  Barrier  L = {BARRIER}")
    print(f"  Strike   K = {BARRIER_STRIKE}")
    print(f"  Maturity T = {BARRIER_MATURITY} years")
    print(f"  MC paths   = {N_MC_PATHS}")

    results = price_all_models(
        const, one_tdep, two_tdep, data,
        barrier=BARRIER, strike=BARRIER_STRIKE, maturity=BARRIER_MATURITY,
        n_paths=N_MC_PATHS)

    print()
    print(f"  MC, constant vol           : {results.mc_constant:.4f}")
    print(f"  MC, 1 time-dependent vol   : {results.mc_one_time_dep:.4f}")
    print(f"  MC, 2 time-dependent vols  : {results.mc_two_time_dep:.4f}")
    print(f"  Closed-form (constant vol) : {results.closed_form_constant:.4f}")
    print("\n  Note: the MC monitors the barrier only at the option maturity")
    print("        grid, while the closed-form assumes continuous monitoring —")
    print("        hence the gap on the constant-vol case.")

    return {
        "data": data,
        "constant": const,
        "one_time_dep": one_tdep,
        "two_time_dep": two_tdep,
        "barrier": results,
    }


if __name__ == "__main__":
    main()
