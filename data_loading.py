"""
Read the Excel data file and prepare everything the calibration needs.

The file has three sheets:
    - Prices             : futures prices (we keep the 4Q25 one)
    - OptionsOnQ42025    : implied vol matrix on the 4Q25 future
    - discounts          : discount factor curve

Output is a MarketData dataclass holding everything in numpy form.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Pricing date — the snapshot is from the 4th of November 2024.
VALUATION_DATE = pd.Timestamp("2024-11-05")
DAYS_IN_YEAR = 365


@dataclass
class MarketData:
    """Container for everything the calibration touches."""
    forward: float                      # F(0, T) for the 4Q25 contract
    strikes: np.ndarray                 # (n_strikes,)
    maturities: np.ndarray              # (n_maturities,) in fractional years
    discount_factors: np.ndarray        # (n_maturities,)
    implied_vols: np.ndarray            # (n_maturities, n_strikes) — annualised
    market_prices: np.ndarray           # (n_maturities, n_strikes) — from Black-76


def _interpolate_discount_factors(curve_dates: np.ndarray,
                                  curve_dfs: np.ndarray,
                                  target_T: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of the discount curve at the option maturities.

    `curve_dates` and `target_T` are both in fractional years from valuation date.
    """
    # np.interp does linear interpolation; for extrapolation beyond the curve
    # ends we extend flat (constant). The original MATLAB code used
    # `interp1(..., 'extrap')` which is linear extrapolation — we use that here
    # since the option maturities all lie inside the curve range anyway.
    return np.interp(target_T, curve_dates, curve_dfs)


def load_market_data(path: str | Path) -> MarketData:
    """
    Load the DEEEX Excel file and return everything as numpy arrays.

    Parameters
    ----------
    path : str or Path
        Path to the .xlsx file. Three sheets are expected:
        'Prices', 'OptionsOnQ42025', 'discounts' (any case variants of
        OptionsOnQ42025 are also accepted).
    """
    path = Path(path)

    # ---- 1. Futures prices ----
    futures = pd.read_excel(path, sheet_name="Prices")
    # The 4Q25 future is the last row of the Prices sheet (per the MATLAB code)
    forward = float(futures.iloc[-1, -1])

    # ---- 2. Option IV matrix on 4Q25 ----
    # Try both casings of the sheet name (the file uses 'OptionsOnQ42025' but
    # the MATLAB code referred to it as 'OptionsONQ42025').
    xl = pd.ExcelFile(path)
    options_sheet = next(s for s in xl.sheet_names if s.lower() == "optionsonq42025")
    options = pd.read_excel(path, sheet_name=options_sheet, header=None)

    strikes = options.iloc[0, 1:].values.astype(float)        # row 0, cols 1..
    maturities = options.iloc[1:, 0].values.astype(float)     # col 0, rows 1..
    implied_vols = options.iloc[1:, 1:].values.astype(float)  # the IV grid

    # ---- 3. Discount curve ----
    discounts_raw = pd.read_excel(path, sheet_name="discounts", header=None)
    curve_dates_dt = pd.to_datetime(discounts_raw.iloc[0, :].values)
    curve_dfs = discounts_raw.iloc[1, :].values.astype(float)

    # Convert dates to fractional years from the valuation date
    curve_T = np.array([(d - VALUATION_DATE).days / DAYS_IN_YEAR
                        for d in curve_dates_dt])

    # Discount factors at the option maturities (linear interpolation)
    discount_factors = _interpolate_discount_factors(curve_T, curve_dfs, maturities)

    # ---- 4. Build the corresponding market price grid ----
    # Import here to avoid a circular dependency at module level
    from black76 import market_prices_from_iv
    market_prices = market_prices_from_iv(
        forward, strikes, maturities, discount_factors, implied_vols)

    return MarketData(
        forward=forward,
        strikes=strikes,
        maturities=maturities,
        discount_factors=discount_factors,
        implied_vols=implied_vols,
        market_prices=market_prices,
    )


def filter_strikes(data: MarketData, lower: float, upper: float) -> MarketData:
    """
    Drop strikes outside [lower, upper] from the market grid.

    The wings of the IV surface are noisy and bias the calibration, so we
    restrict the fit to the central strikes ([410, 540] in the original work).
    """
    mask = (data.strikes >= lower) & (data.strikes <= upper)
    return MarketData(
        forward=data.forward,
        strikes=data.strikes[mask],
        maturities=data.maturities,
        discount_factors=data.discount_factors,
        implied_vols=data.implied_vols[:, mask],
        market_prices=data.market_prices[:, mask],
    )
