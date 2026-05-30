# HJM Multifactor Calibration & Option Pricing — German Power Market

Course project for *Computational Finance* / Energy Markets at Politecnico di Milano
(M.Sc. Mathematical Engineering, A.Y. 2024/25). Originally written in MATLAB, here
ported to Python.

## What the project does

We work with **call options on the 4Q25 German power future** (EEX market, snapshot
as of 4th November 2024) and try to answer two questions:

1. **Calibration.** Can we fit an HJM-style two-factor model to the observed option
   prices? Specifically, what is the simplest volatility structure that reproduces
   the market well?
2. **Pricing.** Once calibrated, how does the model price a barrier (down-and-in)
   call option, and how do the different calibration choices compare?

### Some context — why this is a non-trivial problem

Power markets are not stock markets. A few things make them peculiar:

- **Implied volatilities are huge.** On this dataset, IV ranges from ~25% near
  the money up to 200% on deep wings. Equity options rarely go above 50%.
- **The volatility smile is very pronounced**, especially at short maturities.
  This is because power prices are notoriously spiky (cold snaps, supply
  outages) and the market prices in tail risk explicitly.
- **No spot — only futures.** Power can't be stored, so the natural
  underlying is the futures contract for a given delivery period. That's why
  we price options with **Black-76** (the futures version of Black-Scholes)
  instead of standard Black-Scholes.

### The model — HJM with two Brownian factors

We use the Heath-Jarrow-Morton framework in its simplest form: two independent
Brownian motions and no jumps. Under the risk-neutral measure the futures
price evolves as

```
F(t, τ₁, τ₂) = F(0, τ₁, τ₂) · exp( ∫₀ᵗ A(u) du
                                 + ∫₀ᵗ Σ₁ dW₁(u)
                                 + ∫₀ᵗ Σ₂ dW₂(u) )
```

with the **drift fixed by the martingale (no-arbitrage) condition** to
`A = -½(Σ₁² + Σ₂²)`. The only free parameters are Σ₁ and Σ₂.

Because both volatilities feed a Gaussian, the aggregate volatility that enters
the option-pricing formula is

```
σ̂(T) = √( Σ₁²·T + Σ₂²·T )      (constant-vol case)
σ̂(T) = √( ∫₀ᵀ Σ₁²(u) du + ∫₀ᵀ Σ₂²(u) du )   (time-dependent case)
```

and Black-76 then gives the call price. In the time-dependent versions we
parametrise the **integral increments** between successive maturities (not the
volatility curve itself), which both makes the optimization easier and
guarantees that the cumulative variance is monotonically increasing.

### Calibration — three increasingly flexible models

| | Σ₁ | Σ₂ | # parameters |
|---|---|---|---|
| **Exercise 3** | constant | constant | 2 |
| **Exercise 4** | time-dependent (8 increments) | constant | 9 |
| **Exercise 5** | time-dependent (8 increments) | time-dependent (8 increments) | 16 |

All three minimize `MSE = mean((P_model − P_market)²)` over the option grid
with `Σ ≥ 0` as the only constraint, using sequential quadratic programming.

We deliberately **drop the wings of the strike grid** (we keep K ∈ [410, 540])
because the deep ITM/OTM points have noisy implied vols that throw off the
fit. This brings the MSE from ~1000 down to ~232.

The big finding from the calibration is that **the second volatility Σ₂
barely matters**: its increments come out an order of magnitude smaller than
Σ₁'s. Going from one to two time-dependent factors only moves the MSE from
198.6 to ~198. The model is essentially a one-factor model in disguise on
this data.

### Pricing — down-and-in call (Exercise 6)

With the three calibrated models we price a barrier option:

- **Underlying**: 4Q25 German power future
- **Barrier (down)**: L = 450
- **Strike**: K = 500
- **Maturity**: 6 months
- **Payoff**: `max(F_T − K, 0) · 1{min_t F_t ≤ L}`

We use Monte Carlo (100k paths) for all three models, and we also have a
**closed-form benchmark** for the constant-volatility case (Bjork's Theorem
18.8 on barrier options under Black-Scholes). The closed form assumes
continuous barrier monitoring, while the MC monitors only at maturity points
— this is the main source of the gap between the two prices.

Results on this dataset:

|                       | Constant vol | 1 time-dep | 2 time-dep |
|-----------------------|--------------|------------|------------|
| Monte Carlo price     | ≈ 7.94       | ≈ 1.43     | ≈ 1.42     |
| Closed-form benchmark | ≈ 15.49      | —          | —          |

The time-dependent models give a much lower price because their per-step
variance is small enough that paths rarely fall through the barrier.

> **Note on the closed form.** The original MATLAB version of this project
> had a small dimensional inconsistency in the Bjork formula (the exponent
> was multiplied by an extra T factor). The Python port uses the textbook
> form `(L/F₀)^(2r/σ² − 1)`. As a result the closed-form number here is
> ~15.5 against the report's 14.67; the Monte Carlo benchmarks are
> unaffected and match the report.

## Project layout

| File | Contents |
|---|---|
| `main.py` | Driver — loads data, runs all four exercises in sequence |
| `data_loading.py` | Reads the Excel file, converts dates, builds the market grid |
| `black76.py` | Black-76 pricing primitives (single option + grid) |
| `pricing_models.py` | The three HJM pricers: constant, 1-time-dep, 2-time-dep |
| `calibration.py` | Exercises 3, 4, 5 — calibrate each model to market |
| `barrier.py` | Exercise 6 — Monte Carlo + closed-form for the down-and-in |
| `plotting.py` | The 3D surface plots (market vs calibrated, prices and vols) |
| `requirements.txt` | numpy, scipy, pandas, matplotlib, openpyxl |

## Inputs

A single Excel file is expected: `data/DATA_DEEEX.xlsx`, with three sheets:

- **`Prices`** — futures prices. Columns: `Expiry Date`, `Mth`, `Last`. The
  last row is the 4Q25 contract (the one we calibrate on).
- **`OptionsOnQ42025`** — implied volatility matrix. First row is strikes,
  first column is maturities (in years), the rest is the IV matrix.
- **`discounts`** — discount curve. Row 1: dates, row 2: discount factors.
  These are used to compute discount factors at each option maturity by
  linear interpolation.

The valuation date is **2024-11-05** (4th of November 2024) — change this in
`data_loading.py` if you point the script at a different snapshot.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

Console output:
- Calibrated Σ₁, Σ₂ and MSE for each of the three calibration models
- Table of σ̂ increments per maturity (for the time-dependent models)
- Down-and-in call prices from all three models + the closed-form benchmark

Plots: market option price surface, market implied volatility surface, and
market-vs-model comparisons for each calibration.

## Notes on the MATLAB → Python port

- MATLAB's `fmincon` becomes `scipy.optimize.minimize(method="SLSQP")`. Same
  algorithm family, may take slightly different search directions at the
  boundary `Σ = 0`, so calibrated parameters can differ in the 4th decimal.
- `normcdf` becomes `scipy.stats.norm.cdf`.
- `blkprice` (MATLAB Financial Toolbox) is replaced by a vectorised
  implementation in `black76.py`. We follow the **same convention as the
  original codebase**: the `sigma` argument is always the **aggregated**
  volatility `σ̂ = √(∫ Σ² du)`, not the annualised σ. The conversion
  `sigma_market * sqrt(T) → sigma_hat` happens once when reading the data,
  then everything downstream uses sigma_hat.
- All Monte Carlo uses `numpy.random.default_rng(seed=0)` for reproducibility.
  The original code used MATLAB's default RNG with `rng("default")`, so MC
  prices will not match to the last decimal.

## Authors

Giacomo Papa, Alessandro Torazzi, Andrea Meschieri — A.Y. 2024/25.
