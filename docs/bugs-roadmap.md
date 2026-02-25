# OMEGA NOVA: Bugs, Inaccuracies & Upgrade Roadmap

This document outlines structural bugs, mathematical inaccuracies, data ingestion issues, and an actionable roadmap for upgrading the OMEGA NOVA pipeline. These findings come from a deep expert scan of the entire v7.0 codebase.

---

## 1. Data Ingestion & Transformation 🚨

**1.1. Missing Spot Price Defaulting to Strike (Critical)**
*   **File:** `historical_learning.py` (lines 504, 550)
*   **Issue:** When historical option snapshots from Upstox lack a synchronized underlying spot price, the data pipeline forces `spot = strike`.
*   **Impact:** This completely destroys the moneyness of the option. A deep ITM option trading at ₹1,000 will be treated as an ATM option trading at ₹1,000, causing the implied volatility calculator to output absurd IVs (e.g., >800%). This corrupts the training data fed to the KAN, Neural-JSDE, and ML models.
*   **Fix:** **Never** default `spot = strike`. If spot is missing, either:
    1. Interpolate spot from the historical futures price using put-call parity.
    2. Drop the snapshot entirely.
    3. Forward-fill the last known spot price if the time gap is extremely small.

**1.2. Intraday Time-to-Expiry (T) Clipping**
*   **File:** `omega_model.py` (FeatureFactory)
*   **Issue:** `inv_time = min(1.0 / T, 365.0)`. This caps the inverse time at 1 day.
*   **Impact:** For 0DTE (zero days to expiry) options traded in the last hours of the day (e.g., $T = 0.001$), the network loses granularity on the extreme theta decay and gamma explosion.
*   **Fix:** Increase the cap to represent at least 15-minute intervals (e.g., `inv_time` cap = ~25000), or use a smooth non-linear scaling like $\log(1/T + 1)$.

---

## 2. Mathematical Inaccuracies 🧮

**2.1. Hardcoded 30-Day VIX as ATM Volatility Anchor (Major)**
*   **File:** `nirv_model.py` (`_get_parametric_iv`)
*   **Issue:** The SVI surface directly anchors the ATM implied volatility to `india_vix / 100.0`.
*   **Impact:** India VIX measures 30-day expected volatility. Weekly Nifty options expire in 0-7 days. During earning seasons or elections, the 1DTE IV can be 40% while the 30-day VIX is 15%. Forcing the weekly option's ATM IV to equal the VIX will cause massive underpricing mathematically.
*   **Fix:** Implement a proper Term Structure model for the ATM anchor. $IV_{atm}(T)$ should interpolate between a short-term volatility gauge (e.g., realised 3-day vol) and the 30-day VIX.

**2.2. Hurst Exponent (rBergomi) estimated from Daily Returns**
*   **File:** `unified_pipeline.py` & `MacroFeatureEngine`
*   **Issue:** The Hurst exponent ($H$) is calculated using Rescaled Range (R/S) analysis on 30-60 days of *daily* log returns.
*   **Impact:** Estimating $H$ on $N < 100$ samples has a massive standard error. The estimate is basically noise, randomly flipping between $<0.5$ (rough) and $>0.5$ (persistent). This injects random daily shocks into the rough volatility wings and the rBergomi pricer.
*   **Fix:** Rough volatility inherently requires high-frequency data (1-minute or 5-minute candles). Either pull intraday tick data to estimate $H$, or treat $H$ as a hyperparameter calibrated to the cross-sectional shape of the volatility smile.

---

## 3. Implementation Bugs 🐛

**3.1. Deep Hedger fails to converge due to Nelder-Mead (Critical)**
*   **File:** `deep_hedging.py` (`train` method)
*   **Issue:** The `HedgingNetwork` has 481 parameters. The training uses `scipy.optimize.minimize` with the `Nelder-Mead` simplex method, capped at 100 iterations.
*   **Impact:** Nelder-Mead is a derivative-free optimizer that scales horribly with dimensionality (practically $O(N^2)$ per step). It is mathematically impossible to navigate a 481-dimensional neural network loss surface with 100 simplex steps. The network weights effectively do not update, and the Deep Hedger defaults back to generic BSM performance.
*   **Fix:** Port the training loop to use the `Adam` optimizer + SPSA (Simultaneous Perturbation Stochastic Approximation), exactly as implemented correctly in `neural_jsde.py`.

**3.2. Ensemble Cold-Start Weighting**
*   **File:** `ensemble_pricer.py`
*   **Issue:** New models all start with an equal weight of `1.0`.
*   **Impact:** An uncalibrated Neural-JSDE (which might spit out random 10,000-path MC noise) gets the exact same voting power as the highly stable, 0.05ms Heston COS model until the ensemble sees 5+ live market trades.
*   **Fix:** The initial weight of a model in the ensemble must be strictly proportional to its *calibration goodness-of-fit* (e.g., $1/\text{RMSE}$). Uncalibrated or poorly calibrated models must start with near-zero weight.

**3.3. KAN B-Spline Boundary Truncation**
*   **File:** `kan_corrector.py` (Cox-de Boor recursion)
*   **Issue:** The loop manually truncates the basis functions to `n_basis` to prevent out-of-bounds array access.
*   **Impact:** This violates the partition-of-unity property of B-splines near the boundary knots. While it works for standard market moves, extreme inputs (crash regimes) hitting the boundaries will yield unpredictable step-jumps in the corrector logic.
*   **Fix:** Embed the correct padded knot vector $[t_0, t_0, ... t_{n}, t_n, ...]$ at the edges to handle boundary clamping naturally without manual array slicing.

---

## 4. Prioritized Execution Strategy (Roadmap) 📈

### Phase 1: Immediate Bug Fixes (Days 1-3)
1. **Fix Upstox Spot Proxy:** Open `historical_learning.py` and replace `spot_proxy = strike` with a mechanism that drops the row if `spot`, `close`, and `futures_price` are all unavailable.
2. **Upgrade Deep Hedger Optimizer:** Rewrite `deep_hedging.py.train()` to use `Adam + SPSA` with Common Random Numbers.
3. **Fix Ensemble Cold Start:** Modify `unified_pipeline.py.train_all()` to pass the calibration `RMSE` of each component into `ensemble.register_model()`.

### Phase 2: Mathematical Upgrades (Days 4-7)
1. **Term Structure of ATM Volatility:** Disconnect the SVI anchor from the raw 30-day VIX in `nirv_model.py`. Create an `interpolate_atm_vol(T, realized_vol_3d, vix_30d)` function.
2. **Hurst Parameter Shift:** Remove the daily R/S calculation. Hardcode $H=0.10$ (standard rough vol literature setting) until an intraday tick database is built.

### Phase 3: SOTA Pipeline Enhancements (Days 8-14)
1. **Continuous Online Learning:** The ML corrector currently batches updates every 15 trades. Upgrade the ensemble weights to use **Online Mirror Descent** (OMD) for tick-by-tick regret minimization.
2. **Conformal Interval Scaling:** Ensure that the conformal width scales exactly with $\sqrt{T}$ rather than via the linear heuristic multipliers currently used in `MLPricingCorrector`.
