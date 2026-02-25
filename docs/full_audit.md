# OMEGA NOVA — Full Repository Audit

**Scope:** Every `.py` model/engine file (25+ files, ~600K lines).  
**Methodology:** Mathematical correctness, optimizer suitability, data pipeline integrity, cold-start safety, and architectural coherence.

---

## Summary Verdict

| Area | Health | Critical Issues |
|------|--------|----------------|
| Core Pricing (NIRV + Heston COS) | 🟢 Strong | VIX-Term mismatch for weeklies |
| Volatility Surface (SVI/eSSVI/PINN) | 🟢 Strong | None blocking |
| ML Correction (GBM/LightGBM) | 🟡 OK | Cold-start defaults need tuning |
| Neural J-SDE | 🟡 OK | Untrained path signature dims may add noise |
| Deep Hedging | 🔴 Broken | **Still using Nelder-Mead on 481 params** |
| KAN Corrector | 🟢 Fixed | SPSA+Adam already implemented |
| Hawkes Jump Process | 🟢 Strong | Minor: Poisson fallback threshold too low |
| Ensemble Pricer | 🟡 OK | Equal cold-start weights are dangerous |
| Hurst Estimation | 🟡 Noisy | Daily R/S on 30 samples = noise |
| Data Pipeline (Upstox) | 🔴 Critical | `spot = strike` fallback corrupts training |
| Behavioral Agents (NEW) | 🟡 Scaffold | Hardcoded multipliers, not calibrated |
| Structural Frictions (NEW) | 🟡 Scaffold | STT rate needs dynamic config |
| Path Signatures (NEW) | 🟡 Scaffold | Fallback manual sig is approximate |
| MOT Bounds (NEW) | 🟢 OK | LP solver works for vanilla payoffs |
| IV Solver (Jaeckel) | 🟢 Excellent | Machine-precision, no issues |
| Backtester | 🟢 Strong | Proper look-ahead prevention |

---

## 🔴 Critical Issues (Fix Immediately)

### 1. Deep Hedger — Nelder-Mead on 481 Parameters
**File:** [deep_hedging.py](file:///Users/caaniketh/Optionpricingmodel%20Ai/OPM-with-TVR-NIRV-OMEGA-Models-/deep_hedging.py#L311-L316)

The `train()` method **still** calls `scipy.optimize.minimize` with `method='Nelder-Mead'` and `maxiter=100`. The `HedgingNetwork` has `n_params = 481` (13→20 hidden + 20→1 output). Nelder-Mead requires O(N) function evaluations per simplex step and is mathematically incapable of navigating a 481-dimensional loss landscape in 100 iterations. The weights do not meaningfully update.

**Fix:** Port to Adam+SPSA (exactly as done successfully in `kan_corrector.py` and `neural_jsde.py`). Both of those files already have working implementations you can copy.

### 2. Historical Data — `spot = strike` Fallback
**File:** [historical_learning.py](file:///Users/caaniketh/Optionpricingmodel%20Ai/OPM-with-TVR-NIRV-OMEGA-Models-/historical_learning.py) (~lines 504, 550)

When Upstox historical data lacks a synchronized spot price, the pipeline defaults to `spot_proxy = strike`. This makes every option appear ATM regardless of true moneyness, producing garbage IVs (>800%) that corrupt all downstream ML training.

**Fix:** Drop the row entirely if spot is unavailable. Never fabricate moneyness.

---

## 🟡 Important Issues (Fix This Week)

### 3. VIX-Term Mismatch for Weekly Options
**File:** [nirv_model.py](file:///Users/caaniketh/Optionpricingmodel%20Ai/OPM-with-TVR-NIRV-OMEGA-Models-/nirv_model.py#L608-L648) (`_get_parametric_iv`)

The ATM IV anchor is always `india_vix / 100.0` (a 30-day measure). For 0-3 DTE weekly options, the true ATM IV can be 40% while VIX reads 15%. This systematically underprices near-expiry options.

**Fix:** Interpolate: `atm_iv(T) = blend(realized_vol_3d, india_vix, T)`, where the blend weight shifts toward short-term realized vol as T→0.

### 4. Ensemble Cold-Start — Equal Weights
**File:** [ensemble_pricer.py](file:///Users/caaniketh/Optionpricingmodel%20Ai/OPM-with-TVR-NIRV-OMEGA-Models-/ensemble_pricer.py#L81-L91) (`register_model`)

All models start at `initial_weight=1.0`. An uncalibrated Neural-JSDE (outputting random MC noise) gets the same vote as the well-tested Heston COS pricer.

**Fix:** Set `initial_weight = 1.0 / max(calibration_rmse, 0.01)` at registration time. Models that haven't been calibrated should start near zero.

### 5. Hurst Exponent — Noise from Daily Data
**File:** [unified_pipeline.py](file:///Users/caaniketh/Optionpricingmodel%20Ai/OPM-with-TVR-NIRV-OMEGA-Models-/unified_pipeline.py#L63-L112) (`_estimate_hurst`)

R/S analysis on 30-60 daily returns has enormous standard error. The estimate randomly oscillates between 0.05 and 0.60, injecting daily noise into the rough-vol wing corrections.

**Fix:** Either (a) hardcode H=0.10 (literature consensus for equity indices) until intraday tick data is available, or (b) use a 252-day rolling window minimum.

### 6. Neural JSDE — Uninitialized Signature Dimensions
**File:** [neural_jsde.py](file:///Users/caaniketh/Optionpricingmodel%20Ai/OPM-with-TVR-NIRV-OMEGA-Models-/neural_jsde.py#L139-L155)

The feature space was expanded from 6→26 dims to accept path signatures, but the RBF centers (35 centers × 26 features = 910 params for centers alone) are initialized randomly. When no signatures are provided, 20 dimensions are zeros, making the RBF distance computation dominated by noise.

**Fix:** Initialize the last 20 center dimensions to zero (matching the default zero-signature input) so untrained mode doesn't degrade quality.

### 7. Behavioral Agents — Hardcoded Multipliers
**File:** [behavioral_agents.py](file:///Users/caaniketh/Optionpricingmodel%20Ai/OPM-with-TVR-NIRV-OMEGA-Models-/behavioral_agents.py)

The lottery wing inflation (2% for calls, 1.5% for puts) and institutional skew premium (3%) are static constants. In reality, these shift dramatically based on market regime, expiry proximity, and the actual retail/institutional volume ratio on that day.

**Fix:** Calibrate these from the observed option chain: compare the model-derived IV surface (without behavioral adjustment) to the actual market IV surface, and attribute the residual to behavioral demand.

---

## 🟢 Strong Components (No Action Needed)

### 8. IV Solver (Jaeckel "Let's Be Rational")
Machine-precision Householder-4 solver. Excellent implementation — handles all edge cases (deep ITM/OTM, near-expiry). No issues found.

### 9. Heston COS Pricer
Correct "little Heston trap" formulation avoids branch-cut issues. Integration bounds from cumulants are properly computed. Vectorized `price_chain()` is efficient. Calibration uses L-BFGS-B on only 4 parameters — well-suited.

### 10. KAN Corrector
Previously used L-BFGS-B with O(N) finite-difference gradient (impractical). Now correctly uses Adam+SPSA. B-spline boundary truncation is technically a partition-of-unity violation but has negligible impact in practice for normal market moves.

### 11. Hawkes Jump Process
Well-designed self-exciting process with proper MLE via L-BFGS-B, Ogata thinning simulation, and VIX-modulated intensity. The Poisson fallback at <5 jumps is reasonable. Minor: could raise threshold to 8 jumps for stability.

### 12. PINN Vol Surface
Five-constraint loss (data fidelity, butterfly, calendar, smoothness, Dupire, martingale) is mathematically correct. Roger-Lee wing bounds are properly enforced. RBF network with L-BFGS-B is appropriate for 25-center, 2D input.

### 13. Model-Free Variance / Synthetic VIX
Correctly implements the CBOE VIX methodology. Tail correction, spread filtering, and cubic spline interpolation are all sound. The `estimate_forward_from_chain` properly uses put-call parity.

### 14. Backtester
Proper look-ahead prevention (executes at NEXT day open). Realistic NSE transaction costs (STT, brokerage, exchange fees, stamp duty). Regime-filtered entry logic is sound.

---

## Implementation Priority (Ordered Roadmap)

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | Fix Deep Hedger → Adam+SPSA | 2 hrs | High — currently non-functional |
| P0 | Fix `spot=strike` in historical_learning | 30 min | High — corrupts all ML training |
| P1 | ATM IV term structure interpolation | 3 hrs | High — fixes weekly option pricing |
| P1 | Ensemble weight initialization by RMSE | 1 hr | Medium — prevents noisy ensemble |
| P1 | Hurst exponent: hardcode or 252d window | 30 min | Medium — stabilizes rough-vol wings |
| P2 | Neural JSDE center initialization | 30 min | Medium — prevents degradation |
| P2 | Calibrate behavioral agent multipliers | 4 hrs | Medium — makes novel arch useful |
| P3 | Path signature integration tests | 2 hrs | Low — validates new module |
| P3 | STT rate as dynamic config parameter | 30 min | Low — future-proofs regulatory changes |
