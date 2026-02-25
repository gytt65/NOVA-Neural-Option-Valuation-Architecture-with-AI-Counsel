# Path to Empirical Validation (Rank 8 → 9)

To move NOVA and OMEGA from Rank 8 to Rank 9, we need **empirical validation**. The code is mathematically sound, but quantitative models are graded on their out-of-sample (OOS) P&L, not their theoretical elegance.

Fortunately, the repository already contains the exact tools needed to prove this. You don't need to write new code; you just need to run the existing pipelines systematically.

Here is the exact, step-by-step process you should follow over the next 2-3 weeks.

---

## 1. Validate OMEGA (Tab 8)

OMEGA is an ML residual corrector. An ML model is only as good as its training data. We need to move it from "untrained/cold-started" to "trained on 6 months of true market residuals."

### Step 1.1: Bulk Data Accumulation
Use the **Historical Learning (Tab 10)** to pull real-world option data.
1. Open the app and go to Tab 10 (Historical Learning).
2. Set the date range: **Jan 1, 2024 to Jun 30, 2024** (6 months of training data).
3. Set Interval to `day` (for end-of-day pricing) or `30minute` if you want intraday precision.
4. Click **"Pull Data & Update ML Pricing Corrector"**.
5. *What happens:* The system will download historical Nifty options, run the mathematical NIRV pricer on every single row, compute the pricing error (the "residual"), and train the XGBoost/KAN model to predict those errors.

### Step 1.2: Out-of-Sample Verification
Once the model is trained on Jan-Jun, we test it on Jul-Sep.
1. Change the date range in Tab 10 to **Jul 1, 2024 to Sep 30, 2024**.
2. Run the pull again.
3. *Crucial Metric:* Look at the output logs for the **Walk-Forward Validation metrics** (specifically the `mape_test_pct` and `direction_hit_rate_test`).
4. If the Out-of-Sample Mean Absolute Error (MAE) is significantly lower than the baseline (i.e., the ML model successfully predicted pricing errors on data it had never seen), **OMEGA is empirically validated.**

---

## 2. Validate NOVA (Tab 9)

NOVA is an end-to-end pipeline. We need to prove that its ensemble pricing actually generates positive edge after accounting for real-world slippage and taxes.

### Step 2.1: Run the Synthetic Backtester
The `backtester.py` file contains a highly realistic Heston+Jumps synthetic market generator. We use this to test if NOVA can survive regime shifts.
1. Open your terminal.
2. Run a full 1-year backtest with the NOVA pipeline (which involves the Neural J-SDE and Deep Hedger):
   ```bash
   python3 backtester.py --days 252 --model omega
   ```
   *(Note: The backtester uses `model='omega'` to wrap the full pipeline logic).*
3. The backtester executes trades at the *next day's* simulated bid/ask prices and deducts ₹20 flat brokerage plus 0.0625% STT.
4. *Crucial Metric:* Check the **Sharpe Ratio** and **Max Drawdown**. A good theoretical model might have a Sharpe of 1.5. A truly robust model maintains Sharpe > 1.0 even with full transaction costs.

### Step 2.2: Live "Paper" Incubation (The Shadow Circuit Breaker)
Theoretical and synthetic tests are great, but the ultimate test is live data.
1. Open Tab 9 (NOVA).
2. For the next 2 weeks, whenever you intend to make a manual trade, **run the NOVA pipeline on that option first**.
3. Do not execute the trade yourself. Just let NOVA run. This will push the signal into the **Shadow Feedback Queue** (which you wired during hardening).
4. Do this for ~50 trades.
5. After 2 weeks, check the **Shadow Status** diagnostics in Tab 9.
6. *Crucial Metric:* What is the Win Rate of the shadow queue? Did the cross-exchange (BSE) validator correctly flag bad trades? If the shadow queue's pseudo-P&L is positive, **NOVA is empirically validated.**

---

## The "Rank 9" Checklist

You can confidently say this system is a 9/10 when you have these three numbers recorded:
- [ ] OMEGA Out-of-Sample MAPE < 15% (from Historical Learning pull)
- [ ] NOVA Synthetic Sharpe Ratio > 1.0 (from `backtester.py`)
- [ ] Shadow Queue Win Rate > 55% over 50+ live-market observations

Everything you need to get these numbers is already built into your codebase. You just need to run it.
