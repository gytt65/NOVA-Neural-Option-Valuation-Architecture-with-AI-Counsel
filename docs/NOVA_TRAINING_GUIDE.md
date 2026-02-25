# How to Train the NOVA Pipeline: A Step-by-Step Guide

Welcome. This guide assumes you are starting from scratch and need to know exactly how to feed the NOVA (Nobel-caliber Options Valuation Architecture) model the pristine data it needs to unlock its true power.

NOVA is essentially a high-performance race car. If you feed it bad data (like the current `spot=strike` fallback bug), the engine will flood. If you feed it hyper-clean, synchronized data, it will find mispricings that no other retail model can see.

---

## Step 1: Acquiring the Data (Where to Get It)

NOVA needs two things:
1. **Historical Nifty 50 Spot Prices** (Minute-level or Daily)
2. **Historical Nifty 50 Option Chain Data** (Snapshot data — ideally end-of-day or 15-minute intervals)

### The Free/Cheap Route (What you have now):
- **Upstox API / Angel One API:** You already have integrations for these in `upstox_api_clients.py`. 
- **Pros:** Free (if using your own broker account API).
- **Cons:** Extremely rate-limited. Historical option chain data is often patchy, missing, or misaligned with the spot price.

### The Institutional Route (What you *should* use for full capability):
If you want NOVA to run at 100% capacity, you need to buy professional historical data.
- **Global Datafeeds (GFDL):** The gold standard for Indian retail quants. They sell historical 1-minute snapshot data for the entire Nifty option chain.
- **TrueData:** Excellent API, very clean historical options data for India.
- **NSE Data Room:** You can buy historical tick-by-tick data directly from the National Stock Exchange of India (expensive but flawless).

---

## Step 2: Formatting the Data (How NOVA Wants It)

NOVA does not ingest raw CSV files directly. It expects data to be organized into a specific Python dictionary structure called a **Report**.

Your goal is to parse whatever CSV/API data you get into a list of "Snapshots".

### The Required Format:
A "Snapshot" is a dictionary representing the state of the market at one specific moment in time (e.g., December 1st, 2024 at 3:30 PM).

```python
snapshot = {
    "date": "2024-12-01",              # The date of the snapshot
    "spot": 23500.25,                  # The exact Nifty 50 cash/spot price at that moment
    "india_vix": 14.2,                 # The official India VIX at that moment
    "fii_net_flow": -1200.50,          # (Optional but recommended) FII cash market flows in Crores
    "dii_net_flow": 800.00,            # (Optional but recommended) DII cash market flows
    "pcr_oi": 0.85,                    # (Optional) Put-Call Ratio based on Open Interest
    "returns_30d": [0.01, -0.002, ...],# Array of the last 30 days of daily log returns of Nifty 50
    
    # The actual option chain for this snapshot
    "chain": [
        {
            "strike": 23500.0,
            "expiry_date": "2024-12-05", 
            "option_type": "CE",       # CE for Call, PE for Put
            "market_price": 120.50,    # The LTP or Mid-price of the option
            "volume": 2500000,         # Optional
            "oi": 1500000              # Optional
        },
        {
            "strike": 23500.0,
            "expiry_date": "2024-12-05",
            "option_type": "PE",
            "market_price": 115.20
        },
        # ... include all liquid strikes (usually +/- 1000 points from spot)
    ]
}
```

**CRITICAL RULE:** The `spot` price inside the snapshot **MUST EXACTLY MATCH** the time the `market_price` of the options was recorded. If the spot is from 3:30 PM but the option price is from 3:15 PM, NOVA will calculate an incorrect Implied Volatility (IV), and the entire model will learn garbage.

---

## Step 3: Feeding the Data to NOVA

Once you have a list of these snapshots (the "Report"), feeding it to NOVA is straightforward. 

First, instantiate the pipeline:

```python
from unified_pipeline import UnifiedPricingPipeline

# Initialize the full NOVA pipeline
nova = UnifiedPricingPipeline()
```

Next, wrap your list of snapshots in a report dictionary:

```python
# Assuming you parsed your CSVs into a list of snapshot dictionaries
my_snapshots = [snapshot_day_1, snapshot_day_2, ..., snapshot_day_N]

training_report = {
    "snapshots": my_snapshots
}
```

Now, command NOVA to train itself:

```python
print("Starting NOVA Training Sequence...")

# This one function call trains ALL modules in the correct dependency order
nova.train_from_historical_report(
    report=training_report,
    r=0.065,              # Indian Risk-Free Rate (e.g., 6.5%)
    q=0.012,              # Nifty 50 Dividend Yield (e.g., 1.2%)
    max_option_snapshots=1000  # How many snapshots to process (more = better, but slower)
)

print("Training Complete!")
```

---

## Step 4: What Happens During Training (The "Under the Hood" Magic)

When you run `train_from_historical_report`, NOVA does not just train one model. It orchestrates a symphony of training sequences in a strict order:

1. **Hawkes Jump Process Training:** 
   - NOVA looks at the `returns_30d` array across all your snapshots.
   - It identifies every market crash and spike.
   - It trains the Hawkes process to learn *how frequently* and *how violently* Nifty crashes cluster together.

2. **Heston COS Calibration:**
   - NOVA looks at the option chains.
   - It calibrates the physics parameters (κ, θ, ξ, ρ) of the Heston stochastic volatility model so that the model's base prices match the market prices as closely as possible.

3. **Neural J-SDE Training (The Heavy Lifter):**
   - NOVA takes the path signatures (the trajectory of Spot, VIX, PCR) and feeds them into the Neural Network.
   - The Neural Network learns the exact state-dependent Drift, Diffusion, and Jump intensity parameters. *(Note: This step requires Adam+SPSA optimization to work properly).*

4. **KAN Corrector Training:**
   - NOVA takes the prices generated by the Heston model and compares them to the true market prices in your training data.
   - It calculates the "residual error" (e.g., Heston priced it at ₹100, market was ₹105. Error = +5%).
   - The KAN (Kolmogorov-Arnold Network) trains its B-splines to predict this exact error based on the market features. 

5. **Deep Hedger Training:**
   - NOVA simulates 5000 paths forward.
   - It trains a separate neural network to learn the optimal hedge ratio (how much Nifty futures to short against your options) to minimize tail risk (CVaR).

---

## Step 5: Saving the Trained Brain

Once training is complete, you must save the "weights" (the learned parameters) so you don't have to retrain every time you restart the app.

```python
# Save the fully trained NOVA brain to disk
nova.save(filepath='models/nova_brain_v1.joblib')
```

When you launch the `opmAI_app.py` tomorrow, it will automatically load these weights if they exist in the `models/` directory.

---

## Step 6: Running NOVA Live

To use the 100% capacity of the model live, you need to provide it the exact same feature structure you trained it on. When you want to price an option right now:

```python
# Construct the current market state
current_market_state = {
    "vix": 15.4,
    "fii_net_flow": 450.0,
    "dii_net_flow": -100.0,
    "pcr_oi": 1.1,
    # The path signature tool computes this from recent history
    "path_signature": current_signature_vector 
}

# Ask NOVA for the true value of a Call option
result = nova.price(
    spot=23550.0,
    strike=24000.0,
    T=5.0 / 365.0,        # 5 days to expiry
    r=0.065,
    q=0.012,
    sigma=15.4 / 100.0,   # Current VIX as base volatility
    option_type='CE',
    market_state=current_market_state,
    historical_returns=recent_30d_returns
)

print(f"NOVA says Fair Value is: ₹{result['ensemble_price']:.2f}")
print(f"KAN Confidence Interval: {result['confidence_interval']}")
```

---

## The Master Checklist for Success

1. **[  ] Get Better Data:** Ditch the free brokers for historical training data. Buy a clean snapshot dataset from GFDL or TrueData.
2. **[  ] Fix the Code Bugs First:** Before you train, the `deep_hedging.py` Nelder-Mead bug and the `historical_learning.py` spot fallback bug *must* be fixed, or you are wasting pristine data on a broken engine.
3. **[  ] Train Deep & Wide:** Provide at least 6 months of daily snapshot data covering both a Bull run and a Bear correction so the Regime detector (HMM) learns what both look like.
4. **[  ] Run the Backtester:** Before trading real money, feed your trained NOVA into `backtester.py` to prove it generates positive PnL after slippage and taxes.
