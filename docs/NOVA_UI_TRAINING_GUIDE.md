# How to Train NOVA using the Streamlit UI and Upstox/Angel One (Free APIs)

If you are running the `opmAI_app.py` Streamlit interface and want to utilize your free Upstox or Angel One APIs to train the NOVA model, you don't need to write any custom Python scripts. The system already has a built-in **Historical Learning Engine** designed specifically to handle the rate limits of these free APIs.

Here is the step-by-step process to train NOVA to its maximum capability using only the app's user interface.

---

## Prerequisites: Fixing the Data Bug
Before you run this process, we **must** deploy the fix for the `spot = strike` bug in `historical_learning.py`. Upstox historical APIs frequently return missing spot prices for option ticks. If unpatched, the system fakes the spot price, which instantly corrupts the Implied Volatility calculations and ruins the Neural SDE training. *(Ensure this bug fix from the audit is applied first).*

---

## Step 1: Connect Your Broker in the UI
1. Launch the app `streamlit run opmAI_app.py`.
2. Open the **Sidebar** on the left.
3. Under **🔐 Connection & Settings**, select the **Upstox** tab.
4. Enter your API Key, API Secret, and Redirect URI.
5. Click **"🔗 Open Login Page"**, authorize yourself, and paste the `Authorization Code` back into the app.
6. Click **"🔑 Get Access Token"**. 
   *You should see a green "✅ Connected" box.*

*(Note: While Angel One SmartAPI is available in the sidebar, the deep Historical Learning module is currently hardwired to the Upstox API engine for parallel candle fetching. Ensure Upstox is connected).*

---

## Step 2: Configure the Historical Learning Module
Scroll down in the Sidebar until you find the section titled **🧠 Historical Learning**. 

This is the control center for NOVA's training data. Free APIs have strict rate limits (e.g., 100 requests per second), so this module manages the slow, careful extraction of thousands of 1-minute candles to build the "Snapshots" NOVA needs.

Configure the parameters:

1. **Underlying Instrument Key:** Leave as `NSE_INDEX|Nifty 50` (Do not train NOVA on illiquid stocks first; the math is tuned for the index).
2. **From Date / To Date:**
   - **Crucial Tip:** Do not try to download 1 year of data at once. Upstox will rate-limit you, and it will take 5 hours.
   - Start with a **2-week window** (e.g., the last 14 days) to train the model quickly and verify it works. Later, you can run a 3-month pull overnight.
3. **Candle Interval:** Select `day` or `minute`. *(Note: 1-minute candles provide exponentially better training for the Hawkes jump process, but take much longer to pull).*
4. **Contract Selection:** Choose `ATM Strike Window`. 
5. **ATM ± Strike Count:** Set to `6`. This pulls 6 strikes above and 6 strikes below the spot price. (12 Call and 12 Put options per expiry).
6. **Fast Mode (recommended):** 
   - **CHECK THIS BOX for your first run.** It restricts the data pull to fewer expiries and pages, preventing API timeouts and allowing the training to finish in ~3 minutes.
   - **UNCHECK THIS BOX for final production training.** It will pull deeper into the option chain, giving the KAN corrector much better data for the deep OTM wings.

---

## Step 3: Initiate the Pull & Learn Sequence
Click the massive button: **🚀 Pull & Learn**

Here is exactly what the app is doing autonomously once you click that button:
1. **The Pull:** It hits the Upstox Historical API, calculating exactly how many 1-minute historical candles it needs for the Nifty Spot and the 24 option contracts over your date range.
2. **The Alignment:** It synchronizes the timestamps. If an option traded at 10:01 AM but the Nifty spot didn't update until 10:02 AM, it handles the alignment to prevent IV corruption.
3. **The Snapshot Generation:** It converts thousands of rows of CSV candles into the structured "Market Snapshot" dictionaries that NOVA requires.
4. **The Training:** It feeds these snapshots into `train_from_historical_report()` in `unified_pipeline.py`.
   - It trains the Hawkes jump process.
   - It calibrates the Heston COS physics parameters.
   - It trains the Neural Jump-SDE on path signatures.
   - It trains the KAN interpretable corrector on the residual errors.

You will see a progress bar updating. Do not close the browser tab.

---

## Step 4: Reviewing the Training Report
Once the progress bar reaches 100%, a green success message will appear in the sidebar. Below the "Pull & Learn" button, small grey text will output the training diagnostics.

Look for these key metrics:
- **Rows pulled:** (Should be > 10,000 for a multi-day minute-level pull)
- **Rows processed:** (The clean data surviving the synchronization filters)
- **RMSE (Root Mean Square Error):** This is the ultimate health check of the KAN corrector. You want to see an RMSE under `5.0` (meaning the model's fair value prediction is, on average, within ₹5 of the actual market price).

---

## Step 5: Live Execution
The weights of the fully trained NOVA pipeline (including the Neural SDE and KAN) have now been automatically saved to your local disk by the app.

1. Scroll back up the sidebar to **🎯 Select Contract**.
2. Select an option (e.g., 1 DTE, 24000 CE).
3. Look at the main dashboard tabs.
4. Go to **Tab 9: NOVA**.

Because the model was just trained, NOVA will instantly load the saved weights and output the ultra-precise Corrected Fair Value, along with the Hawkes cluster risk analysis and the KAN B-spline component breakdowns based on the live data flowing from Upstox.

**You are now running institutional-grade AI on a free retail API.**
