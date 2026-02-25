#!/usr/bin/env python3
"""
cross_exchange_validator.py — BSE Kicker: Cross-Exchange Signal Validation
==========================================================================

Validates Nifty (NSE) mispricing signals against equivalent Sensex (BSE)
strikes.  Nifty–Sensex correlation is > 0.98: if NIRV/OMEGA says a Nifty
option is mispriced but the equivalent Sensex strike disagrees, the Nifty
signal is likely a data artifact or liquidity vacuum, not real alpha.

Usage
-----
    from cross_exchange_validator import CrossExchangeValidator

    validator = CrossExchangeValidator()

    # Feed BSE chain data when available
    validator.update_bse_chain(bse_strikes, bse_prices, bse_ivs, sensex_spot)

    # Validate a Nifty signal
    result = validator.validate_signal(
        nifty_spot=24500, nifty_strike=24000, nifty_iv=0.14,
        nifty_mispricing_pct=3.5, option_type='CE'
    )

    if result['confirmed']:
        # True cross-validated signal — high confidence
        ...
    else:
        # Likely data artifact — reduce size or skip
        ...

Reference
---------
The Nifty → Sensex mapping uses a fixed ratio approach:
    equivalent_sensex_strike = (nifty_strike / nifty_spot) * sensex_spot

Because both indices track very similar baskets of Indian large-caps,
an identical moneyness level should have nearly identical IV structure.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple


class CrossExchangeValidator:
    """
    Cross-Exchange Signal Validator (BSE Kicker).

    Validates Nifty (NSE) mispricing signals against Sensex (BSE) data.
    The core insight: Nifty–Sensex have > 0.98 correlation, so their
    IV surfaces should agree at equivalent moneyness. Disagreement
    implies the Nifty signal is noise, not alpha.

    Three validation layers:
      1. IV Parity — Nifty IV ≈ Sensex IV at same moneyness
      2. Mispricing Parity — if Nifty is mispriced, Sensex should be too
      3. Spread Anomaly — wide Nifty spread with tight Sensex spread = data issue
    """

    # Historical Nifty:Sensex ratio (approximately 0.3:1)
    # e.g., Nifty 24000 ≈ Sensex 80000 → ratio ≈ 0.30
    NIFTY_SENSEX_RATIO_DEFAULT = 0.30

    # IV tolerance: how much IV can differ before flagging discrepancy
    IV_TOLERANCE = 0.03  # 3 vol points

    # Mispricing confirmation threshold: BSE must agree within this %
    MISPRICING_TOLERANCE = 0.5  # ±0.5% agreement

    def __init__(self, iv_tolerance: float = 0.03, mispricing_tolerance: float = 0.5):
        self.iv_tolerance = iv_tolerance
        self.mispricing_tolerance = mispricing_tolerance

        # BSE chain cache
        self._bse_strikes: Optional[np.ndarray] = None
        self._bse_prices: Optional[Dict[float, float]] = None  # {strike: mid_price}
        self._bse_ivs: Optional[Dict[float, float]] = None     # {strike: iv}
        self._bse_bids: Optional[Dict[float, float]] = None
        self._bse_asks: Optional[Dict[float, float]] = None
        self._sensex_spot: float = 0.0
        self._nifty_sensex_ratio: float = self.NIFTY_SENSEX_RATIO_DEFAULT
        self._last_update_time: float = 0.0

        # Historical correlation tracker
        self._iv_diff_history: list = []  # tracks rolling IV discrepancy

    def update_bse_chain(self, strikes: List[float], prices: Dict[float, float],
                         ivs: Dict[float, float], sensex_spot: float,
                         nifty_spot: float = 0.0, update_time: float = 0.0,
                         bids: Optional[Dict[float, float]] = None,
                         asks: Optional[Dict[float, float]] = None):
        """
        Feed in the latest BSE Sensex option chain data.

        Parameters
        ----------
        strikes     : list — sorted BSE strike prices
        prices      : dict — {strike: mid_price} for Sensex options
        ivs         : dict — {strike: implied_vol} for Sensex options
        sensex_spot : float — current Sensex spot price
        nifty_spot  : float — current Nifty spot (to compute live ratio)
        update_time : float — Unix timestamp of this snapshot
        bids/asks   : dict — optional bid/ask prices for spread validation
        """
        self._bse_strikes = np.asarray(sorted(strikes), dtype=float)
        self._bse_prices = dict(prices)
        self._bse_ivs = dict(ivs)
        self._sensex_spot = float(sensex_spot)
        self._bse_bids = dict(bids) if bids else None
        self._bse_asks = dict(asks) if asks else None
        self._last_update_time = update_time

        # Update live Nifty/Sensex ratio
        if nifty_spot > 0 and sensex_spot > 0:
            self._nifty_sensex_ratio = nifty_spot / sensex_spot

    def _find_equivalent_sensex_strike(self, nifty_strike: float,
                                        nifty_spot: float) -> Optional[float]:
        """
        Map a Nifty strike to the equivalent Sensex strike at same moneyness.

        Equivalent moneyness:
            K_sensex / S_sensex = K_nifty / S_nifty
            → K_sensex = K_nifty × (S_sensex / S_nifty)
        """
        if self._sensex_spot <= 0 or nifty_spot <= 0 or self._bse_strikes is None:
            return None

        target_sensex_strike = nifty_strike * (self._sensex_spot / nifty_spot)

        # Find nearest available BSE strike
        idx = np.argmin(np.abs(self._bse_strikes - target_sensex_strike))
        nearest = float(self._bse_strikes[idx])

        # Only accept if within 1% of target (strikes are spaced differently)
        if abs(nearest - target_sensex_strike) / target_sensex_strike > 0.01:
            return None

        return nearest

    def validate_signal(self, nifty_spot: float, nifty_strike: float,
                        nifty_iv: float, nifty_mispricing_pct: float,
                        option_type: str = 'CE',
                        nifty_bid: Optional[float] = None,
                        nifty_ask: Optional[float] = None) -> Dict:
        """
        Validate a Nifty mispricing signal against BSE Sensex data.

        Returns
        -------
        dict with:
            confirmed      : bool — True if BSE confirms the Nifty signal
            confidence     : float — 0–1 confidence in the signal
            rejection_reason : str — why the signal was rejected (if applicable)
            sensex_strike  : float — equivalent Sensex strike used
            iv_diff        : float — Nifty IV - Sensex IV at same moneyness
            bse_mispricing : float — equivalent BSE mispricing signal (if available)
        """
        result = {
            'confirmed': False,
            'confidence': 0.0,
            'rejection_reason': None,
            'sensex_strike': None,
            'iv_diff': None,
            'bse_mispricing': None,
            'checks_passed': [],
            'checks_failed': [],
        }

        # ── Guard: BSE data must be available ──
        if self._bse_strikes is None or self._bse_ivs is None:
            result['rejection_reason'] = 'NO_BSE_DATA'
            result['confirmed'] = True  # fail-open: don't block without data
            result['confidence'] = 0.3  # but low confidence
            return result

        # ── Step 1: Find equivalent Sensex strike ──
        sensex_strike = self._find_equivalent_sensex_strike(nifty_strike, nifty_spot)
        if sensex_strike is None:
            result['rejection_reason'] = 'NO_EQUIVALENT_STRIKE'
            result['confirmed'] = True  # fail-open
            result['confidence'] = 0.3
            return result

        result['sensex_strike'] = sensex_strike
        sensex_iv = self._bse_ivs.get(sensex_strike)

        # ── Layer 1: IV Parity Check ──
        if sensex_iv is not None and sensex_iv > 0:
            iv_diff = nifty_iv - sensex_iv
            result['iv_diff'] = round(float(iv_diff), 4)

            # Track for rolling statistics
            self._iv_diff_history.append(iv_diff)
            if len(self._iv_diff_history) > 200:
                self._iv_diff_history = self._iv_diff_history[-100:]

            if abs(iv_diff) <= self.iv_tolerance:
                result['checks_passed'].append('IV_PARITY')
            else:
                result['checks_failed'].append(f'IV_PARITY (diff={iv_diff:.4f})')
                # Large IV discrepancy → Nifty IV anomaly is likely data artifact
                if abs(iv_diff) > self.iv_tolerance * 2:
                    result['rejection_reason'] = 'IV_DIVERGENCE'

        # ── Layer 2: Mispricing Parity Check ──
        # If Nifty says "3.5% underpriced", Sensex should also show mispricing
        # at the equivalent strike. If Sensex disagrees, Nifty signal is noise.
        sensex_price = self._bse_prices.get(sensex_strike) if self._bse_prices else None
        if sensex_price and sensex_price > 0 and sensex_iv and sensex_iv > 0:
            # Rough mispricing proxy for Sensex: compare market IV vs Nifty IV
            # If Nifty IV is "too low" (underpriced), Sensex IV should also be low
            nifty_moneyness = np.log(nifty_spot / nifty_strike)
            sensex_moneyness = np.log(self._sensex_spot / sensex_strike)

            # At same moneyness, IV should be very similar
            iv_ratio = nifty_iv / max(sensex_iv, 0.001)

            if 0.85 <= iv_ratio <= 1.15:
                result['checks_passed'].append('MISPRICING_PARITY')
            else:
                result['checks_failed'].append(f'MISPRICING_PARITY (ratio={iv_ratio:.3f})')

            # Estimate BSE-side mispricing direction
            if iv_ratio < 0.9:
                result['bse_mispricing'] = 'NIFTY_IV_LOW'  # Nifty underpriced vs BSE
            elif iv_ratio > 1.1:
                result['bse_mispricing'] = 'NIFTY_IV_HIGH'  # Nifty overpriced vs BSE
            else:
                result['bse_mispricing'] = 'ALIGNED'

        # ── Layer 3: Spread Anomaly Check ──
        # Wide Nifty spread + tight Sensex spread → Nifty liquidity vacuum
        if (nifty_bid is not None and nifty_ask is not None
                and self._bse_bids and self._bse_asks):
            nifty_spread = (nifty_ask - nifty_bid) / max((nifty_bid + nifty_ask) / 2, 0.01)
            bse_bid = self._bse_bids.get(sensex_strike, 0)
            bse_ask = self._bse_asks.get(sensex_strike, 0)

            if bse_bid > 0 and bse_ask > bse_bid:
                sensex_spread = (bse_ask - bse_bid) / ((bse_bid + bse_ask) / 2)

                if nifty_spread > 0.02 and sensex_spread < 0.01:
                    # Nifty has wide spread while Sensex is tight
                    # → Nifty "mispricing" is just illiquidity, not real
                    result['checks_failed'].append(
                        f'SPREAD_ANOMALY (nifty={nifty_spread:.3f}, sensex={sensex_spread:.3f})'
                    )
                    if result['rejection_reason'] is None:
                        result['rejection_reason'] = 'LIQUIDITY_VACUUM'
                else:
                    result['checks_passed'].append('SPREAD_CHECK')

        # ── Final Decision ──
        n_passed = len(result['checks_passed'])
        n_failed = len(result['checks_failed'])
        n_total = n_passed + n_failed

        if n_total == 0:
            # No checks possible (missing data)
            result['confirmed'] = True  # fail-open
            result['confidence'] = 0.3
        elif n_failed == 0:
            # All checks passed → strong confirmation
            result['confirmed'] = True
            result['confidence'] = min(0.95, 0.6 + 0.15 * n_passed)
        elif n_passed >= n_failed:
            # More passes than fails → cautious confirmation
            result['confirmed'] = True
            result['confidence'] = 0.5 * (n_passed / max(n_total, 1))
        else:
            # More fails → reject the signal
            result['confirmed'] = False
            result['confidence'] = 0.1
            if result['rejection_reason'] is None:
                result['rejection_reason'] = 'CROSS_EXCHANGE_DISAGREEMENT'

        return result

    def iv_surface_coherence(self) -> Dict:
        """
        Compute rolling Nifty-Sensex IV surface coherence statistics.

        Returns
        -------
        dict with mean_iv_diff, std_iv_diff, is_coherent, n_observations
        """
        if len(self._iv_diff_history) < 5:
            return {'is_coherent': True, 'n_observations': 0}

        diffs = np.array(self._iv_diff_history)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        recent_mean = float(np.mean(diffs[-20:])) if len(diffs) >= 20 else mean_diff

        # Coherent if mean IV diff is small and stable
        is_coherent = abs(recent_mean) < 0.02 and std_diff < 0.04

        return {
            'mean_iv_diff': round(mean_diff, 4),
            'std_iv_diff': round(std_diff, 4),
            'recent_mean_iv_diff': round(recent_mean, 4),
            'is_coherent': is_coherent,
            'n_observations': len(diffs),
        }

    def get_status(self) -> Dict:
        """Return current validator status for UI display."""
        return {
            'bse_data_available': self._bse_strikes is not None,
            'sensex_spot': self._sensex_spot,
            'n_bse_strikes': len(self._bse_strikes) if self._bse_strikes is not None else 0,
            'nifty_sensex_ratio': round(self._nifty_sensex_ratio, 4),
            'coherence': self.iv_surface_coherence(),
            'last_update_time': self._last_update_time,
        }
