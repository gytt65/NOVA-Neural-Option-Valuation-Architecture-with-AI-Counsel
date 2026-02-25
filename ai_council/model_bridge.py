from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class ModelHealthResult:
    findings: List[Dict[str, Any]]
    disagreement_score: float
    actionability_penalty: float


class ModelBridge:
    """Extracts OMEGA/NIRV/NOVA outputs and computes consistency diagnostics."""

    def __init__(self, session_state: Mapping[str, Any]):
        self.state = session_state

    def read_outputs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "omega": self._read_omega(),
            "nirv": self._read_nirv(),
            "nova": self._read_nova(),
        }

    def evaluate_health(self, outputs: Optional[Dict[str, Dict[str, Any]]] = None) -> ModelHealthResult:
        outputs = outputs or self.read_outputs()
        findings: List[Dict[str, Any]] = []
        penalty = 0.0

        signals = [x.get("signal") for x in outputs.values() if x.get("signal")]
        unique_signals = {str(s).upper() for s in signals if s is not None}
        if len(unique_signals) >= 2:
            findings.append(
                {
                    "type": "MODEL_SIGNAL_CONFLICT",
                    "severity": "high",
                    "details": f"Conflicting signals: {sorted(unique_signals)}",
                    "recommended_action": "Downgrade to HOLD unless evidence quality is high.",
                }
            )
            penalty += 20.0

        for model, data in outputs.items():
            conf = _safe_float(data.get("confidence"), 0.0)
            if conf < 35 and data.get("signal"):
                findings.append(
                    {
                        "type": "LOW_MODEL_CONFIDENCE",
                        "severity": "medium",
                        "model": model,
                        "details": f"{model.upper()} confidence={conf:.1f}%",
                        "recommended_action": "Reduce position size or wait for confirmation.",
                    }
                )
                penalty += 5.0

        spread_pct = _safe_float(self.state.get("bid_ask_spread_pct"), 0.0)
        if spread_pct > 2.5:
            findings.append(
                {
                    "type": "WIDE_SPREAD",
                    "severity": "high",
                    "details": f"Bid/ask spread is {spread_pct:.2f}%",
                    "recommended_action": "Treat directional signals as non-executable.",
                }
            )
            penalty += 25.0

        quote_age = _safe_float(self.state.get("quote_age_seconds"), 0.0)
        if quote_age > 180:
            findings.append(
                {
                    "type": "STALE_DATA",
                    "severity": "high",
                    "details": f"Quote age {quote_age:.0f}s",
                    "recommended_action": "Refresh live data before acting.",
                }
            )
            penalty += 20.0

        disagreement = min(100.0, penalty)
        return ModelHealthResult(findings=findings, disagreement_score=disagreement, actionability_penalty=penalty)

    def _read_omega(self) -> Dict[str, Any]:
        res = self.state.get("omega_result")
        if res is None:
            return {}
        return {
            "signal": _signal_string(getattr(res, "signal", None)),
            "confidence": _safe_float(getattr(res, "confidence_level", 0.0), 0.0),
            "regime": getattr(res, "regime", ""),
            "fair_value": _safe_float(getattr(res, "fair_value", 0.0), 0.0),
            "market_price": _safe_float(getattr(res, "market_price", 0.0), 0.0),
            "mispricing_pct": _safe_float(getattr(res, "mispricing_pct", 0.0), 0.0),
        }

    def _read_nirv(self) -> Dict[str, Any]:
        res = self.state.get("nirv_result")
        if res is None:
            return {}
        return {
            "signal": _signal_string(getattr(res, "signal", None)),
            "confidence": _safe_float(getattr(res, "confidence_level", 0.0), 0.0),
            "regime": getattr(res, "regime", ""),
            "fair_value": _safe_float(getattr(res, "fair_value", 0.0), 0.0),
            "market_price": _safe_float(getattr(res, "market_price", 0.0), 0.0),
            "mispricing_pct": _safe_float(getattr(res, "mispricing_pct", 0.0), 0.0),
        }

    def _read_nova(self) -> Dict[str, Any]:
        # NOVA may be stored as dict output in session.
        res = self.state.get("nova_last_result") or self.state.get("nova_result")
        if res is None:
            return {}
        if isinstance(res, dict):
            return {
                "signal": _signal_string(res.get("signal") or res.get("action")),
                "confidence": _safe_float(res.get("confidence") or res.get("confidence_pct"), 0.0),
                "regime": res.get("regime", ""),
                "fair_value": _safe_float(res.get("final_price") or res.get("fair_value"), 0.0),
                "market_price": _safe_float(res.get("market_price"), 0.0),
                "mispricing_pct": _safe_float(res.get("mispricing_pct"), 0.0),
            }
        return {
            "signal": _signal_string(getattr(res, "signal", None)),
            "confidence": _safe_float(getattr(res, "confidence", 0.0), 0.0),
            "regime": getattr(res, "regime", ""),
            "fair_value": _safe_float(getattr(res, "final_price", 0.0), 0.0),
            "market_price": _safe_float(getattr(res, "market_price", 0.0), 0.0),
            "mispricing_pct": _safe_float(getattr(res, "mispricing_pct", 0.0), 0.0),
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _signal_string(signal: Any) -> Optional[str]:
    if signal is None:
        return None
    text = str(signal)
    text = text.replace("✅", "").replace("🔴", "").replace("⚠️", "").replace("🚀", "")
    text = text.replace("⏸️", "").replace("✖", "")
    text = " ".join(text.split()).upper()
    if "STRONG BUY" in text:
        return "BUY"
    if "STRONG SELL" in text:
        return "SELL"
    if "BUY" in text:
        return "BUY"
    if "SELL" in text:
        return "SELL"
    if "HOLD" in text:
        return "HOLD"
    return text or None
