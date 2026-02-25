from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests


@dataclass
class AlertConfig:
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    enable_telegram: bool = False
    min_priority: str = "high"


_PRIORITY = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class CouncilAlertPublisher:
    def __init__(self, config: AlertConfig):
        self.config = config
        self._inbox: List[Dict[str, Any]] = []

    def publish(self, verdict: Dict[str, Any], underlying: str) -> Dict[str, Any]:
        priority = str(verdict.get("telegram_priority", "low")).lower()
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "underlying": underlying,
            "priority": priority,
            "signal": verdict.get("final_signal", "HOLD"),
            "confidence": verdict.get("confidence", 0.0),
            "actionability": verdict.get("actionability", False),
            "reasons": verdict.get("actionability_reasons", []),
            "disagreement_score": verdict.get("disagreement_score", 0.0),
        }
        self._inbox.append(payload)
        if len(self._inbox) > 200:
            self._inbox = self._inbox[-200:]

        telegram_status = {"sent": False, "error": None}
        if self.should_send_telegram(priority):
            ok, err = self._send_telegram(self._format_telegram(verdict, underlying))
            telegram_status = {"sent": bool(ok), "error": err}
        payload["telegram"] = telegram_status
        return payload

    def pull_inbox(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self._inbox[-int(limit):])

    def should_send_telegram(self, priority: str) -> bool:
        if not self.config.enable_telegram:
            return False
        if not (self.config.telegram_bot_token and self.config.telegram_chat_id):
            return False
        p = _PRIORITY.get(str(priority).lower(), 1)
        floor = _PRIORITY.get(str(self.config.min_priority).lower(), 3)
        return p >= floor

    def _format_telegram(self, verdict: Dict[str, Any], underlying: str) -> str:
        signal = str(verdict.get("final_signal", "HOLD"))
        conf = float(verdict.get("confidence", 0.0) or 0.0)
        dis = float(verdict.get("disagreement_score", 0.0) or 0.0)
        reasons = verdict.get("actionability_reasons", []) or []
        reason_txt = " | ".join([str(r) for r in reasons[:4]]) if reasons else "None"
        return (
            "🤖 <b>AI Council Alert</b>\n"
            f"<b>Underlying:</b> {underlying}\n"
            f"<b>Signal:</b> {signal}\n"
            f"<b>Confidence:</b> {conf:.1f}%\n"
            f"<b>Disagreement:</b> {dis:.1f}\n"
            f"<b>Reasons:</b> {reason_txt}"
        )

    def _send_telegram(self, message: str) -> tuple[bool, Optional[str]]:
        try:
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            data = {
                "chat_id": self.config.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                return True, None
            return False, response.text
        except Exception as exc:  # pragma: no cover - non-deterministic network errors
            return False, str(exc)
