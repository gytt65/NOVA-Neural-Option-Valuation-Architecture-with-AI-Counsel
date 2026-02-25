from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

LOGGER = logging.getLogger(__name__)


DEFAULT_AGENT_SCHEMA: Dict[str, Any] = {
    "direction": str,
    "confidence": (int, float),
    "expected_move_pct": (int, float),
    "iv_view": str,
    "evidence_score": (int, float),
    "citations": list,
    "risk_flags": list,
    "recommended_params": dict,
    "scenario_probs": dict,
    "rationale": str,
    "seat_id": str,
    "thesis": str,
    "counterfactual": str,
    "evidence_items": list,
    "uncertainty_notes": list,
    "self_critique": str,
}

_COST_TIER_SCORE = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


@dataclass
class ProviderResponse:
    provider: str
    model: str
    payload: Dict[str, Any]
    raw_text: str = ""
    citations: List[str] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class ProviderRoutingPolicy:
    cost_mode: str = "mostly_free"  # mostly_free | quality_first | cost_capped
    max_daily_budget_usd: float = 8.0
    spent_today_usd: float = 0.0
    spent_day_utc: str = ""
    strict_hold_bias_on_failure: bool = True

    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if not self.spent_day_utc:
            self.spent_day_utc = today
            return
        if self.spent_day_utc != today:
            self.spent_day_utc = today
            self.spent_today_usd = 0.0

    def estimate_call_cost(self, provider: Optional["LLMProvider"]) -> float:
        tier = "medium"
        if provider is not None:
            tier = str(getattr(provider, "cost_tier", "medium") or "medium").lower()
        return {"low": 0.002, "medium": 0.01, "high": 0.02}.get(tier, 0.01)

    def record_call(self, provider: Optional["LLMProvider"], multiplier: float = 1.0) -> None:
        self._roll_day()
        self.spent_today_usd += max(0.0, self.estimate_call_cost(provider) * max(0.0, float(multiplier)))

    def choose_provider(
        self,
        provider_map: Dict[str, "LLMProvider"],
        provider_priority: List[str],
        high_impact: bool,
    ) -> Optional["LLMProvider"]:
        self._roll_day()
        candidates: List[LLMProvider] = []
        for name in provider_priority:
            p = provider_map.get(name)
            if p is None or not p.enabled:
                continue
            candidates.append(p)
        if not candidates:
            return None

        if self.cost_mode == "quality_first":
            return candidates[0]

        if self.cost_mode == "cost_capped" and self.max_daily_budget_usd > 0:
            if self.spent_today_usd >= 0.9 * self.max_daily_budget_usd:
                cheap = [p for p in candidates if _COST_TIER_SCORE.get(p.cost_tier, 2) <= 1]
                if cheap:
                    return cheap[0]

        if self.cost_mode == "mostly_free" and not high_impact:
            cheap = [p for p in candidates if _COST_TIER_SCORE.get(p.cost_tier, 2) <= 1]
            if cheap:
                return cheap[0]
            medium = [p for p in candidates if _COST_TIER_SCORE.get(p.cost_tier, 2) <= 2]
            if medium:
                return medium[0]

        return candidates[0]


class LLMProvider:
    """Adapter wrapper around hosted model APIs with metadata for routing."""

    def __init__(
        self,
        name: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        timeout: int = 30,
        cost_tier: str = "medium",
        latency_tier: str = "medium",
        reliability_score: float = 0.70,
    ):
        self.name = name
        self.api_key = (api_key or "").strip()
        self.model = model
        self.temperature = float(temperature)
        self.timeout = int(timeout)
        self.cost_tier = str(cost_tier).lower()
        self.latency_tier = str(latency_tier).lower()
        self.reliability_score = float(reliability_score)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def generate_json(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        if not self.enabled:
            return ProviderResponse(provider=self.name, model=self.model, payload={}, error="provider_disabled")
            
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if self.name == "gemini":
                    raw, latency = _call_gemini(self.api_key, self.model, system_prompt, user_prompt, self.temperature, self.timeout)
                elif self.name == "perplexity":
                    raw, latency = _call_perplexity(self.api_key, self.model, system_prompt, user_prompt, self.temperature, self.timeout)
                elif self.name == "openai":
                    raw, latency = _call_openai(self.api_key, self.model, system_prompt, user_prompt, self.temperature, self.timeout)
                elif self.name == "anthropic":
                    raw, latency = _call_anthropic(self.api_key, self.model, system_prompt, user_prompt, self.temperature, self.timeout)
                else:
                    return ProviderResponse(provider=self.name, model=self.model, payload={}, error="unknown_provider")
                parsed = parse_first_json_object(raw)
                cites = extract_citations(raw)
                return ProviderResponse(
                    provider=self.name,
                    model=self.model,
                    payload=parsed,
                    raw_text=raw,
                    citations=cites,
                    latency_ms=latency,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Provider %s attempt %d failed: %s", self.name, attempt + 1, exc)
                if attempt == max_retries - 1:
                    return ProviderResponse(provider=self.name, model=self.model, payload={}, error=str(exc))
                time.sleep(base_delay * (2 ** attempt))
                
        return ProviderResponse(provider=self.name, model=self.model, payload={}, error="max_retries_exceeded")


def build_strict_json_prompt(role_name: str, schema: Optional[Dict[str, Any]] = None) -> str:
    schema = schema or DEFAULT_AGENT_SCHEMA
    fields = "\n".join([f"- {k}" for k in schema.keys()])
    return (
        f"You are {role_name}. Return ONLY one strict JSON object with these keys:\n"
        f"{fields}\n"
        "No markdown. No code fences. No commentary."
    )


def validate_payload(payload: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
    schema = schema or DEFAULT_AGENT_SCHEMA
    issues: List[str] = []
    if not isinstance(payload, dict):
        return False, ["payload_not_dict"]

    optional_defaults: Dict[str, Any] = {
        "seat_id": "",
        "thesis": "",
        "counterfactual": "",
        "evidence_items": [],
        "uncertainty_notes": [],
        "self_critique": "",
    }

    for key, key_type in schema.items():
        if key not in payload:
            if key in optional_defaults:
                payload[key] = optional_defaults[key]
                continue
            issues.append(f"missing:{key}")
            continue
        val = payload.get(key)
        if not isinstance(val, key_type):
            issues.append(f"bad_type:{key}")

    probs = payload.get("scenario_probs")
    if isinstance(probs, dict):
        up = _safe_float(probs.get("up"), 0.0)
        down = _safe_float(probs.get("down"), 0.0)
        flat = _safe_float(probs.get("flat"), 0.0)
        total = up + down + flat
        if total <= 1e-9:
            payload["scenario_probs"] = {"up": 0.34, "down": 0.33, "flat": 0.33}
        else:
            payload["scenario_probs"] = {
                "up": max(0.0, up) / total,
                "down": max(0.0, down) / total,
                "flat": max(0.0, flat) / total,
            }

    evidence_items = payload.get("evidence_items")
    if isinstance(evidence_items, list):
        normalized: List[Dict[str, Any]] = []
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "source": str(item.get("source", "unknown")),
                    "timestamp": str(item.get("timestamp", "")),
                    "url": str(item.get("url", "")),
                    "relevance": max(0.0, min(1.0, _safe_float(item.get("relevance"), 0.5))),
                    "staleness_seconds": max(0.0, _safe_float(item.get("staleness_seconds"), 0.0)),
                    "quality_score": max(0.0, min(1.0, _safe_float(item.get("quality_score"), 0.5))),
                }
            )
        payload["evidence_items"] = normalized

    return len(issues) == 0, issues


def extract_citations(text: str) -> List[str]:
    if not text:
        return []
    urls = re.findall(r"https?://[^\s\]\)\">]+", text)
    clean = []
    for url in urls:
        if url not in clean:
            clean.append(url)
    return clean[:12]


def parse_first_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    stripped = text.strip()
    try:
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def _call_gemini(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature: float, timeout: int) -> Tuple[str, float]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": 2048,
        },
        "contents": [{"parts": [{"text": user_prompt}]}],
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return "", 0.0
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        return "", 0.0
    return str(parts[0].get("text", "") or ""), _safe_float(response.elapsed.total_seconds() * 1000.0, 0.0)


def _call_perplexity(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature: float, timeout: int) -> Tuple[str, float]:
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": 2048,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return "", 0.0
    text = str(choices[0].get("message", {}).get("content", "") or "")
    return text, _safe_float(response.elapsed.total_seconds() * 1000.0, 0.0)


def _call_openai(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature: float, timeout: int) -> Tuple[str, float]:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return "", 0.0
    text = str(choices[0].get("message", {}).get("content", "") or "")
    return text, _safe_float(response.elapsed.total_seconds() * 1000.0, 0.0)


def _call_anthropic(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature: float, timeout: int) -> Tuple[str, float]:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 2048,
        "temperature": float(temperature),
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    blocks = data.get("content") or []
    if not blocks:
        return "", 0.0
    text = "\n".join([str(b.get("text", "")) for b in blocks if isinstance(b, dict)])
    return text, _safe_float(response.elapsed.total_seconds() * 1000.0, 0.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)
