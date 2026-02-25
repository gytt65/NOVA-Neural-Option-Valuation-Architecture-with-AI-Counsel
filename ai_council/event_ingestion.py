from __future__ import annotations

import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class EventShock:
    direction_bias: float
    vol_bias: float
    confidence_boost: float


_CORP_KEYWORDS = {
    "results": ["result", "earnings", "quarter", "guidance", "profit", "loss"],
    "dividend": ["dividend", "bonus", "record date", "ex-date", "split", "buyback"],
    "policy": ["rbi", "fed", "rate", "policy", "inflation", "cpi", "gdp", "budget"],
    "risk": ["war", "conflict", "sanction", "tariff", "geopolitical", "default", "downgrade"],
}

_SOURCE_QUALITY = {
    "rbi": 0.95,
    "fed": 0.95,
    "ecb": 0.92,
    "reuters": 0.90,
    "bloomberg": 0.90,
    "moneycontrol": 0.78,
    "economic_times": 0.76,
    "livemint": 0.74,
    "calendar": 0.70,
    "perplexity": 0.68,
    "unknown": 0.55,
}


def normalize_news_events(raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in raw_events or []:
        title = str(item.get("title") or item.get("headline") or "").strip()
        if not title:
            continue
        sentiment = str(item.get("sentiment") or "neutral").lower()
        published = str(item.get("published") or item.get("time") or "")
        source = str(item.get("source") or item.get("provider") or "unknown")
        category = classify_event_category(title)
        impact = classify_impact(title, sentiment)
        staleness = compute_staleness_seconds(published)
        quality = source_quality_score(source)
        normalized.append(
            {
                "title": title,
                "sentiment": sentiment,
                "published": published,
                "source": source,
                "category": category,
                "impact": impact,
                "staleness_seconds": staleness,
                "quality_score": quality,
            }
        )
    return normalized


def normalize_corporate_events(raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in raw_events or []:
        title = str(item.get("title") or item.get("event") or "").strip()
        if not title:
            continue
        category = classify_event_category(title)
        if category not in {"results", "dividend"}:
            continue
        sentiment = str(item.get("sentiment") or infer_sentiment_from_title(title)).lower()
        effective_date = str(item.get("date") or item.get("effective_date") or "")
        normalized.append(
            {
                "title": title,
                "sentiment": sentiment,
                "category": category,
                "impact": classify_impact(title, sentiment),
                "effective_date": effective_date,
                "staleness_seconds": compute_staleness_seconds(effective_date),
                "quality_score": source_quality_score(str(item.get("source") or "calendar")),
            }
        )
    return normalized


def classify_event_category(title: str) -> str:
    t = (title or "").lower()
    for category, words in _CORP_KEYWORDS.items():
        if any(w in t for w in words):
            return category
    return "market"


def infer_sentiment_from_title(title: str) -> str:
    t = (title or "").lower()
    positive = ["beat", "surge", "upgrade", "record high", "growth", "buyback", "dividend"]
    negative = ["miss", "fall", "downgrade", "warning", "default", "loss", "fraud"]
    pos_hits = sum(1 for w in positive if w in t)
    neg_hits = sum(1 for w in negative if w in t)
    if pos_hits > neg_hits:
        return "bullish"
    if neg_hits > pos_hits:
        return "bearish"
    return "neutral"


def classify_impact(title: str, sentiment: str) -> str:
    t = (title or "").lower()
    s = (sentiment or "neutral").lower()
    if any(k in t for k in ["rbi", "fed", "budget", "war", "default", "rating downgrade"]):
        return "high"
    if any(k in t for k in ["results", "earnings", "dividend", "buyback"]):
        return "medium"
    if s in {"bullish", "bearish"}:
        return "medium"
    return "low"


def compute_event_shock(news_events: List[Dict[str, Any]], corporate_events: List[Dict[str, Any]]) -> EventShock:
    direction = 0.0
    vol = 0.0
    conf = 0.0

    for ev in (news_events or []) + (corporate_events or []):
        s = str(ev.get("sentiment", "neutral")).lower()
        impact = str(ev.get("impact", "low")).lower()
        quality = max(0.1, min(1.0, float(ev.get("quality_score", 0.6) or 0.6)))
        staleness_seconds = max(0.0, float(ev.get("staleness_seconds", 0.0) or 0.0))
        freshness_penalty = max(0.4, 1.0 - min(0.6, staleness_seconds / 86400.0))
        weight = (1.0 if impact == "low" else 1.8 if impact == "medium" else 2.8) * quality * freshness_penalty
        if s == "bullish":
            direction += 0.8 * weight
        elif s == "bearish":
            direction -= 0.8 * weight
        if impact == "high":
            vol += 0.25
            conf += 0.04
        elif impact == "medium":
            vol += 0.12
            conf += 0.02

    # Normalize to sensible bounds.
    direction = max(-2.5, min(2.5, direction / 8.0))
    vol = max(0.0, min(0.9, vol))
    conf = max(0.0, min(0.2, conf))
    return EventShock(direction_bias=direction, vol_bias=vol, confidence_boost=conf)


def source_quality_score(source: str) -> float:
    key = str(source or "unknown").strip().lower()
    if key in _SOURCE_QUALITY:
        return _SOURCE_QUALITY[key]
    for name, score in _SOURCE_QUALITY.items():
        if name != "unknown" and name in key:
            return score
    return _SOURCE_QUALITY["unknown"]


def compute_staleness_seconds(ts_text: str) -> float:
    ts = str(ts_text or "").strip()
    if not ts:
        return 0.0
    candidates = [ts.replace("Z", "+00:00"), ts]
    for c in candidates:
        try:
            dt = datetime.fromisoformat(c)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return max(0.0, (now - dt.astimezone(timezone.utc)).total_seconds())
        except Exception:
            continue
    return 0.0


def extract_watchlist_terms(news_events: List[Dict[str, Any]], corporate_events: List[Dict[str, Any]]) -> List[str]:
    terms: List[str] = []
    pattern = re.compile(r"\b([A-Z]{2,6})\b")
    for ev in (news_events or []) + (corporate_events or []):
        title = str(ev.get("title", ""))
        for hit in pattern.findall(title):
            if hit not in terms:
                terms.append(hit)
        cat = str(ev.get("category", ""))
        if cat and cat not in terms:
            terms.append(cat.upper())
    return terms[:25]


def summarize_events(news_events: List[Dict[str, Any]], corporate_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    shock = compute_event_shock(news_events, corporate_events)
    return {
        "news_count": len(news_events or []),
        "corporate_count": len(corporate_events or []),
        "direction_bias": shock.direction_bias,
        "vol_bias": shock.vol_bias,
        "confidence_boost": shock.confidence_boost,
        "watchlist": extract_watchlist_terms(news_events, corporate_events),
    }
