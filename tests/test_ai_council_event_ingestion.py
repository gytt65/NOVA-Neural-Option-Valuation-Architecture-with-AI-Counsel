from ai_council.event_ingestion import (
    compute_event_shock,
    normalize_news_events,
    source_quality_score,
)


def test_source_quality_score_prefers_tier1():
    assert source_quality_score("Reuters") > source_quality_score("unknown")


def test_normalize_news_events_adds_quality_and_staleness():
    events = normalize_news_events([
        {"title": "RBI policy surprise", "source": "RBI", "published": "2026-02-25T10:00:00+05:30", "sentiment": "bullish"}
    ])
    assert events
    ev = events[0]
    assert "quality_score" in ev and "staleness_seconds" in ev


def test_compute_event_shock_respects_quality_weighting():
    hi = [{"title": "Macro", "sentiment": "bullish", "impact": "high", "quality_score": 0.9, "staleness_seconds": 60}]
    lo = [{"title": "Macro", "sentiment": "bullish", "impact": "high", "quality_score": 0.3, "staleness_seconds": 60}]
    s_hi = compute_event_shock(hi, [])
    s_lo = compute_event_shock(lo, [])
    assert s_hi.direction_bias > s_lo.direction_bias
