from ai_council.orchestrator import AICouncilOrchestrator
from ai_council.types import CouncilContext
from ai_council.worker import CouncilWorker


def _ctx(spot, vix, signal, high_impact_news=0):
    news = [{"title": "Event", "impact": "high"} for _ in range(high_impact_news)]
    return CouncilContext(
        timestamp_ist="2026-02-25T10:00:00+05:30",
        underlying="Nifty 50",
        spot=float(spot),
        expiry="2026-02-27",
        chain_slice=[],
        microstructure={"spread_pct": 1.0, "volume_oi_ratio": 0.1},
        macro_data={"india_vix": float(vix)},
        corporate_events=[],
        news_events=news,
        model_outputs={"omega": {"signal": signal, "confidence": 70.0}},
        model_health={"findings": []},
        data_freshness={"market_open": True},
    )


def test_worker_event_trigger_high_impact_news_increase():
    worker = CouncilWorker(AICouncilOrchestrator(), provider_map={})
    worker._last_context = _ctx(22500, 14.0, "BUY", high_impact_news=0)
    assert worker._is_event_trigger(_ctx(22500, 14.0, "BUY", high_impact_news=1))


def test_worker_event_trigger_model_signal_flip():
    worker = CouncilWorker(AICouncilOrchestrator(), provider_map={})
    worker._last_context = _ctx(22500, 14.0, "BUY", high_impact_news=0)
    assert worker._is_event_trigger(_ctx(22500, 14.0, "SELL", high_impact_news=0))
