from types import SimpleNamespace

from ai_council.model_bridge import ModelBridge


def test_model_bridge_detects_conflict_and_execution_issues():
    state = {
        "omega_result": SimpleNamespace(signal="✅ BUY", confidence_level=72.0, regime="Bull", fair_value=120, market_price=100, mispricing_pct=-20),
        "nirv_result": SimpleNamespace(signal="🔴 SELL", confidence_level=68.0, regime="Bear", fair_value=80, market_price=100, mispricing_pct=25),
        "bid_ask_spread_pct": 3.4,
        "quote_age_seconds": 420,
    }
    bridge = ModelBridge(state)
    outputs = bridge.read_outputs()
    health = bridge.evaluate_health(outputs)

    assert outputs["omega"]["signal"] == "BUY"
    assert outputs["nirv"]["signal"] == "SELL"
    types = {x["type"] for x in health.findings}
    assert "MODEL_SIGNAL_CONFLICT" in types
    assert "WIDE_SPREAD" in types
    assert "STALE_DATA" in types
    assert health.disagreement_score > 40
