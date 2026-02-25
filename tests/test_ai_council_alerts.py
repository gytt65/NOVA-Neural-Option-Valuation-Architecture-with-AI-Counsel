from ai_council.alerts import AlertConfig, CouncilAlertPublisher


def test_alert_priority_gate_without_telegram():
    pub = CouncilAlertPublisher(AlertConfig(enable_telegram=False, min_priority="high"))
    verdict = {
        "final_signal": "BUY",
        "confidence": 80,
        "telegram_priority": "high",
        "actionability": True,
        "actionability_reasons": [],
        "disagreement_score": 18,
    }
    payload = pub.publish(verdict, "Nifty 50")
    assert payload["signal"] == "BUY"
    assert payload["telegram"]["sent"] is False


def test_alert_priority_threshold_logic():
    cfg = AlertConfig(enable_telegram=True, telegram_bot_token="x", telegram_chat_id="y", min_priority="high")
    pub = CouncilAlertPublisher(cfg)
    assert pub.should_send_telegram("critical")
    assert pub.should_send_telegram("high")
    assert not pub.should_send_telegram("medium")
