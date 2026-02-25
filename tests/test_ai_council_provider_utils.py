from ai_council.providers import parse_first_json_object, validate_payload


def test_parse_first_json_object_handles_noisy_text():
    text = "analysis start... {\"direction\": \"BUY\", \"confidence\": 70, \"expected_move_pct\": 0.6, \"iv_view\": \"expand\", \"evidence_score\": 75, \"citations\": [], \"risk_flags\": [], \"recommended_params\": {}, \"scenario_probs\": {\"up\": 0.5, \"down\": 0.2, \"flat\": 0.3}, \"rationale\": \"ok\"} trailing"
    parsed = parse_first_json_object(text)
    assert parsed["direction"] == "BUY"
    assert parsed["confidence"] == 70


def test_validate_payload_normalizes_probabilities():
    payload = {
        "direction": "SELL",
        "confidence": 66,
        "expected_move_pct": -0.4,
        "iv_view": "contract",
        "evidence_score": 61,
        "citations": [],
        "risk_flags": [],
        "recommended_params": {},
        "scenario_probs": {"up": 3, "down": 1, "flat": 0},
        "rationale": "normalized",
    }
    ok, issues = validate_payload(payload)
    assert ok
    assert not issues
    probs = payload["scenario_probs"]
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["up"] > probs["down"] > probs["flat"]
