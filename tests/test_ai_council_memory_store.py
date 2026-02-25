import tempfile

from ai_council.memory_store import CouncilMemoryStore


def test_memory_store_extended_tables_work():
    with tempfile.TemporaryDirectory() as td:
        db = CouncilMemoryStore(f"{td}/council.db")
        cid = db.save_cycle("Nifty 50", {"final_signal": "HOLD", "confidence": 55.0})
        assert cid > 0

        db.upsert_seat_reliability("risk_manager_cro", 1, 0.5, 0.6)
        seat = db.get_seat_reliability("risk_manager_cro")
        assert seat["sample_count"] >= 1

        db.upsert_provider_reliability("openai", 1, 1.0, 120.0)
        prov = db.get_provider_reliability("openai")
        assert prov["sample_count"] >= 1

        db.upsert_evidence_quality("citation", 1, 0.7, 120.0)
        ev = db.get_evidence_quality("citation")
        assert ev["sample_count"] >= 1

        oid = db.save_verdict_outcome(cid, "next_day", 0.12, "UP", True)
        assert oid > 0
