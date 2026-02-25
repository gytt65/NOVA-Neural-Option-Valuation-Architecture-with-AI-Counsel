from dataclasses import dataclass
import tempfile

from ai_council.orchestrator import AICouncilOrchestrator, CouncilConfig
from ai_council.memory_store import CouncilMemoryStore
from ai_council.types import AgentOpinion, CouncilContext, CritiqueReport


@dataclass
class StaticAgent:
    agent_id: str
    op: AgentOpinion

    def opinion(self, ctx):
        return self.op

    def critique(self, own, target, ctx):
        sev = 20.0 if own.direction != target.direction else 5.0
        return CritiqueReport(
            critic_agent_id=self.agent_id,
            target_agent_id=target.agent_id,
            issues=["direction_conflict"] if sev > 10 else [],
            severity=sev,
            evidence_gap=max(0.0, own.evidence_score - target.evidence_score),
            schema_violations=[],
        )


def _context():
    return CouncilContext(
        timestamp_ist="2026-02-25T10:00:00+05:30",
        underlying="Nifty 50",
        spot=22500.0,
        expiry="2026-02-27",
        chain_slice=[],
        microstructure={"spread_pct": 1.2, "volume_oi_ratio": 0.08, "quote_age_seconds": 20},
        macro_data={"india_vix": 14.0},
        corporate_events=[],
        news_events=[],
        model_outputs={
            "omega": {"signal": "BUY", "confidence": 70},
            "nirv": {"signal": "BUY", "confidence": 60},
            "nova": {"signal": "HOLD", "confidence": 45},
        },
        model_health={"findings": [], "disagreement_score": 0.0},
        data_freshness={"quote_age_seconds": 20, "market_open": True},
    )


def test_orchestrator_outputs_normalized_forecasts():
    ctx = _context()
    agents = [
        StaticAgent(
            "a1",
            AgentOpinion(
                agent_id="a1",
                direction="BUY",
                confidence=72,
                expected_move_pct=0.5,
                iv_view="expand",
                evidence_score=70,
                citations=[],
                risk_flags=[],
                recommended_params={},
                scenario_probs={"up": 0.6, "down": 0.2, "flat": 0.2},
                rationale="",
            ),
        ),
        StaticAgent(
            "a2",
            AgentOpinion(
                agent_id="a2",
                direction="BUY",
                confidence=68,
                expected_move_pct=0.4,
                iv_view="neutral",
                evidence_score=66,
                citations=[],
                risk_flags=[],
                recommended_params={},
                scenario_probs={"up": 0.55, "down": 0.2, "flat": 0.25},
                rationale="",
            ),
        ),
    ]
    out = AICouncilOrchestrator(CouncilConfig()).run_cycle(ctx, agents)
    assert out.final_signal in {"BUY", "HOLD"}
    assert out.probabilistic_forecast
    for f in out.probabilistic_forecast:
        total = f.p_up + f.p_down + f.p_flat
        assert abs(total - 1.0) < 1e-3
        assert f.q05 <= f.q25 <= f.q50 <= f.q75 <= f.q95


def test_orchestrator_forces_hold_on_high_disagreement():
    ctx = _context()
    agents = [
        StaticAgent(
            "a1",
            AgentOpinion(
                agent_id="a1",
                direction="BUY",
                confidence=90,
                expected_move_pct=1.2,
                iv_view="expand",
                evidence_score=58,
                citations=[],
                risk_flags=[],
                recommended_params={},
                scenario_probs={"up": 0.9, "down": 0.05, "flat": 0.05},
                rationale="",
            ),
        ),
        StaticAgent(
            "a2",
            AgentOpinion(
                agent_id="a2",
                direction="SELL",
                confidence=90,
                expected_move_pct=-1.2,
                iv_view="contract",
                evidence_score=58,
                citations=[],
                risk_flags=[],
                recommended_params={},
                scenario_probs={"up": 0.05, "down": 0.9, "flat": 0.05},
                rationale="",
            ),
        ),
    ]
    out = AICouncilOrchestrator(CouncilConfig()).run_cycle(ctx, agents)
    assert out.final_signal == "HOLD"
    assert any("DISAGREEMENT" in r or "LOW_EVIDENCE" in r for r in out.actionability_reasons)


def test_orchestrator_calibration_shrinks_to_neutral_when_unreliable():
    class _MemoryStub:
        def get_reliability(self, horizon):
            return {"calibration_score": 0.0}

        def save_cycle(self, underlying, verdict):
            return 1

        def upsert_provider_reliability(self, provider_name, sample_count, success_rate, avg_latency_ms):
            return None

        def upsert_seat_reliability(self, seat_id, sample_count, hit_rate, calibration_score):
            return None

        def upsert_evidence_quality(self, source, sample_count, avg_quality, avg_staleness):
            return None

    ctx = _context()
    agents = [
        StaticAgent(
            "a1",
            AgentOpinion(
                agent_id="a1",
                direction="BUY",
                confidence=90,
                expected_move_pct=1.0,
                iv_view="expand",
                evidence_score=90,
                citations=[],
                risk_flags=[],
                recommended_params={},
                scenario_probs={"up": 0.95, "down": 0.02, "flat": 0.03},
                rationale="",
            ),
        )
    ]
    out = AICouncilOrchestrator(CouncilConfig(), memory_store=_MemoryStub()).run_cycle(ctx, agents)
    fc = out.probabilistic_forecast[0]
    # calibration score 0.0 should neutralize direction toward ~balanced priors
    assert abs(fc.p_up - 0.33) < 0.03
    assert abs(fc.p_down - 0.33) < 0.03


def test_orchestrator_updates_reliability_using_previous_cycle_spot():
    with tempfile.TemporaryDirectory() as td:
        db = CouncilMemoryStore(f"{td}/council.db")
        orch = AICouncilOrchestrator(CouncilConfig(), memory_store=db)
        agents = [
            StaticAgent(
                "a1",
                AgentOpinion(
                    agent_id="a1",
                    direction="BUY",
                    confidence=72,
                    expected_move_pct=0.5,
                    iv_view="expand",
                    evidence_score=70,
                    citations=["https://example.com"],
                    risk_flags=[],
                    recommended_params={},
                    scenario_probs={"up": 0.6, "down": 0.2, "flat": 0.2},
                    rationale="",
                    seat_id="risk_manager_cro",
                ),
            ),
        ]

        first_ctx = _context()
        orch.run_cycle(first_ctx, agents)

        second_ctx = _context()
        second_ctx.spot = 22600.0
        orch.run_cycle(second_ctx, agents)

        rel = db.get_reliability("next_day")
        assert rel["sample_count"] >= 1


def test_orchestrator_updates_quality_metrics_tables():
    with tempfile.TemporaryDirectory() as td:
        db = CouncilMemoryStore(f"{td}/council.db")
        orch = AICouncilOrchestrator(CouncilConfig(), memory_store=db)
        agents = [
            StaticAgent(
                "a1",
                AgentOpinion(
                    agent_id="a1",
                    direction="BUY",
                    confidence=70,
                    expected_move_pct=0.4,
                    iv_view="neutral",
                    evidence_score=65,
                    citations=["https://example.com"],
                    risk_flags=[],
                    recommended_params={},
                    scenario_probs={"up": 0.55, "down": 0.2, "flat": 0.25},
                    rationale="",
                    seat_id="risk_manager_cro",
                ),
            ),
        ]
        ctx = _context()
        ctx.model_health = {
            "findings": [],
            "providers_active": ["openai"],
            "providers_expected": ["openai"],
            "provider_degraded": False,
        }
        orch.run_cycle(ctx, agents)
        seat = db.get_seat_reliability("risk_manager_cro")
        provider = db.get_provider_reliability("openai")
        assert seat["sample_count"] >= 1
        assert provider["sample_count"] >= 1


def test_orchestrator_flags_unverified_staleness_for_directional_citations():
    ctx = _context()
    agents = [
        StaticAgent(
            "a1",
            AgentOpinion(
                agent_id="a1",
                direction="BUY",
                confidence=80,
                expected_move_pct=0.8,
                iv_view="expand",
                evidence_score=75,
                citations=["https://example.com/no-ts"],
                risk_flags=[],
                recommended_params={},
                scenario_probs={"up": 0.8, "down": 0.1, "flat": 0.1},
                rationale="",
                seat_id="risk_manager_cro",
                evidence_items=[],
            ),
        ),
    ]
    out = AICouncilOrchestrator(CouncilConfig()).run_cycle(ctx, agents)
    assert out.final_signal == "HOLD"
    assert "SOURCE_STALENESS_UNVERIFIED" in out.actionability_reasons
