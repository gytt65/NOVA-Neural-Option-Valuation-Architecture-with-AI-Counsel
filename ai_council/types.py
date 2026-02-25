from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Direction = Literal["BUY", "SELL", "HOLD"]
IVView = Literal["expand", "contract", "neutral"]
Horizon = Literal["intraday_60m", "today_close", "next_day", "next_week"]
PersonaStyle = Literal["archetype", "named", "hybrid"]


@dataclass
class AgentProfile:
    seat_id: str
    display_name: str
    domain: str
    persona_style: PersonaStyle = "hybrid"
    provider_priority: List[str] = field(default_factory=lambda: ["gemini", "openai", "perplexity", "anthropic"])
    temperature_cap: float = 0.2
    must_cite_min: int = 1
    max_confidence_cap: float = 85.0
    seat_weight: float = 1.0
    reliability_prior: float = 0.60
    voting: bool = True
    enabled: bool = True


@dataclass
class EvidenceItem:
    source: str
    timestamp: str = ""
    url: str = ""
    relevance: float = 0.5
    staleness_seconds: float = 0.0
    quality_score: float = 0.5


@dataclass
class CouncilContext:
    timestamp_ist: str
    underlying: str
    spot: float
    expiry: Optional[str] = None
    chain_slice: List[Dict[str, Any]] = field(default_factory=list)
    microstructure: Dict[str, Any] = field(default_factory=dict)
    macro_data: Dict[str, Any] = field(default_factory=dict)
    corporate_events: List[Dict[str, Any]] = field(default_factory=list)
    news_events: List[Dict[str, Any]] = field(default_factory=list)
    model_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    model_health: Dict[str, Any] = field(default_factory=dict)
    data_freshness: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentOpinion:
    agent_id: str
    direction: Direction
    confidence: float
    expected_move_pct: float
    iv_view: IVView
    evidence_score: float
    citations: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    recommended_params: Dict[str, Any] = field(default_factory=dict)
    scenario_probs: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    seat_id: str = ""
    thesis: str = ""
    counterfactual: str = ""
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    uncertainty_notes: List[str] = field(default_factory=list)
    self_critique: str = ""


@dataclass
class CritiqueReport:
    critic_agent_id: str
    target_agent_id: str
    issues: List[str] = field(default_factory=list)
    severity: float = 0.0
    evidence_gap: float = 0.0
    schema_violations: List[str] = field(default_factory=list)


@dataclass
class ProbabilisticForecast:
    horizon: Horizon
    p_up: float
    p_down: float
    p_flat: float
    return_mu: float
    return_sigma: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    p_iv_expand: float
    p_iv_contract: float
    p_gap_up: float
    p_gap_down: float
    calibration_score: float


@dataclass
class CouncilVerdict:
    final_signal: Direction
    confidence: float
    consensus_score: float
    disagreement_score: float
    actionability: bool
    actionability_reasons: List[str] = field(default_factory=list)
    probabilistic_forecast: List[ProbabilisticForecast] = field(default_factory=list)
    parameter_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    model_health_findings: List[Dict[str, Any]] = field(default_factory=list)
    watchlist: List[str] = field(default_factory=list)
    telegram_priority: Literal["low", "medium", "high", "critical"] = "low"
    disagreement_matrix: List[Dict[str, Any]] = field(default_factory=list)
    agent_opinions: List[AgentOpinion] = field(default_factory=list)
    critiques: List[CritiqueReport] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    seat_votes: List[Dict[str, Any]] = field(default_factory=list)
    veto_flags: List[str] = field(default_factory=list)
    quality_gate_status: Dict[str, Any] = field(default_factory=dict)
    operator_checklist: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def _evidence(ev: EvidenceItem) -> Dict[str, Any]:
            return {
                "source": ev.source,
                "timestamp": ev.timestamp,
                "url": ev.url,
                "relevance": float(ev.relevance),
                "staleness_seconds": float(ev.staleness_seconds),
                "quality_score": float(ev.quality_score),
            }

        def _opinion(op: AgentOpinion) -> Dict[str, Any]:
            return {
                "agent_id": op.agent_id,
                "direction": op.direction,
                "confidence": float(op.confidence),
                "expected_move_pct": float(op.expected_move_pct),
                "iv_view": op.iv_view,
                "evidence_score": float(op.evidence_score),
                "citations": list(op.citations),
                "risk_flags": list(op.risk_flags),
                "recommended_params": dict(op.recommended_params),
                "scenario_probs": dict(op.scenario_probs),
                "rationale": op.rationale,
                "seat_id": op.seat_id,
                "thesis": op.thesis,
                "counterfactual": op.counterfactual,
                "evidence_items": [_evidence(x) for x in op.evidence_items],
                "uncertainty_notes": list(op.uncertainty_notes),
                "self_critique": op.self_critique,
            }

        def _critique(cr: CritiqueReport) -> Dict[str, Any]:
            return {
                "critic_agent_id": cr.critic_agent_id,
                "target_agent_id": cr.target_agent_id,
                "issues": list(cr.issues),
                "severity": float(cr.severity),
                "evidence_gap": float(cr.evidence_gap),
                "schema_violations": list(cr.schema_violations),
            }

        def _forecast(fc: ProbabilisticForecast) -> Dict[str, Any]:
            return {
                "horizon": fc.horizon,
                "p_up": float(fc.p_up),
                "p_down": float(fc.p_down),
                "p_flat": float(fc.p_flat),
                "return_mu": float(fc.return_mu),
                "return_sigma": float(fc.return_sigma),
                "q05": float(fc.q05),
                "q25": float(fc.q25),
                "q50": float(fc.q50),
                "q75": float(fc.q75),
                "q95": float(fc.q95),
                "p_iv_expand": float(fc.p_iv_expand),
                "p_iv_contract": float(fc.p_iv_contract),
                "p_gap_up": float(fc.p_gap_up),
                "p_gap_down": float(fc.p_gap_down),
                "calibration_score": float(fc.calibration_score),
            }

        return {
            "final_signal": self.final_signal,
            "confidence": float(self.confidence),
            "consensus_score": float(self.consensus_score),
            "disagreement_score": float(self.disagreement_score),
            "actionability": bool(self.actionability),
            "actionability_reasons": list(self.actionability_reasons),
            "probabilistic_forecast": [_forecast(x) for x in self.probabilistic_forecast],
            "parameter_recommendations": list(self.parameter_recommendations),
            "model_health_findings": list(self.model_health_findings),
            "watchlist": list(self.watchlist),
            "telegram_priority": self.telegram_priority,
            "disagreement_matrix": list(self.disagreement_matrix),
            "agent_opinions": [_opinion(x) for x in self.agent_opinions],
            "critiques": [_critique(x) for x in self.critiques],
            "metadata": dict(self.metadata),
            "seat_votes": list(self.seat_votes),
            "veto_flags": list(self.veto_flags),
            "quality_gate_status": dict(self.quality_gate_status),
            "operator_checklist": list(self.operator_checklist),
        }
