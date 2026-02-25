from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .providers import (
    LLMProvider,
    ProviderRoutingPolicy,
    build_strict_json_prompt,
    extract_citations,
    validate_payload,
)
from .types import AgentOpinion, AgentProfile, CouncilContext, CritiqueReport, EvidenceItem


DEFAULT_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "senate_profiles.yaml")


@dataclass
class BaseAgent:
    agent_id: str
    role: str
    provider: Optional[LLMProvider] = None
    temperature: float = 0.1

    def opinion(self, ctx: CouncilContext) -> AgentOpinion:  # pragma: no cover - overridden
        raise NotImplementedError

    def critique(self, own: AgentOpinion, target: AgentOpinion, ctx: CouncilContext) -> CritiqueReport:
        issues: List[str] = []
        severity = 0.0

        if target.evidence_score < 55:
            issues.append("weak_evidence")
            severity += 20
        if target.confidence > 80 and target.evidence_score < 70:
            issues.append("overconfident_vs_evidence")
            severity += 15
        if target.direction != own.direction and abs(target.expected_move_pct - own.expected_move_pct) > 0.7:
            issues.append("directional_conflict")
            severity += 12

        return CritiqueReport(
            critic_agent_id=self.agent_id,
            target_agent_id=target.agent_id,
            issues=issues,
            severity=min(100.0, severity),
            evidence_gap=max(0.0, own.evidence_score - target.evidence_score),
            schema_violations=[],
        )


@dataclass
class PersonaAgent(BaseAgent):
    profile: AgentProfile = None  # type: ignore[assignment]
    provider_map: Dict[str, LLMProvider] = None  # type: ignore[assignment]
    routing_policy: Optional[ProviderRoutingPolicy] = None

    def opinion(self, ctx: CouncilContext) -> AgentOpinion:
        provider = self._pick_provider(ctx)
        prompt = self._build_prompt(ctx)
        if provider and provider.enabled:
            sys_prompt = build_strict_json_prompt(self.role)
            resp = provider.generate_json(sys_prompt, prompt)
            if self.routing_policy is not None:
                self.routing_policy.record_call(provider)
            if not resp.error:
                ok, issues = validate_payload(resp.payload)
                if ok:
                    payload = dict(resp.payload)
                    citations = list(payload.get("citations") or [])
                    if not citations:
                        citations = resp.citations
                    payload["citations"] = citations
                    payload["seat_id"] = str(payload.get("seat_id") or self.profile.seat_id)
                    return self._from_payload(payload)
                return self._fallback(ctx, reason=f"schema_issues:{','.join(issues)}")
            return self._fallback(ctx, reason=f"provider_error:{provider.name}:{resp.error}")
        return self._fallback(ctx, reason="provider_unavailable")

    def critique(self, own: AgentOpinion, target: AgentOpinion, ctx: CouncilContext) -> CritiqueReport:
        if self.profile.seat_id == "red_team_prosecutor":
            issues: List[str] = []
            severity = 0.0
            if target.direction != "HOLD" and target.evidence_score < 70:
                issues.append("insufficient_evidence_for_directional_call")
                severity += 25
            if len(target.citations) < self.profile.must_cite_min:
                issues.append("citation_deficit")
                severity += 25
            if target.confidence > self.profile.max_confidence_cap:
                issues.append("confidence_cap_breach")
                severity += 20
            return CritiqueReport(
                critic_agent_id=self.agent_id,
                target_agent_id=target.agent_id,
                issues=issues,
                severity=min(100.0, severity),
                evidence_gap=max(0.0, own.evidence_score - target.evidence_score),
                schema_violations=[],
            )
        return super().critique(own, target, ctx)

    def _pick_provider(self, ctx: CouncilContext) -> Optional[LLMProvider]:
        high_impact = _is_high_impact_context(ctx)
        routing = self.routing_policy or ProviderRoutingPolicy()
        return routing.choose_provider(self.provider_map or {}, self.profile.provider_priority, high_impact)

    def _from_payload(self, payload: Dict[str, Any]) -> AgentOpinion:
        probs = payload.get("scenario_probs") or {"up": 0.34, "down": 0.33, "flat": 0.33}
        citations = [str(x) for x in payload.get("citations", []) if str(x).strip()]
        evidence_items = _build_evidence_items(payload.get("evidence_items"), citations)

        confidence = float(payload.get("confidence", 50.0))
        confidence = min(confidence, float(self.profile.max_confidence_cap))
        if len(citations) < int(self.profile.must_cite_min):
            confidence = min(confidence, 55.0)

        return AgentOpinion(
            agent_id=self.agent_id,
            direction=_norm_direction(payload.get("direction")),
            confidence=confidence,
            expected_move_pct=float(payload.get("expected_move_pct", 0.0)),
            iv_view=_norm_iv_view(payload.get("iv_view")),
            evidence_score=float(payload.get("evidence_score", 50.0)),
            citations=citations[:12],
            risk_flags=[str(x) for x in payload.get("risk_flags", [])][:12],
            recommended_params=dict(payload.get("recommended_params", {})),
            scenario_probs={
                "up": float(probs.get("up", 0.34)),
                "down": float(probs.get("down", 0.33)),
                "flat": float(probs.get("flat", 0.33)),
            },
            rationale=str(payload.get("rationale", "")),
            seat_id=str(payload.get("seat_id") or self.profile.seat_id),
            thesis=str(payload.get("thesis", "")),
            counterfactual=str(payload.get("counterfactual", "")),
            evidence_items=evidence_items,
            uncertainty_notes=[str(x) for x in payload.get("uncertainty_notes", [])][:8],
            self_critique=str(payload.get("self_critique", "")),
        )

    def _fallback(self, ctx: CouncilContext, reason: str) -> AgentOpinion:
        baseline_signal = _pick_baseline_signal(ctx)
        scenario = _scenario_from_signal(baseline_signal)
        return AgentOpinion(
            agent_id=self.agent_id,
            direction=baseline_signal,
            confidence=min(50.0, self.profile.max_confidence_cap),
            expected_move_pct=_expected_move_from_signal(baseline_signal),
            iv_view="neutral",
            evidence_score=45.0,
            citations=[],
            risk_flags=[f"fallback:{reason}"],
            recommended_params={},
            scenario_probs=scenario,
            rationale=f"Deterministic fallback due to {reason}.",
            seat_id=self.profile.seat_id,
            thesis="Fallback to quant priors due to provider unavailability.",
            counterfactual="If event-quality evidence improves, reevaluate direction.",
            evidence_items=[],
            uncertainty_notes=["provider degraded"],
            self_critique="No external evidence available in fallback.",
        )

    def _build_prompt(self, ctx: CouncilContext) -> str:
        base = (
            f"Seat: {self.profile.display_name} ({self.profile.domain})\n"
            f"Persona style: {self.profile.persona_style}\n"
            f"Timestamp IST: {ctx.timestamp_ist}\n"
            f"Underlying: {ctx.underlying}\n"
            f"Spot: {ctx.spot}\n"
            f"Model outputs: {ctx.model_outputs}\n"
            f"Model health: {ctx.model_health}\n"
            f"Microstructure: {ctx.microstructure}\n"
            f"Macro data: {ctx.macro_data}\n"
            f"Corporate events (latest): {ctx.corporate_events[:8]}\n"
            f"News events (latest): {ctx.news_events[:8]}\n"
            "You are part of a senate with strict anti-noise governance."
            " Provide direction and evidence with uncertainty honesty."
            " Directional calls need strong citations and executable context."
        )
        
        domain = self.profile.domain.lower()
        if "macro" in domain or "monetary" in domain or "policy" in domain:
            base += "\n\nDOMAIN FOCUS: Emphasize central bank policy, macro factors (VIX/PCR/FII flows), and broader geopolitical risks. Do not get bogged down in micro-level technicals."
        elif "corporate" in domain:
            base += "\n\nDOMAIN FOCUS: Focus strictly on corporate events—earnings, dividends, buybacks, and single-stock news sentiment. Evaluate post-event implied volatility crush potential."
        elif "micro" in domain or "volatility" in domain or "systematic" in domain:
            base += "\n\nDOMAIN FOCUS: Prioritize microstructure metrics (Spread, OI Ratio), VIX/IV relationships, Volatility Surface Arb opportunities, and quantitative signals. Be extremely precise and data-driven."
        elif "risk" in domain or "audit" in domain or "credit" in domain:
            base += "\n\nDOMAIN FOCUS: Focus on risk parity, tail risk, model conviction divergence, and downside protection. Strongly favor HOLD or neutral/hedged views when actionable edge is low or model health is degraded."
        elif "climate" in domain or "geopolitic" in domain:
            base += "\n\nDOMAIN FOCUS: Focus explicitly on macro supply-chain anomalies, weather/climate shocks, and real-world geopolitical flashpoints that traditional quant models miss."
        elif "challenge" in domain or "prosecutor" in domain:
            base += "\n\nDOMAIN FOCUS: You are the Red Team Prosecutor. Attack the consensus. Highlight confirmation bias, stale quotes, weak correlations, and what the other models/agents might be ignoring."
            
        return base


def build_default_agents(provider_map: Dict[str, Optional[LLMProvider]]) -> List[BaseAgent]:
    return build_senate_agents(provider_map)


def build_senate_agents(
    provider_map: Dict[str, Optional[LLMProvider]],
    senate_profiles_path: Optional[str] = None,
    routing_policy: Optional[ProviderRoutingPolicy] = None,
    seat_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[BaseAgent]:
    profiles = load_senate_profiles(senate_profiles_path)
    overrides = seat_overrides or {}
    agents: List[BaseAgent] = []

    safe_provider_map: Dict[str, LLMProvider] = {
        k: v for k, v in (provider_map or {}).items() if isinstance(v, LLMProvider)
    }

    for profile in profiles:
        ov = overrides.get(profile.seat_id, {})
        enabled = bool(ov.get("enabled", profile.enabled))
        if not enabled:
            continue
        if "provider_priority" in ov and isinstance(ov.get("provider_priority"), list):
            profile.provider_priority = [str(x) for x in ov.get("provider_priority") if str(x).strip()]
        if "must_cite_min" in ov:
            try:
                profile.must_cite_min = int(ov.get("must_cite_min"))
            except Exception:
                pass

        agents.append(
            PersonaAgent(
                agent_id=profile.seat_id,
                role=profile.display_name,
                profile=profile,
                provider_map=safe_provider_map,
                routing_policy=routing_policy,
                temperature=float(profile.temperature_cap),
            )
        )

    return agents


def load_senate_profiles(path: Optional[str] = None) -> List[AgentProfile]:
    profile_path = path or DEFAULT_PROFILE_PATH
    records = _load_profile_records(profile_path)
    if not records:
        records = _default_profile_records()

    profiles: List[AgentProfile] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        profiles.append(
            AgentProfile(
                seat_id=str(rec.get("seat_id", "")).strip(),
                display_name=str(rec.get("display_name", "")).strip() or str(rec.get("seat_id", "")).strip(),
                domain=str(rec.get("domain", "market")).strip(),
                persona_style=str(rec.get("persona_style", "hybrid")).strip() or "hybrid",  # type: ignore[arg-type]
                provider_priority=[str(x) for x in (rec.get("provider_priority") or ["gemini", "openai", "perplexity", "anthropic"])],
                temperature_cap=float(rec.get("temperature_cap", 0.2)),
                must_cite_min=int(rec.get("must_cite_min", 1)),
                max_confidence_cap=float(rec.get("max_confidence_cap", 85.0)),
                seat_weight=float(rec.get("seat_weight", 1.0)),
                reliability_prior=float(rec.get("reliability_prior", 0.60)),
                voting=bool(rec.get("voting", True)),
                enabled=bool(rec.get("enabled", True)),
            )
        )

    out = [p for p in profiles if p.seat_id]
    if not out:
        out = [
            AgentProfile(
                seat_id="risk_manager_cro",
                display_name="Risk Manager CRO",
                domain="risk",
                provider_priority=["openai", "gemini", "perplexity", "anthropic"],
            )
        ]
    return out


def _load_profile_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    text = ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception:
        return []

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except Exception:
        pass

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except Exception:
        pass

    return []


def _default_profile_records() -> List[Dict[str, Any]]:
    return [
        {"seat_id": "monetary_policy_chair", "display_name": "Monetary Policy Chair", "domain": "monetary policy", "provider_priority": ["perplexity", "gemini", "openai", "anthropic"], "seat_weight": 1.2, "must_cite_min": 2},
        {"seat_id": "global_macro_cio", "display_name": "Global Macro CIO", "domain": "macro", "provider_priority": ["gemini", "openai", "perplexity", "anthropic"], "seat_weight": 1.1, "must_cite_min": 2},
        {"seat_id": "india_policy_political_risk", "display_name": "India Policy & Political Risk", "domain": "policy", "provider_priority": ["perplexity", "gemini", "openai", "anthropic"], "seat_weight": 1.1, "must_cite_min": 2},
        {"seat_id": "corporate_actions_earnings", "display_name": "Corporate Actions & Earnings", "domain": "corporate events", "provider_priority": ["perplexity", "openai", "gemini", "anthropic"], "seat_weight": 1.0, "must_cite_min": 2},
        {"seat_id": "credit_liquidity_stress", "display_name": "Credit & Liquidity Stress", "domain": "credit/liquidity", "provider_priority": ["openai", "perplexity", "gemini", "anthropic"], "seat_weight": 1.0, "must_cite_min": 1},
        {"seat_id": "derivatives_microstructure", "display_name": "Derivatives Microstructure", "domain": "microstructure", "provider_priority": ["openai", "gemini", "perplexity", "anthropic"], "seat_weight": 1.2, "must_cite_min": 1},
        {"seat_id": "volatility_surface_arb", "display_name": "Volatility Surface Arb", "domain": "volatility", "provider_priority": ["gemini", "openai", "perplexity", "anthropic"], "seat_weight": 1.1, "must_cite_min": 1},
        {"seat_id": "systematic_statistical_pm", "display_name": "Systematic Statistical PM", "domain": "systematic", "provider_priority": ["openai", "gemini", "perplexity", "anthropic"], "seat_weight": 1.1, "must_cite_min": 1},
        {"seat_id": "geopolitical_supply_chain", "display_name": "Geopolitical Supply Chain", "domain": "geopolitics", "provider_priority": ["perplexity", "gemini", "openai", "anthropic"], "seat_weight": 1.0, "must_cite_min": 2},
        {"seat_id": "climate_weather_commodities", "display_name": "Climate Weather Commodities", "domain": "climate/weather", "provider_priority": ["perplexity", "openai", "gemini", "anthropic"], "seat_weight": 0.9, "must_cite_min": 1},
        {"seat_id": "risk_manager_cro", "display_name": "Risk Manager CRO", "domain": "risk", "provider_priority": ["openai", "gemini", "perplexity", "anthropic"], "seat_weight": 1.3, "must_cite_min": 1},
        {"seat_id": "quant_model_auditor", "display_name": "Quant Model Auditor", "domain": "model audit", "provider_priority": ["gemini", "openai", "perplexity", "anthropic"], "seat_weight": 1.3, "must_cite_min": 1},
        {"seat_id": "red_team_prosecutor", "display_name": "Red Team Prosecutor", "domain": "challenge", "provider_priority": ["anthropic", "openai", "gemini", "perplexity"], "seat_weight": 0.0, "must_cite_min": 1, "voting": False},
        {"seat_id": "arbiter_chair", "display_name": "Arbiter Chair", "domain": "arbitration", "provider_priority": ["openai", "anthropic", "gemini", "perplexity"], "seat_weight": 0.0, "must_cite_min": 1, "voting": False},
    ]


def _is_high_impact_context(ctx: CouncilContext) -> bool:
    if any(str(ev.get("impact", "")).lower() == "high" for ev in (ctx.news_events or [])):
        return True
    if any(str(ev.get("impact", "")).lower() == "high" for ev in (ctx.corporate_events or [])):
        return True
    for finding in ctx.model_health.get("findings", []):
        if str(finding.get("severity", "")).lower() in {"high", "critical"}:
            return True
    return False


def _build_evidence_items(raw_items: Any, citations: List[str]) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    if isinstance(raw_items, list):
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            items.append(
                EvidenceItem(
                    source=str(raw.get("source", "unknown")),
                    timestamp=str(raw.get("timestamp", "")),
                    url=str(raw.get("url", "")),
                    relevance=max(0.0, min(1.0, _safe_float(raw.get("relevance"), 0.5))),
                    staleness_seconds=max(0.0, _safe_float(raw.get("staleness_seconds"), 0.0)),
                    quality_score=max(0.0, min(1.0, _safe_float(raw.get("quality_score"), 0.5))),
                )
            )
    if not items and citations:
        for url in citations[:4]:
            items.append(EvidenceItem(source="citation", url=str(url), relevance=0.5, quality_score=0.5))
    return items[:8]


def _norm_direction(value: Any) -> str:
    txt = str(value or "HOLD").upper()
    if "BUY" in txt:
        return "BUY"
    if "SELL" in txt:
        return "SELL"
    return "HOLD"


def _norm_iv_view(value: Any) -> str:
    txt = str(value or "neutral").lower().strip()
    if txt.startswith("exp"):
        return "expand"
    if txt.startswith("con"):
        return "contract"
    return "neutral"


def _pick_baseline_signal(ctx: CouncilContext) -> str:
    outs = ctx.model_outputs or {}
    weighted = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
    for data in outs.values():
        sig = _norm_direction(data.get("signal"))
        conf = float(data.get("confidence", 40.0))
        weighted[sig] += max(0.0, conf)
    best = max(weighted, key=lambda k: weighted[k])
    if weighted[best] <= 0:
        return "HOLD"
    return best


def _expected_move_from_signal(signal: str) -> float:
    if signal == "BUY":
        return 0.45
    if signal == "SELL":
        return -0.45
    return 0.0


def _scenario_from_signal(signal: str) -> Dict[str, float]:
    if signal == "BUY":
        return {"up": 0.56, "down": 0.22, "flat": 0.22}
    if signal == "SELL":
        return {"up": 0.22, "down": 0.56, "flat": 0.22}
    return {"up": 0.33, "down": 0.33, "flat": 0.34}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)
