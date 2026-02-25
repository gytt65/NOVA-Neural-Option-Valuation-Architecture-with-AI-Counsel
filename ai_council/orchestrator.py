from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

from .agents import BaseAgent
from .event_ingestion import compute_event_shock, extract_watchlist_terms
from .memory_store import CouncilMemoryStore
from .types import AgentOpinion, CouncilContext, CouncilVerdict, CritiqueReport, ProbabilisticForecast


@dataclass
class CouncilConfig:
    buy_threshold: float = 0.58
    sell_threshold: float = 0.58
    hold_if_disagreement_gt: float = 45.0
    hold_if_evidence_lt: float = 60.0
    strict_mode: bool = True
    min_citations: int = 1
    max_source_age_seconds: float = 7200.0
    hard_veto: bool = True
    cost_mode: str = "mostly_free"
    strict_hold_bias_on_provider_failure: bool = True


class AICouncilOrchestrator:
    """Runs senate rounds and emits a deterministic advisory verdict."""

    HORIZONS = ("intraday_60m", "today_close", "next_day", "next_week")
    NON_VOTING_SEATS = {"red_team_prosecutor", "arbiter_chair"}
    SEAT_WEIGHTS = {
        "monetary_policy_chair": 1.2,
        "global_macro_cio": 1.1,
        "india_policy_political_risk": 1.1,
        "corporate_actions_earnings": 1.0,
        "credit_liquidity_stress": 1.0,
        "derivatives_microstructure": 1.2,
        "volatility_surface_arb": 1.1,
        "systematic_statistical_pm": 1.1,
        "geopolitical_supply_chain": 1.0,
        "climate_weather_commodities": 0.9,
        "risk_manager_cro": 1.3,
        "quant_model_auditor": 1.3,
        "red_team_prosecutor": 0.0,
        "arbiter_chair": 0.0,
    }

    def __init__(self, config: Optional[CouncilConfig] = None, memory_store: Optional[CouncilMemoryStore] = None):
        self.config = config or CouncilConfig()
        self.memory_store = memory_store

    def run_cycle(self, context: CouncilContext, agents: List[BaseAgent]) -> CouncilVerdict:
        opinions: List[AgentOpinion] = [agent.opinion(context) for agent in agents]
        critiques: List[CritiqueReport] = self._run_critiques(agents, opinions, context)

        voting_opinions = [op for op in opinions if self._seat_weight(op.seat_id or op.agent_id) > 0.0]
        disagreement_matrix, disagreement_score = self._compute_disagreement(voting_opinions)
        contradiction_tags = self._contradiction_tags(voting_opinions)
        avg_evidence = mean([max(0.0, min(100.0, op.evidence_score)) for op in voting_opinions]) if voting_opinions else 0.0

        consensus_probs = self._blend_probabilities(context, voting_opinions)
        forecasts = self._build_forecasts(consensus_probs, context)

        consensus_score = self._consensus_score(voting_opinions, critiques)
        confidence = self._confidence(consensus_score, disagreement_score, avg_evidence)

        seat_votes = self._seat_votes(voting_opinions)
        veto_flags, quality_gate_status, operator_checklist = self._quality_gates(
            context,
            voting_opinions,
            disagreement_score,
            avg_evidence,
            contradiction_tags,
        )

        actionability = len(veto_flags) == 0
        actionability_reasons = list(dict.fromkeys(veto_flags))

        signal = self._chair_synthesis(consensus_probs, actionability, veto_flags)
        parameter_recos = self._parameter_recommendations(context, actionability_reasons, disagreement_score, avg_evidence)
        priority = self._priority(signal, confidence, actionability_reasons)

        verdict = CouncilVerdict(
            final_signal=signal,
            confidence=confidence,
            consensus_score=consensus_score,
            disagreement_score=disagreement_score,
            actionability=actionability,
            actionability_reasons=actionability_reasons,
            probabilistic_forecast=forecasts,
            parameter_recommendations=parameter_recos,
            model_health_findings=list(context.model_health.get("findings", [])),
            watchlist=extract_watchlist_terms(context.news_events, context.corporate_events),
            telegram_priority=priority,
            disagreement_matrix=disagreement_matrix,
            agent_opinions=opinions,
            critiques=critiques,
            seat_votes=seat_votes,
            veto_flags=veto_flags,
            quality_gate_status=quality_gate_status,
            operator_checklist=operator_checklist,
            metadata={
                "avg_evidence": avg_evidence,
                "providers_active": context.model_health.get("providers_active", []),
                "providers_expected": context.model_health.get("providers_expected", []),
                "provider_degraded": bool(context.model_health.get("provider_degraded", False)),
                "timestamp_ist": context.timestamp_ist,
                "underlying": context.underlying,
                "spot": float(context.spot or 0.0),
                "cost_mode": self.config.cost_mode,
                "contradiction_tags": contradiction_tags,
            },
        )

        if self.memory_store is not None:
            previous_cycle = None
            if hasattr(self.memory_store, "latest_cycle"):
                previous_cycle = self.memory_store.latest_cycle()
            cycle_id = self.memory_store.save_cycle(context.underlying, verdict.to_dict())
            self._update_reliability(previous_cycle, context)
            self._update_quality_metrics(cycle_id, voting_opinions, context)

        return verdict

    def _run_critiques(self, agents: List[BaseAgent], opinions: List[AgentOpinion], ctx: CouncilContext) -> List[CritiqueReport]:
        critiques: List[CritiqueReport] = []
        if not opinions:
            return critiques
        n = len(opinions)
        op_by_agent = {op.agent_id: op for op in opinions}

        for i, agent in enumerate(agents):
            own = op_by_agent.get(agent.agent_id)
            if own is None:
                continue
            if "red_team" in agent.agent_id or "prosecutor" in agent.agent_id:
                for target in opinions:
                    if target.agent_id == own.agent_id:
                        continue
                    critiques.append(agent.critique(own, target, ctx))
                continue
            target = opinions[(i + 1) % n]
            if target.agent_id == own.agent_id and n > 1:
                target = opinions[(i + 2) % n]
            if target.agent_id != own.agent_id:
                critiques.append(agent.critique(own, target, ctx))
        return critiques

    def _seat_weight(self, seat_id: str) -> float:
        return float(self.SEAT_WEIGHTS.get(str(seat_id), 1.0))

    def _seat_votes(self, opinions: List[AgentOpinion]) -> List[Dict[str, Any]]:
        votes: List[Dict[str, Any]] = []
        for op in opinions:
            seat_id = op.seat_id or op.agent_id
            weight = self._seat_weight(seat_id)
            votes.append(
                {
                    "seat_id": seat_id,
                    "direction": op.direction,
                    "confidence": round(float(op.confidence), 2),
                    "evidence_score": round(float(op.evidence_score), 2),
                    "weight": round(weight, 3),
                    "weighted_vote": round(weight * (op.confidence / 100.0), 4),
                    "citations_count": len(op.citations),
                    "rationale": op.rationale,
                }
            )
        return votes

    def _compute_disagreement(self, opinions: List[AgentOpinion]) -> Tuple[List[Dict[str, Any]], float]:
        matrix: List[Dict[str, Any]] = []
        if len(opinions) <= 1:
            return matrix, 0.0

        total = 0.0
        count = 0
        for i, left in enumerate(opinions):
            for right in opinions[i + 1 :]:
                direction_conflict = 1.0 if left.direction != right.direction else 0.0
                prob_gap = abs(left.scenario_probs.get("up", 0.33) - right.scenario_probs.get("up", 0.33))
                conf_gap = abs(left.confidence - right.confidence) / 100.0
                pair_score = (0.6 * direction_conflict + 0.3 * prob_gap + 0.1 * conf_gap) * 100.0
                matrix.append(
                    {
                        "left": left.seat_id or left.agent_id,
                        "right": right.seat_id or right.agent_id,
                        "direction_conflict": direction_conflict,
                        "prob_gap": round(prob_gap, 4),
                        "confidence_gap": round(conf_gap, 4),
                        "score": round(pair_score, 2),
                    }
                )
                total += pair_score
                count += 1
        return matrix, round(total / max(1, count), 2)

    def _contradiction_tags(self, opinions: List[AgentOpinion]) -> List[str]:
        tags: List[str] = []
        buy_high = [x for x in opinions if x.direction == "BUY" and x.confidence >= 65 and x.evidence_score >= 60]
        sell_high = [x for x in opinions if x.direction == "SELL" and x.confidence >= 65 and x.evidence_score >= 60]
        if buy_high and sell_high:
            tags.append("HIGH_CONFIDENCE_DIRECTION_CONFLICT")

        thin_citation_directional = [x for x in opinions if x.direction != "HOLD" and len(x.citations) < self.config.min_citations]
        if thin_citation_directional:
            tags.append("DIRECTIONAL_WITH_THIN_CITATIONS")

        return tags

    def _consensus_score(self, opinions: List[AgentOpinion], critiques: List[CritiqueReport]) -> float:
        if not opinions:
            return 0.0
        weighted_scores: List[float] = []
        for op in opinions:
            w = self._seat_weight(op.seat_id or op.agent_id)
            score = 0.55 * op.confidence + 0.45 * op.evidence_score
            weighted_scores.append(w * score)
        base = sum(weighted_scores) / max(1e-9, sum(self._seat_weight(op.seat_id or op.agent_id) for op in opinions))
        critique_penalty = mean([cr.severity for cr in critiques]) if critiques else 0.0
        return round(max(0.0, min(100.0, base - 0.25 * critique_penalty)), 2)

    def _confidence(self, consensus_score: float, disagreement_score: float, avg_evidence: float) -> float:
        conf = 0.45 * consensus_score + 0.35 * avg_evidence + 0.20 * max(0.0, 100.0 - disagreement_score)
        return round(max(1.0, min(99.0, conf)), 2)

    def _model_priors(self, context: CouncilContext) -> Dict[str, float]:
        weights = {"up": 0.0, "down": 0.0, "flat": 0.0}
        outs = context.model_outputs or {}
        for data in outs.values():
            sig = str(data.get("signal") or "HOLD").upper()
            conf = max(0.0, min(100.0, float(data.get("confidence") or 0.0))) / 100.0
            if "BUY" in sig:
                weights["up"] += 0.50 * conf
                weights["down"] += 0.20 * conf
                weights["flat"] += 0.30 * conf
            elif "SELL" in sig:
                weights["up"] += 0.20 * conf
                weights["down"] += 0.50 * conf
                weights["flat"] += 0.30 * conf
            else:
                weights["up"] += 0.30 * conf
                weights["down"] += 0.30 * conf
                weights["flat"] += 0.40 * conf
        total = weights["up"] + weights["down"] + weights["flat"]
        if total <= 1e-9:
            return {"up": 0.33, "down": 0.33, "flat": 0.34}
        return {k: v / total for k, v in weights.items()}

    def _agent_probs(self, opinions: List[AgentOpinion]) -> Dict[str, float]:
        if not opinions:
            return {"up": 0.33, "down": 0.33, "flat": 0.34}
        w_up = 0.0
        w_down = 0.0
        w_flat = 0.0
        total_w = 0.0
        for op in opinions:
            seat_weight = self._seat_weight(op.seat_id or op.agent_id)
            w = max(0.1, seat_weight * (op.confidence * 0.6 + op.evidence_score * 0.4) / 100.0)
            p = op.scenario_probs or {}
            w_up += w * float(p.get("up", 0.33))
            w_down += w * float(p.get("down", 0.33))
            w_flat += w * float(p.get("flat", 0.34))
            total_w += w
        if total_w <= 1e-9:
            return {"up": 0.33, "down": 0.33, "flat": 0.34}
        out = {"up": w_up / total_w, "down": w_down / total_w, "flat": w_flat / total_w}
        return _norm_probs(out)

    def _blend_probabilities(self, context: CouncilContext, opinions: List[AgentOpinion]) -> Dict[str, float]:
        model_probs = self._model_priors(context)
        agent_probs = self._agent_probs(opinions)
        shock = compute_event_shock(context.news_events, context.corporate_events)

        up = 0.55 * model_probs["up"] + 0.35 * agent_probs["up"]
        down = 0.55 * model_probs["down"] + 0.35 * agent_probs["down"]
        flat = 0.55 * model_probs["flat"] + 0.35 * agent_probs["flat"]

        up += 0.10 * max(0.0, shock.direction_bias)
        down += 0.10 * max(0.0, -shock.direction_bias)
        flat += 0.10 * (1.0 - abs(shock.direction_bias) / 2.5)

        blended = _norm_probs({"up": up, "down": down, "flat": flat})
        return self._apply_calibration_correction(blended)

    def _build_forecasts(self, probs: Dict[str, float], context: CouncilContext) -> List[ProbabilisticForecast]:
        shock = compute_event_shock(context.news_events, context.corporate_events)
        base_sigma = max(0.3, min(3.0, 0.75 + 1.5 * shock.vol_bias))
        flat_penalty = 1.0 - probs["flat"]
        out: List[ProbabilisticForecast] = []

        horizon_scale = {
            "intraday_60m": 0.45,
            "today_close": 0.80,
            "next_day": 1.10,
            "next_week": 2.30,
        }

        for hz in self.HORIZONS:
            scale = horizon_scale[hz]
            mu = (probs["up"] - probs["down"]) * 1.2 * scale
            sigma = base_sigma * scale * (0.8 + 0.4 * flat_penalty)
            q50 = mu
            q25 = mu - 0.675 * sigma
            q75 = mu + 0.675 * sigma
            q05 = mu - 1.645 * sigma
            q95 = mu + 1.645 * sigma
            iv_expand = min(0.95, max(0.05, 0.35 + 0.45 * shock.vol_bias + 0.10 * probs["down"]))
            iv_contract = min(0.95, max(0.05, 1.0 - iv_expand))
            p_gap_up = min(0.8, max(0.02, 0.12 + 0.30 * probs["up"] * (1.0 if hz != "intraday_60m" else 0.3)))
            p_gap_down = min(0.8, max(0.02, 0.12 + 0.30 * probs["down"] * (1.0 if hz != "intraday_60m" else 0.3)))

            calibration_score = 0.6
            if self.memory_store is not None:
                rel = self.memory_store.get_reliability(hz)
                calibration_score = float(rel.get("calibration_score", 0.6))

            out.append(
                ProbabilisticForecast(
                    horizon=hz,  # type: ignore[arg-type]
                    p_up=round(probs["up"], 4),
                    p_down=round(probs["down"], 4),
                    p_flat=round(probs["flat"], 4),
                    return_mu=round(mu, 4),
                    return_sigma=round(sigma, 4),
                    q05=round(q05, 4),
                    q25=round(q25, 4),
                    q50=round(q50, 4),
                    q75=round(q75, 4),
                    q95=round(q95, 4),
                    p_iv_expand=round(iv_expand, 4),
                    p_iv_contract=round(iv_contract, 4),
                    p_gap_up=round(p_gap_up, 4),
                    p_gap_down=round(p_gap_down, 4),
                    calibration_score=round(calibration_score, 4),
                )
            )

        return out

    def _apply_calibration_correction(self, probs: Dict[str, float]) -> Dict[str, float]:
        if self.memory_store is None:
            return probs
        rel = self.memory_store.get_reliability("next_day")
        calibration = max(0.0, min(1.0, _safe_float(rel.get("calibration_score"), 0.6)))
        neutral = {"up": 0.33, "down": 0.33, "flat": 0.34}
        corrected = {
            "up": neutral["up"] + calibration * (probs["up"] - neutral["up"]),
            "down": neutral["down"] + calibration * (probs["down"] - neutral["down"]),
            "flat": neutral["flat"] + calibration * (probs["flat"] - neutral["flat"]),
        }
        return _norm_probs(corrected)

    def _quality_gates(
        self,
        context: CouncilContext,
        opinions: List[AgentOpinion],
        disagreement: float,
        avg_evidence: float,
        contradiction_tags: List[str],
    ) -> Tuple[List[str], Dict[str, Any], List[str]]:
        veto_flags: List[str] = []
        checklist: List[str] = []

        if disagreement > self.config.hold_if_disagreement_gt:
            veto_flags.append(f"DISAGREEMENT_{disagreement:.1f}")
            checklist.append("Check seat-level conflicts before directional trades.")

        if avg_evidence < self.config.hold_if_evidence_lt:
            veto_flags.append(f"LOW_EVIDENCE_{avg_evidence:.1f}")
            checklist.append("Require stronger evidence/citations before action.")

        directional = [op for op in opinions if op.direction != "HOLD"]
        if directional:
            min_citations = min((len(op.citations) for op in directional), default=0)
            if min_citations < self.config.min_citations:
                veto_flags.append("MIN_CITATIONS_NOT_MET")
                checklist.append("Wait for corroborating sources with timestamps.")

            stale_breach = False
            stale_unverified = False
            for op in directional:
                if not op.evidence_items and op.citations:
                    stale_unverified = True
                    break
                for ev in op.evidence_items:
                    if ev.staleness_seconds > self.config.max_source_age_seconds:
                        stale_breach = True
                        break
                if stale_breach:
                    break
            if stale_breach:
                veto_flags.append("SOURCE_STALENESS_BREACH")
                checklist.append("Refresh data and rerun senate cycle.")
            elif stale_unverified:
                veto_flags.append("SOURCE_STALENESS_UNVERIFIED")
                checklist.append("Require timestamped evidence metadata before directional action.")

        micro = context.microstructure or {}
        spread = _safe_float(micro.get("spread_pct"), 0.0)
        voi = _safe_float(micro.get("volume_oi_ratio"), 1.0)
        if spread > 2.0 or voi < 0.03:
            veto_flags.append("MICROSTRUCTURE_NON_EXECUTABLE")
            checklist.append("Do not act in illiquid/wide-spread contract.")

        high_conflict = any(
            str(f.get("type", "")).upper() == "MODEL_SIGNAL_CONFLICT" and str(f.get("severity", "")).lower() in {"high", "critical"}
            for f in context.model_health.get("findings", [])
        )
        if high_conflict:
            veto_flags.append("MODEL_CONFLICT_HIGH_SEVERITY")
            checklist.append("Require model alignment before directional bias.")

        provider_degraded = bool(context.model_health.get("provider_degraded", False))
        if provider_degraded and self.config.strict_hold_bias_on_provider_failure:
            veto_flags.append("PROVIDER_DEGRADED_STRICT_HOLD_BIAS")
            checklist.append("Provider outage detected. Prefer HOLD until restoration.")

        for tag in contradiction_tags:
            veto_flags.append(f"CONTRADICTION_{tag}")
            checklist.append("Use dissent matrix; avoid forcing a directional call.")

        status = {
            "disagreement_ok": disagreement <= self.config.hold_if_disagreement_gt,
            "evidence_ok": avg_evidence >= self.config.hold_if_evidence_lt,
            "citations_ok": "MIN_CITATIONS_NOT_MET" not in veto_flags,
            "staleness_ok": (
                "SOURCE_STALENESS_BREACH" not in veto_flags
                and "SOURCE_STALENESS_UNVERIFIED" not in veto_flags
            ),
            "micro_ok": "MICROSTRUCTURE_NON_EXECUTABLE" not in veto_flags,
            "model_conflict_ok": "MODEL_CONFLICT_HIGH_SEVERITY" not in veto_flags,
            "provider_ok": "PROVIDER_DEGRADED_STRICT_HOLD_BIAS" not in veto_flags,
            "hard_veto": bool(self.config.hard_veto),
        }
        return list(dict.fromkeys(veto_flags)), status, list(dict.fromkeys(checklist))

    def _chair_synthesis(self, probs: Dict[str, float], actionability: bool, veto_flags: List[str]) -> str:
        if self.config.hard_veto and veto_flags:
            return "HOLD"
        if not actionability:
            return "HOLD"
        if probs["up"] >= self.config.buy_threshold:
            return "BUY"
        if probs["down"] >= self.config.sell_threshold:
            return "SELL"
        return "HOLD"

    def _priority(self, signal: str, confidence: float, reasons: List[str]) -> str:
        if any("MICROSTRUCTURE" in r or "SOURCE_STALENESS" in r for r in reasons):
            return "high"
        if signal in {"BUY", "SELL"} and confidence >= 75:
            return "high"
        if signal in {"BUY", "SELL"} and confidence >= 62:
            return "medium"
        return "low"

    def _parameter_recommendations(
        self,
        context: CouncilContext,
        reasons: List[str],
        disagreement_score: float,
        avg_evidence: float,
    ) -> List[Dict[str, Any]]:
        recos: List[Dict[str, Any]] = []
        if any("MICROSTRUCTURE_NON_EXECUTABLE" in r for r in reasons):
            recos.append(
                {
                    "parameter": "max_spread_pct",
                    "current": 2.0,
                    "suggested": 1.5,
                    "expected_impact": "Fewer non-executable false positives.",
                    "risk": "May miss some volatile opportunities.",
                    "revert_condition": "If missed-signal rate rises for 10 sessions.",
                }
            )
        if disagreement_score > self.config.hold_if_disagreement_gt:
            recos.append(
                {
                    "parameter": "consensus_min_score",
                    "current": 55,
                    "suggested": 65,
                    "expected_impact": "Cleaner outputs during contradiction-heavy windows.",
                    "risk": "Higher HOLD frequency.",
                    "revert_condition": "If directional hit-rate stable for 4 weeks.",
                }
            )
        if avg_evidence < self.config.hold_if_evidence_lt:
            recos.append(
                {
                    "parameter": "min_evidence_score",
                    "current": self.config.hold_if_evidence_lt,
                    "suggested": min(85, self.config.hold_if_evidence_lt + 10),
                    "expected_impact": "Improves evidence rigor for directional calls.",
                    "risk": "Lower trade cadence.",
                    "revert_condition": "If sample size becomes too low.",
                }
            )
        if any(str(f.get("type", "")).startswith("LOW_MODEL_CONFIDENCE") for f in context.model_health.get("findings", [])):
            recos.append(
                {
                    "parameter": "regime_sensitivity",
                    "current": "medium",
                    "suggested": "high",
                    "expected_impact": "Faster adaptation to unstable regimes.",
                    "risk": "Potential for regime flip noise.",
                    "revert_condition": "When disagreement stays <25 for 20 cycles.",
                }
            )
        return recos

    def _update_reliability(self, previous: Optional[Dict[str, Any]], context: CouncilContext) -> None:
        """Update calibration scores based on previous forecast vs actual spot move."""
        if self.memory_store is None:
            return
        try:
            if previous is None:
                return
            prev_spot = _safe_float(previous.get("metadata", {}).get("spot"), 0.0)
            if prev_spot <= 0:
                prev_spot = _safe_float(previous.get("spot"), 0.0)
            curr_spot = float(context.spot or 0.0)
            if prev_spot <= 0 or curr_spot <= 0:
                return
            actual_move_pct = (curr_spot - prev_spot) / prev_spot * 100.0
            if actual_move_pct > 0.15:
                actual = "up"
            elif actual_move_pct < -0.15:
                actual = "down"
            else:
                actual = "flat"

            for fc in previous.get("probabilistic_forecast", []):
                hz = str(fc.get("horizon", ""))
                if not hz:
                    continue
                p_actual = float(fc.get(f"p_{actual}", 0.33))
                # Brier score component: lower is better, 0=perfect
                brier = (1.0 - p_actual) ** 2
                old = self.memory_store.get_reliability(hz)
                old_count = int(old.get("sample_count", 0))
                old_brier = float(old.get("brier_score", 0.0))
                new_count = old_count + 1
                # Exponential moving average for Brier score
                alpha = min(1.0, 2.0 / (new_count + 1))
                new_brier = old_brier * (1.0 - alpha) + brier * alpha
                # calibration_score: 1.0 = perfect, 0.0 = worst
                new_calibration = max(0.1, min(1.0, 1.0 - new_brier))
                self.memory_store.upsert_reliability(hz, new_brier, new_count, new_calibration)
                if hasattr(self.memory_store, "save_verdict_outcome"):
                    prior_id = int(previous.get("_id", 0) or 0)
                    if prior_id > 0:
                        predicted = max(
                            ("up", "down", "flat"),
                            key=lambda label: float(fc.get(f"p_{label}", 0.0)),
                        )
                        self.memory_store.save_verdict_outcome(
                            cycle_id=prior_id,
                            horizon=hz,
                            realized_return=actual_move_pct,
                            realized_direction=str(actual).upper(),
                            outcome_hit=(predicted == actual),
                        )
        except Exception as exc:
            LOGGER.debug("Reliability update skipped: %s", exc)

    def _update_quality_metrics(self, cycle_id: int, opinions: List[AgentOpinion], context: CouncilContext) -> None:
        if self.memory_store is None:
            return
        if not hasattr(self.memory_store, "upsert_provider_reliability"):
            return

        provider_names = list(context.model_health.get("providers_active", []))
        provider_success = 0.5 if bool(context.model_health.get("provider_degraded", False)) else 1.0
        for provider in provider_names:
            self.memory_store.upsert_provider_reliability(
                provider_name=str(provider),
                sample_count=1,
                success_rate=provider_success,
                avg_latency_ms=0.0,
            )

        source_scores: Dict[str, List[Tuple[float, float]]] = {}
        for op in opinions:
            for ev in op.evidence_items:
                source = ev.source or "unknown"
                source_scores.setdefault(source, []).append((ev.quality_score, ev.staleness_seconds))
            self.memory_store.upsert_seat_reliability(
                seat_id=op.seat_id or op.agent_id,
                sample_count=1,
                hit_rate=0.5,
                calibration_score=max(0.0, min(1.0, op.evidence_score / 100.0)),
            )

        for source, vals in source_scores.items():
            mean_quality = mean([v[0] for v in vals])
            mean_stale = mean([v[1] for v in vals])
            self.memory_store.upsert_evidence_quality(
                source=source,
                sample_count=len(vals),
                avg_quality=mean_quality,
                avg_staleness=mean_stale,
            )


def _norm_probs(probs: Dict[str, float]) -> Dict[str, float]:
    up = max(0.0, float(probs.get("up", 0.0)))
    down = max(0.0, float(probs.get("down", 0.0)))
    flat = max(0.0, float(probs.get("flat", 0.0)))
    total = up + down + flat
    if total <= 1e-9:
        return {"up": 0.33, "down": 0.33, "flat": 0.34}
    return {"up": up / total, "down": down / total, "flat": flat / total}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)
