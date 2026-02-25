from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agents import build_senate_agents
from .orchestrator import AICouncilOrchestrator
from .providers import ProviderRoutingPolicy
from .types import CouncilContext


@dataclass
class CouncilWorkerConfig:
    cycle_seconds: int = 300
    backoff_seconds: int = 15
    event_spot_shock_pct: float = 0.7
    event_vix_shock_abs: float = 1.5
    enabled: bool = False
    senate_profiles_path: Optional[str] = None
    cost_mode: str = "mostly_free"
    max_daily_budget_usd: float = 8.0
    seat_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class CouncilWorker:
    def __init__(self, orchestrator: AICouncilOrchestrator, provider_map: Dict[str, Any], config: Optional[CouncilWorkerConfig] = None):
        self.orchestrator = orchestrator
        self.provider_map = provider_map
        self.config = config or CouncilWorkerConfig()
        self._routing_policy = ProviderRoutingPolicy(
            cost_mode=str(self.config.cost_mode),
            max_daily_budget_usd=float(self.config.max_daily_budget_usd),
            strict_hold_bias_on_failure=True,
        )

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()

        self._context: Optional[CouncilContext] = None
        self._last_context: Optional[CouncilContext] = None
        self._last_run_ts = 0.0
        self._last_error: Optional[str] = None
        self._latest_verdict: Optional[Dict[str, Any]] = None
        self._recent_verdicts: List[Dict[str, Any]] = []
        self._error_count = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name="ai-council-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def update_context(self, context: CouncilContext) -> None:
        with self._lock:
            self._context = copy.deepcopy(context)

    def latest_verdict(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._latest_verdict)

    def recent_verdicts(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._recent_verdicts[-int(limit):])

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self.is_running(),
                "last_run_ts": self._last_run_ts,
                "last_error": self._last_error,
                "has_context": self._context is not None,
                "cycle_seconds": self.config.cycle_seconds,
                "recent_count": len(self._recent_verdicts),
                "cost_mode": self.config.cost_mode,
                "spent_today_usd": round(float(self._routing_policy.spent_today_usd), 6),
                "profile_path": self.config.senate_profiles_path,
            }

    def run_once_now(self) -> Optional[Dict[str, Any]]:
        ctx = None
        with self._lock:
            if self._context is not None:
                ctx = copy.deepcopy(self._context)
        if ctx is None:
            return None
        return self._run_cycle(ctx)

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                ctx = None
                with self._lock:
                    if self._context is not None:
                        ctx = copy.deepcopy(self._context)

                if ctx is None:
                    time.sleep(1.0)
                    continue

                now = time.time()
                market_open = bool((ctx.data_freshness or {}).get("market_open", True))
                due_timer = market_open and (now - self._last_run_ts) >= max(5, int(self.config.cycle_seconds))
                due_event = self._is_event_trigger(ctx)

                if due_timer or due_event:
                    self._run_cycle(ctx)

                time.sleep(1.0)
                self._error_count = 0
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                self._error_count = min(8, self._error_count + 1)
                delay = max(2, int(self.config.backoff_seconds)) * (2 ** (self._error_count - 1))
                time.sleep(min(300, delay))

    def _run_cycle(self, ctx: CouncilContext) -> Dict[str, Any]:
        routing = self._sync_routing_policy_config()
        agents = build_senate_agents(
            self.provider_map,
            senate_profiles_path=self.config.senate_profiles_path,
            routing_policy=routing,
            seat_overrides=self.config.seat_overrides,
        )
        verdict = self.orchestrator.run_cycle(ctx, agents)
        verdict_dict = verdict.to_dict()
        verdict_dict["_worker_ts"] = time.time()
        with self._lock:
            self._latest_verdict = verdict_dict
            self._recent_verdicts.append(verdict_dict)
            if len(self._recent_verdicts) > 200:
                self._recent_verdicts = self._recent_verdicts[-200:]
            self._last_context = ctx
            self._last_run_ts = time.time()
            self._last_error = None
        return verdict_dict

    def _sync_routing_policy_config(self) -> ProviderRoutingPolicy:
        self._routing_policy.cost_mode = str(self.config.cost_mode or "mostly_free")
        self._routing_policy.max_daily_budget_usd = float(self.config.max_daily_budget_usd)
        self._routing_policy.strict_hold_bias_on_failure = True
        return self._routing_policy

    def _is_event_trigger(self, ctx: CouncilContext) -> bool:
        prev = None
        with self._lock:
            if self._last_context is not None:
                prev = copy.deepcopy(self._last_context)
        if prev is None:
            return False

        prev_spot = float(prev.spot or 0.0)
        curr_spot = float(ctx.spot or 0.0)
        if prev_spot > 0 and curr_spot > 0:
            shock_pct = abs(curr_spot - prev_spot) / prev_spot * 100.0
            if shock_pct >= self.config.event_spot_shock_pct:
                return True

        prev_vix = float((prev.macro_data or {}).get("india_vix") or 0.0)
        curr_vix = float((ctx.macro_data or {}).get("india_vix") or 0.0)
        if prev_vix > 0 and curr_vix > 0 and abs(curr_vix - prev_vix) >= self.config.event_vix_shock_abs:
            return True

        prev_signal = _dominant_signal(prev.model_outputs)
        curr_signal = _dominant_signal(ctx.model_outputs)
        if prev_signal and curr_signal and prev_signal != curr_signal:
            return True

        prev_news_high = _high_impact_news_count(prev.news_events)
        curr_news_high = _high_impact_news_count(ctx.news_events)
        if curr_news_high > prev_news_high:
            return True

        return False


def _dominant_signal(outputs: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if not outputs:
        return None
    scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
    for data in outputs.values():
        sig = str(data.get("signal") or "HOLD").upper()
        conf = float(data.get("confidence") or 0.0)
        if "BUY" in sig:
            scores["BUY"] += conf
        elif "SELL" in sig:
            scores["SELL"] += conf
        else:
            scores["HOLD"] += conf
    return max(scores, key=lambda k: scores[k])


def _high_impact_news_count(events: Optional[List[Dict[str, Any]]]) -> int:
    count = 0
    for ev in events or []:
        if str(ev.get("impact", "")).lower() == "high":
            count += 1
    return count


_WORKER_REGISTRY: Dict[str, CouncilWorker] = {}
_REGISTRY_LOCK = threading.Lock()


def get_worker(key: str = "default") -> Optional[CouncilWorker]:
    with _REGISTRY_LOCK:
        return _WORKER_REGISTRY.get(key)


def set_worker(worker: CouncilWorker, key: str = "default") -> None:
    with _REGISTRY_LOCK:
        _WORKER_REGISTRY[key] = worker


def stop_worker(key: str = "default") -> None:
    with _REGISTRY_LOCK:
        wrk = _WORKER_REGISTRY.get(key)
    if wrk is not None:
        wrk.stop()
