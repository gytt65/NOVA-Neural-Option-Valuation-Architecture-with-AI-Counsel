from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional


class CouncilMemoryStore:
    """SQLite-backed persistence for council cycles, reliability and senate quality."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS council_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    underlying TEXT,
                    verdict_json TEXT NOT NULL,
                    final_signal TEXT,
                    confidence REAL,
                    consensus_score REAL,
                    disagreement_score REAL,
                    actionability INTEGER,
                    telegram_priority TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reliability (
                    horizon TEXT PRIMARY KEY,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    brier_score REAL NOT NULL DEFAULT 0.0,
                    calibration_score REAL NOT NULL DEFAULT 0.5,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seat_reliability (
                    seat_id TEXT PRIMARY KEY,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    hit_rate REAL NOT NULL DEFAULT 0.0,
                    calibration_score REAL NOT NULL DEFAULT 0.5,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_reliability (
                    provider_name TEXT PRIMARY KEY,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    success_rate REAL NOT NULL DEFAULT 0.0,
                    avg_latency_ms REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_quality (
                    source TEXT PRIMARY KEY,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    avg_quality REAL NOT NULL DEFAULT 0.0,
                    avg_staleness REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verdict_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_id INTEGER,
                    horizon TEXT,
                    realized_return REAL,
                    realized_direction TEXT,
                    outcome_hit INTEGER,
                    scored_at TEXT,
                    FOREIGN KEY(cycle_id) REFERENCES council_cycles(id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cycles_ts ON council_cycles(ts_utc)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_cycle ON verdict_outcomes(cycle_id)")
            conn.commit()

    def save_cycle(self, underlying: str, verdict: Dict[str, Any]) -> int:
        ts = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO council_cycles (
                    ts_utc, underlying, verdict_json, final_signal, confidence,
                    consensus_score, disagreement_score, actionability, telegram_priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    underlying,
                    json.dumps(verdict, ensure_ascii=True),
                    str(verdict.get("final_signal", "HOLD")),
                    float(verdict.get("confidence", 0.0) or 0.0),
                    float(verdict.get("consensus_score", 0.0) or 0.0),
                    float(verdict.get("disagreement_score", 0.0) or 0.0),
                    1 if verdict.get("actionability") else 0,
                    str(verdict.get("telegram_priority", "low")),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def latest_cycle(self) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM council_cycles ORDER BY id DESC LIMIT 1").fetchone()
            if not row:
                return None
            payload = json.loads(row["verdict_json"])
            payload["_id"] = int(row["id"])
            payload["_ts_utc"] = row["ts_utc"]
            payload["_underlying"] = row["underlying"]
            return payload

    def recent_cycles(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM council_cycles ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                payload = json.loads(row["verdict_json"])
                payload["_id"] = int(row["id"])
                payload["_ts_utc"] = row["ts_utc"]
                payload["_underlying"] = row["underlying"]
                out.append(payload)
            return out

    def upsert_reliability(self, horizon: str, brier_score: float, sample_count: int, calibration_score: float) -> None:
        ts = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reliability (horizon, sample_count, brier_score, calibration_score, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(horizon) DO UPDATE SET
                    sample_count=excluded.sample_count,
                    brier_score=excluded.brier_score,
                    calibration_score=excluded.calibration_score,
                    updated_at=excluded.updated_at
                """,
                (horizon, int(sample_count), float(brier_score), float(calibration_score), ts),
            )
            conn.commit()

    def get_reliability(self, horizon: str) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM reliability WHERE horizon=?", (str(horizon),)).fetchone()
            if not row:
                return {
                    "horizon": horizon,
                    "sample_count": 0,
                    "brier_score": 0.0,
                    "calibration_score": 0.5,
                }
            return {
                "horizon": row["horizon"],
                "sample_count": int(row["sample_count"]),
                "brier_score": float(row["brier_score"]),
                "calibration_score": float(row["calibration_score"]),
                "updated_at": row["updated_at"],
            }

    def upsert_seat_reliability(self, seat_id: str, sample_count: int, hit_rate: float, calibration_score: float) -> None:
        ts = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO seat_reliability (seat_id, sample_count, hit_rate, calibration_score, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(seat_id) DO UPDATE SET
                    sample_count=seat_reliability.sample_count + excluded.sample_count,
                    hit_rate=(seat_reliability.hit_rate + excluded.hit_rate)/2.0,
                    calibration_score=(seat_reliability.calibration_score + excluded.calibration_score)/2.0,
                    updated_at=excluded.updated_at
                """,
                (str(seat_id), int(sample_count), float(hit_rate), float(calibration_score), ts),
            )
            conn.commit()

    def get_seat_reliability(self, seat_id: str) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM seat_reliability WHERE seat_id=?", (str(seat_id),)).fetchone()
            if not row:
                return {"seat_id": seat_id, "sample_count": 0, "hit_rate": 0.0, "calibration_score": 0.5}
            return {
                "seat_id": row["seat_id"],
                "sample_count": int(row["sample_count"]),
                "hit_rate": float(row["hit_rate"]),
                "calibration_score": float(row["calibration_score"]),
                "updated_at": row["updated_at"],
            }

    def upsert_provider_reliability(self, provider_name: str, sample_count: int, success_rate: float, avg_latency_ms: float) -> None:
        ts = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO provider_reliability (provider_name, sample_count, success_rate, avg_latency_ms, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(provider_name) DO UPDATE SET
                    sample_count=provider_reliability.sample_count + excluded.sample_count,
                    success_rate=(provider_reliability.success_rate + excluded.success_rate)/2.0,
                    avg_latency_ms=(provider_reliability.avg_latency_ms + excluded.avg_latency_ms)/2.0,
                    updated_at=excluded.updated_at
                """,
                (str(provider_name), int(sample_count), float(success_rate), float(avg_latency_ms), ts),
            )
            conn.commit()

    def get_provider_reliability(self, provider_name: str) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM provider_reliability WHERE provider_name=?", (str(provider_name),)).fetchone()
            if not row:
                return {"provider_name": provider_name, "sample_count": 0, "success_rate": 0.0, "avg_latency_ms": 0.0}
            return {
                "provider_name": row["provider_name"],
                "sample_count": int(row["sample_count"]),
                "success_rate": float(row["success_rate"]),
                "avg_latency_ms": float(row["avg_latency_ms"]),
                "updated_at": row["updated_at"],
            }

    def upsert_evidence_quality(self, source: str, sample_count: int, avg_quality: float, avg_staleness: float) -> None:
        ts = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evidence_quality (source, sample_count, avg_quality, avg_staleness, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    sample_count=evidence_quality.sample_count + excluded.sample_count,
                    avg_quality=(evidence_quality.avg_quality + excluded.avg_quality)/2.0,
                    avg_staleness=(evidence_quality.avg_staleness + excluded.avg_staleness)/2.0,
                    updated_at=excluded.updated_at
                """,
                (str(source), int(sample_count), float(avg_quality), float(avg_staleness), ts),
            )
            conn.commit()

    def get_evidence_quality(self, source: str) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM evidence_quality WHERE source=?", (str(source),)).fetchone()
            if not row:
                return {"source": source, "sample_count": 0, "avg_quality": 0.0, "avg_staleness": 0.0}
            return {
                "source": row["source"],
                "sample_count": int(row["sample_count"]),
                "avg_quality": float(row["avg_quality"]),
                "avg_staleness": float(row["avg_staleness"]),
                "updated_at": row["updated_at"],
            }

    def save_verdict_outcome(
        self,
        cycle_id: int,
        horizon: str,
        realized_return: float,
        realized_direction: str,
        outcome_hit: bool,
    ) -> int:
        ts = datetime.utcnow().isoformat()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO verdict_outcomes (cycle_id, horizon, realized_return, realized_direction, outcome_hit, scored_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    int(cycle_id),
                    str(horizon),
                    float(realized_return),
                    str(realized_direction),
                    1 if outcome_hit else 0,
                    ts,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
