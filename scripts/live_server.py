"""Local live dashboard server. Not a claude.ai artifact -- runs on your own
machine, because a genuinely live feed needs a backend that can poll the
database/OpenSky continuously, and a published web page's CSP blocks exactly
that kind of outbound call.

What it does:
- Every POLL_SECONDS, queries track_state for rows newer than the last poll
  (the `ingestion` Docker service is already writing these continuously --
  this does not re-poll OpenSky itself, it rides on that existing pipeline).
- Feeds each new row through the SAME real components the rest of this
  project uses: a persistent KalmanTrackManager (one filter per icao24,
  carried across polls -- this is what makes it "live" rather than
  reprocessing history each time), the trained IsolationForestDetector,
  MLATSimulator/RadarSimulator's real-traffic check() path, a per-track
  IdentityTracker, and an EvidenceFusion model fit once at startup on recent
  real history.
- Broadcasts every processed update over a WebSocket to any connected
  browser tab (scripts/live_client.html).
- POST /api/inject_attack runs one of the real attack generators (the exact
  functions in absproj/attacks/) against a real recent substrate track at a
  requested severity, evaluates it with evaluate_scenario() -- the same
  function the Phase 7 benchmark uses -- and streams the real per-point
  result back over the same WebSocket for the client to animate. This is a
  genuine on-demand pipeline run, not a canned response.

Run: .venv/Scripts/python.exe scripts/live_server.py
Then open http://localhost:8000 in a browser.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import sys
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from absproj.attacks.injector import generate_variants  # noqa: E402
from absproj.attacks.substrate import build_substrate_pool  # noqa: E402
from absproj.attacks.types import AttackClass  # noqa: E402
from absproj.config import AppConfig, SeverityRange, get_config  # noqa: E402
from absproj.evaluation.fusion import EvidenceFusion, FusionRow  # noqa: E402
from absproj.evaluation.identity_check import IdentityTracker  # noqa: E402
from absproj.evaluation.pipeline import evaluate_scenario  # noqa: E402
from absproj.evaluation.rule_based import check_rule_based  # noqa: E402
from absproj.logging_setup import configure_logging  # noqa: E402
from absproj.ml.features import build_feature_frame, feature_matrix  # noqa: E402
from absproj.ml.isolation_forest import IsolationForestDetector  # noqa: E402
from absproj.storage import repository  # noqa: E402
from absproj.storage.db import get_connection  # noqa: E402
from absproj.tracking.track_manager import KalmanTrackManager  # noqa: E402
from absproj.verification.mlat import MLATSimulator  # noqa: E402
from absproj.verification.radar import RadarSimulator  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "isolation_forest.joblib"
CLIENT_PATH = Path(__file__).resolve().parent / "live_client.html"

POLL_SECONDS = 5.0
FUSION_FIT_LOOKBACK_MINUTES = 90
FUSION_FIT_VARIANTS_PER_CLASS = 5


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


class PipelineState:
    """All the persistent, cross-poll real pipeline state, mirroring what
    KalmanTrackManager already does internally for the filter itself --
    the rest (identity, ML rolling window, last-broadcast-for-rule-based)
    needs the same treatment here since a live server processes one small
    batch of new rows per poll rather than a full history at once."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.kalman = KalmanTrackManager(config.kalman)
        self.identity: dict[str, IdentityTracker] = defaultdict(IdentityTracker)
        self.last_broadcast: dict[str, object] = {}
        self.ml_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=config.ml.rolling_window))

        self.ml_detector = IsolationForestDetector(config.ml.isolation_forest)
        self.ml_detector.load(MODEL_PATH)

        origin_lat = (config.opensky.bbox.lamin + config.opensky.bbox.lamax) / 2.0
        origin_lon = (config.opensky.bbox.lomin + config.opensky.bbox.lomax) / 2.0
        self.mlat_sim = MLATSimulator(config.mlat, origin_lat, origin_lon)
        self.radar_sim = RadarSimulator(config.radar)

        self.fusion: Optional[EvidenceFusion] = None
        self.last_seen_time: Optional[datetime] = None
        self.callsigns: dict[str, str] = {}

    def process_row(self, sv) -> dict:
        icao24 = sv.icao24
        rule_flag = check_rule_based(self.last_broadcast.get(icao24), sv, self.config.rule_based)
        self.last_broadcast[icao24] = sv

        identity_flag = self.identity[icao24].check(sv)

        mlat_res = self.mlat_sim.check(sv)
        radar_res = self.radar_sim.check(sv)

        record = self.kalman.process(sv)

        nis = ml_score = fused_score = None
        nis_flag = ml_flag = fused_flag = False
        if record is not None:
            nis = record.nis
            nis_flag = record.is_anomalous
            self.ml_history[icao24].append({
                "orig_index": 0, "time": record.time, "icao24": icao24,
                "category": record.category.value, "dt_seconds": record.dt_seconds,
                "innovation_x": record.innovation[0], "innovation_y": record.innovation[1],
                "innovation_z": record.innovation[2], "nis": record.nis,
                "vx": record.vx, "vy": record.vy, "vz": record.vz,
            })
            df = build_feature_frame(list(self.ml_history[icao24]), self.config.ml)
            X = feature_matrix(df)
            ml_score = float(self.ml_detector.anomaly_score(X)[-1])
            ml_flag = bool(self.ml_detector.predict_is_anomaly(X)[-1])

            if self.fusion is not None:
                row = FusionRow(
                    nis=nis, ml_score=ml_score,
                    mlat_disagreement_m=mlat_res.disagreement_m, mlat_no_corroboration=mlat_res.no_corroboration,
                    radar_disagreement_m=radar_res.disagreement_m, radar_no_corroboration=radar_res.no_corroboration,
                    identity_mismatch=identity_flag,
                )
                fused_score = float(self.fusion.suspicion([row])[0])
                fused_flag = bool(fused_score > self.fusion.decision_threshold)

        if (sv.callsign or "").strip():
            self.callsigns[icao24] = sv.callsign.strip()

        return {
            "type": "update",
            "icao24": icao24,
            "callsign": self.callsigns.get(icao24),
            "category": record.category.value if record else None,
            "t": sv.observed_at.isoformat(),
            "lat": sv.latitude, "lon": sv.longitude,
            "alt": sv.baro_altitude, "velocity": sv.velocity, "track_deg": sv.true_track,
            "nis": nis, "ml_score": ml_score,
            "mlat_disagreement_m": None if mlat_res.no_corroboration else mlat_res.disagreement_m,
            "radar_disagreement_m": None if radar_res.no_corroboration else radar_res.disagreement_m,
            "fused_score": fused_score,
            "flags": {
                "rule_based": rule_flag, "nis": nis_flag, "ml": ml_flag,
                "mlat": mlat_res.is_anomalous, "radar": radar_res.is_anomalous,
                "identity": identity_flag, "fused": fused_flag,
            },
        }


async def fit_fusion(state: PipelineState) -> None:
    """One-time real fit at startup, same methodology as scripts/export_dashboard_data.py
    (small demo-scale attack generation on real recent substrate + real clean pool),
    just run once in a thread so it doesn't block the poll loop."""
    def _fit():
        config = state.config
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=FUSION_FIT_LOOKBACK_MINUTES)
        with get_connection(config.database) as conn:
            history = list(repository.fetch_track_state_history_for_kalman(conn, time_range=(start, now)))
        pool = build_substrate_pool(history, min_length=config.attacks.min_substrate_track_length)
        if len(pool.tracks) < 20:
            logger.warning("fusion_fit_skipped_insufficient_substrate", extra={"tracks": len(pool.tracks)})
            return None

        attacks_cfg = dataclasses.replace(config.attacks, variants_per_class=FUSION_FIT_VARIANTS_PER_CLASS)
        rng = np.random.default_rng(config.attacks.random_seed + 2)
        rows, labels = [], []
        used = set()
        for ac in AttackClass:
            for v in generate_variants(ac, attacks_cfg, pool, config.opensky.bbox, rng, now):
                used.add(v.icao24)
                r = evaluate_scenario(
                    v.broadcast_state_vectors, v.true_state_vectors, v.is_attacked,
                    config, state.ml_detector, state.mlat_sim, state.radar_sim,
                    attack_class=ac.value, variant_id=v.variant_id, split="fit",
                )
                for row, attacked in zip(r.fusion_rows, r.is_attacked):
                    if row is not None:
                        rows.append(row); labels.append(attacked)

        clean_icao24s = [i for i in pool.tracks if i not in used][:80]
        clean_rows = []
        for icao24 in clean_icao24s:
            track = pool.tracks[icao24]
            r = evaluate_scenario(track, track, [False] * len(track), config, state.ml_detector,
                                   state.mlat_sim, state.radar_sim, attack_class=None, variant_id=icao24, split="clean")
            for row in r.fusion_rows:
                if row is not None:
                    rows.append(row); labels.append(False); clean_rows.append(row)

        fusion = EvidenceFusion(random_state=config.evaluation.fusion_random_state)
        fusion.fit(rows, labels)
        if clean_rows:
            fusion.decision_threshold = float(np.percentile(fusion.suspicion(clean_rows), 99))
        logger.info("fusion_fit_complete", extra={"rows": len(rows), "attacked": sum(labels), "threshold": fusion.decision_threshold})
        return fusion

    state.fusion = await asyncio.to_thread(_fit)


async def poll_loop(state: PipelineState, manager: ConnectionManager) -> None:
    config = state.config
    if state.last_seen_time is None:
        state.last_seen_time = datetime.now(timezone.utc) - timedelta(minutes=2)

    while True:
        try:
            def _fetch():
                with get_connection(config.database) as conn:
                    return list(repository.fetch_track_state_history_for_kalman(
                        conn, time_range=(state.last_seen_time, datetime.now(timezone.utc) + timedelta(seconds=1)),
                    ))
            rows = await asyncio.to_thread(_fetch)
            new_rows = [r for r in rows if r.observed_at > state.last_seen_time]
            new_rows.sort(key=lambda r: r.observed_at)

            for sv in new_rows:
                # process_row does real CPU work (MLAT's multi-start least-squares
                # solve, pandas feature-frame construction) -- run it off the event
                # loop thread so HTTP/WebSocket requests aren't starved while a
                # batch of new rows is being processed.
                msg = await asyncio.to_thread(state.process_row, sv)
                await manager.broadcast(msg)
                state.last_seen_time = max(state.last_seen_time, sv.observed_at)

            if new_rows:
                logger.info("poll_processed", extra={"n": len(new_rows)})
        except Exception:
            logger.exception("poll_loop_error")

        await asyncio.sleep(POLL_SECONDS)


app = FastAPI(title="ADS-B Live Dashboard (local)")
manager = ConnectionManager()
_state: Optional[PipelineState] = None


@app.on_event("startup")
async def startup():
    global _state
    config = get_config()
    configure_logging(config.logging.level, config.logging.format)
    _state = PipelineState(config)
    asyncio.create_task(poll_loop(_state, manager))
    asyncio.create_task(fit_fusion(_state))
    logger.info("live_server_started", extra={"poll_seconds": POLL_SECONDS})


@app.get("/")
async def index():
    return FileResponse(CLIENT_PATH)


@app.get("/api/status")
async def status():
    return JSONResponse({
        "fusion_ready": _state.fusion is not None,
        "last_seen_time": _state.last_seen_time.isoformat() if _state.last_seen_time else None,
        "tracks_active": len(_state.kalman._tracks),
        "clients_connected": len(manager.active),
    })


class InjectRequest(BaseModel):
    attack_class: str
    severity: float


@app.post("/api/inject_attack")
async def inject_attack(req: InjectRequest):
    config = _state.config
    ac = AttackClass(req.attack_class)

    def _run():
        now = datetime.now(timezone.utc)
        start = now - timedelta(minutes=30)
        with get_connection(config.database) as conn:
            history = list(repository.fetch_track_state_history_for_kalman(conn, time_range=(start, now)))
        pool = build_substrate_pool(history, min_length=config.attacks.min_substrate_track_length)
        if len(pool.tracks) < 2:
            return None

        srange = SeverityRange(min=req.severity, max=req.severity)
        attacks_cfg = dataclasses.replace(
            config.attacks, variants_per_class=1,
            severity_ranges={**config.attacks.severity_ranges, ac.value: srange},
        )
        rng = np.random.default_rng()
        variants = generate_variants(ac, attacks_cfg, pool, config.opensky.bbox, rng, now)
        if not variants:
            return None
        v = variants[0]
        r = evaluate_scenario(
            v.broadcast_state_vectors, v.true_state_vectors, v.is_attacked,
            config, _state.ml_detector, _state.mlat_sim, _state.radar_sim,
            attack_class=ac.value, variant_id=v.variant_id, split="live_inject",
        )
        fused_pred, fused_score = [], []
        for row in r.fusion_rows:
            if row is not None and _state.fusion is not None:
                p = float(_state.fusion.suspicion([row])[0])
                fused_score.append(p)
                fused_pred.append(bool(p > _state.fusion.decision_threshold))
            else:
                fused_score.append(None)
                fused_pred.append(False)
        r.predictions["fused"] = fused_pred

        points = []
        for i, bsv in enumerate(v.broadcast_state_vectors):
            tsv = v.true_state_vectors[i]
            points.append({
                "t": bsv.observed_at.isoformat(), "lat": bsv.latitude, "lon": bsv.longitude,
                "true_lat": tsv.latitude if tsv else None, "true_lon": tsv.longitude if tsv else None,
                "is_attacked": bool(v.is_attacked[i]), "callsign": (bsv.callsign or "").strip() or None,
                "nis": r.fusion_rows[i].nis if r.fusion_rows[i] else None,
                "fused_score": fused_score[i],
                "flags": {m: r.predictions[m][i] for m in r.predictions},
            })
        return {
            "type": "attack_result", "request_id": str(uuid.uuid4()),
            "attack_class": ac.value, "severity": req.severity, "variant_id": v.variant_id,
            "substrate_icao24": v.icao24, "n_points": len(points),
            "n_attacked_points": sum(v.is_attacked), "points": points,
        }

    result = await asyncio.to_thread(_run)
    if result is None:
        return JSONResponse({"error": "not enough real substrate traffic in the last 30 minutes to build this variant"}, status_code=503)
    await manager.broadcast(result)
    return JSONResponse({"ok": True, "request_id": result["request_id"], "n_points": result["n_points"]})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
