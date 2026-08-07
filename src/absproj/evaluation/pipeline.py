"""Runs every detection method (rule-based, NIS, ML, MLAT, radar, identity)
over one scenario -- an attack variant or a clean track -- producing per-row
predictions that metrics.py aggregates and fusion.py combines.

A "scenario" is just a (broadcast, true, is_attacked) triple of equal-length
sequences: attacks/types.AttackedTrack fields directly, or a clean track with
true == broadcast and is_attacked all False. Each scenario gets its own fresh
KalmanTrackManager -- consistent with how the DB-batch phases process many
independent per-icao24 tracks, just done here in memory per scenario instead
of over the whole accumulated history at once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from absproj.config import AppConfig
from absproj.evaluation.fusion import FusionRow
from absproj.evaluation.identity_check import IdentityTracker
from absproj.evaluation.rule_based import check_rule_based
from absproj.ingestion.normalize import StateVector
from absproj.ml.features import build_feature_frame, feature_matrix
from absproj.ml.isolation_forest import IsolationForestDetector
from absproj.tracking.track_manager import KalmanTrackManager
from absproj.verification.mlat import MLATSimulator
from absproj.verification.radar import RadarSimulator

METHODS = ("rule_based", "nis", "ml", "mlat", "radar", "identity")


@dataclass
class ScenarioResult:
    attack_class: Optional[str]
    variant_id: str
    split: str
    times: list[datetime]
    is_attacked: list[bool]
    has_full_signal: list[bool]
    predictions: dict[str, list[bool]] = field(default_factory=dict)
    fusion_rows: list[Optional[FusionRow]] = field(default_factory=list)


def evaluate_scenario(
    broadcast_svs: list[StateVector],
    true_svs: list[Optional[StateVector]],
    is_attacked: list[bool],
    config: AppConfig,
    ml_detector: IsolationForestDetector,
    mlat_sim: MLATSimulator,
    radar_sim: RadarSimulator,
    attack_class: Optional[str],
    variant_id: str,
    split: str,
) -> ScenarioResult:
    n = len(broadcast_svs)
    times = [sv.observed_at for sv in broadcast_svs]

    manager = KalmanTrackManager(config.kalman)
    identity_tracker = IdentityTracker()
    nis_pred = [False] * n
    nis_values = [0.0] * n
    rule_pred = [False] * n
    identity_pred = [False] * n
    mlat_pred = [False] * n
    mlat_disagreement = [0.0] * n
    mlat_no_corrob = [False] * n
    radar_pred = [False] * n
    radar_disagreement = [0.0] * n
    radar_no_corrob = [False] * n

    kalman_feature_rows = []
    prev_broadcast = None

    for i, (bsv, tsv) in enumerate(zip(broadcast_svs, true_svs)):
        record = manager.process(bsv)
        if record is not None:
            nis_pred[i] = record.is_anomalous
            nis_values[i] = record.nis
            kalman_feature_rows.append({
                "orig_index": i,
                "time": record.time,
                "icao24": record.icao24,
                "category": record.category.value,
                "dt_seconds": record.dt_seconds,
                "innovation_x": record.innovation[0],
                "innovation_y": record.innovation[1],
                "innovation_z": record.innovation[2],
                "nis": record.nis,
                "vx": record.vx,
                "vy": record.vy,
                "vz": record.vz,
            })

        rule_pred[i] = check_rule_based(prev_broadcast, bsv, config.rule_based)
        identity_pred[i] = identity_tracker.check(bsv)
        prev_broadcast = bsv

        mres = mlat_sim.check_with_ground_truth(broadcast_sv=bsv, true_sv=tsv)
        mlat_pred[i] = mres.is_anomalous
        mlat_disagreement[i] = mres.disagreement_m
        mlat_no_corrob[i] = mres.no_corroboration

        rres = radar_sim.check_with_ground_truth(broadcast_sv=bsv, true_sv=tsv)
        radar_pred[i] = rres.is_anomalous
        radar_disagreement[i] = rres.disagreement_m
        radar_no_corrob[i] = rres.no_corroboration

    ml_pred = [False] * n
    ml_scores = [0.0] * n
    has_full_signal = [False] * n
    if kalman_feature_rows:
        df = build_feature_frame(kalman_feature_rows, config.ml)
        X = feature_matrix(df)
        scores = ml_detector.anomaly_score(X)
        flags = ml_detector.predict_is_anomaly(X)
        for orig_index, score, flag in zip(df["orig_index"], scores, flags):
            ml_pred[orig_index] = bool(flag)
            ml_scores[orig_index] = float(score)
            has_full_signal[orig_index] = True

    fusion_rows: list[Optional[FusionRow]] = [
        FusionRow(
            nis=nis_values[i], ml_score=ml_scores[i],
            mlat_disagreement_m=mlat_disagreement[i], mlat_no_corroboration=mlat_no_corrob[i],
            radar_disagreement_m=radar_disagreement[i], radar_no_corroboration=radar_no_corrob[i],
            identity_mismatch=identity_pred[i],
        ) if has_full_signal[i] else None
        for i in range(n)
    ]

    return ScenarioResult(
        attack_class=attack_class,
        variant_id=variant_id,
        split=split,
        times=times,
        is_attacked=is_attacked,
        has_full_signal=has_full_signal,
        predictions={
            "rule_based": rule_pred,
            "nis": nis_pred,
            "ml": ml_pred,
            "mlat": mlat_pred,
            "radar": radar_pred,
            "identity": identity_pred,
        },
        fusion_rows=fusion_rows,
    )
