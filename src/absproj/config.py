"""Loads non-secret config from config/config.yaml and secrets from environment/.env.

Nothing downstream should read os.environ or the yaml file directly; go through
get_config() so there is exactly one place that assembles configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class BBox:
    lamin: float
    lamax: float
    lomin: float
    lomax: float


@dataclass
class OpenSkyConfig:
    states_url: str
    token_url: str
    bbox: BBox
    request_timeout_seconds: float
    poll_interval_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

    @property
    def has_credentials(self) -> bool:
        return bool(self.client_id and self.client_secret)


@dataclass
class DatabaseConfig:
    host: str
    port: int
    name: str
    user: str
    password: str
    connect_timeout_seconds: float

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.name} "
            f"user={self.user} password={self.password} "
            f"connect_timeout={int(self.connect_timeout_seconds)}"
        )


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"


@dataclass
class KalmanConfig:
    process_noise_sigma_a: dict[str, float]
    sigma_horizontal_m: float
    sigma_vertical_m: float
    initial_velocity_std_mps: float
    chi_square_alpha: float
    track_gap_reset_seconds: float


@dataclass
class IsolationForestConfig:
    n_estimators: int
    contamination: float
    random_state: int


@dataclass
class MLConfig:
    rolling_window: int
    level_flight_vz_threshold_mps: float
    isolation_forest: IsolationForestConfig


@dataclass
class ReceiverConfig:
    name: str
    lat: float
    lon: float
    alt: float


@dataclass
class MLATConfig:
    receivers: list[ReceiverConfig]
    reference_receiver_index: int
    speed_of_light_mps: float
    timing_noise_std_ns: float
    disagreement_threshold_m: float
    random_seed: int


@dataclass
class RadarSiteConfig:
    lat: float
    lon: float


@dataclass
class RadarConfig:
    site: RadarSiteConfig
    plot_interval_s: float
    range_noise_std_m: float
    azimuth_noise_std_deg: float
    disagreement_threshold_m: float
    random_seed: int


@dataclass
class SeverityRange:
    min: float
    max: float


@dataclass
class AttacksConfig:
    min_substrate_track_length: int
    variants_per_class: int
    train_fraction: float
    holdout_class: str
    random_seed: int
    severity_ranges: dict[str, SeverityRange]


@dataclass
class RuleBasedConfig:
    max_speed_mps: float
    max_turn_rate_deg_s: float


@dataclass
class EvaluationConfig:
    clean_train_track_count: int
    clean_test_track_count: int
    fusion_random_state: int
    random_seed: int


@dataclass
class JammingZoneConfig:
    bbox: BBox
    source_tag: str
    collection_minutes: float
    poll_interval_seconds: float
    mlat_receivers: list[ReceiverConfig]
    radar_site: RadarSiteConfig
    control_window_minutes: float
    control_random_seed: int


@dataclass
class AppConfig:
    opensky: OpenSkyConfig
    database: DatabaseConfig
    kalman: KalmanConfig
    ml: MLConfig
    mlat: MLATConfig
    radar: RadarConfig
    attacks: AttacksConfig
    rule_based: RuleBasedConfig
    evaluation: EvaluationConfig
    jamming_zone: JammingZoneConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_config(config_path: Optional[Path] = None, env_path: Optional[Path] = None) -> AppConfig:
    load_dotenv(dotenv_path=env_path or (REPO_ROOT / ".env"), override=False)

    raw = _load_yaml(config_path or DEFAULT_CONFIG_PATH)

    osky_raw = raw["opensky"]
    opensky = OpenSkyConfig(
        states_url=osky_raw["states_url"],
        token_url=osky_raw["token_url"],
        bbox=BBox(**osky_raw["bbox"]),
        request_timeout_seconds=float(osky_raw["request_timeout_seconds"]),
        poll_interval_seconds=float(osky_raw["poll_interval_seconds"]),
        max_retries=int(osky_raw["max_retries"]),
        retry_backoff_seconds=float(osky_raw["retry_backoff_seconds"]),
        client_id=os.environ.get("OPENSKY_CLIENT_ID") or None,
        client_secret=os.environ.get("OPENSKY_CLIENT_SECRET") or None,
    )

    db_raw = raw["database"]
    database = DatabaseConfig(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        name=os.environ.get("POSTGRES_DB", "adsb"),
        user=os.environ.get("POSTGRES_USER", "adsb"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        connect_timeout_seconds=float(db_raw["connect_timeout_seconds"]),
    )

    log_raw = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level=log_raw.get("level", "INFO"),
        format=log_raw.get("format", "json"),
    )

    kalman_raw = raw["kalman"]
    kalman = KalmanConfig(
        process_noise_sigma_a={k: float(v) for k, v in kalman_raw["process_noise_sigma_a"].items()},
        sigma_horizontal_m=float(kalman_raw["measurement_noise"]["sigma_horizontal_m"]),
        sigma_vertical_m=float(kalman_raw["measurement_noise"]["sigma_vertical_m"]),
        initial_velocity_std_mps=float(kalman_raw["initial_velocity_std_mps"]),
        chi_square_alpha=float(kalman_raw["chi_square_alpha"]),
        track_gap_reset_seconds=float(kalman_raw["track_gap_reset_seconds"]),
    )

    ml_raw = raw["ml"]
    if_raw = ml_raw["isolation_forest"]
    ml = MLConfig(
        rolling_window=int(ml_raw["rolling_window"]),
        level_flight_vz_threshold_mps=float(ml_raw["level_flight_vz_threshold_mps"]),
        isolation_forest=IsolationForestConfig(
            n_estimators=int(if_raw["n_estimators"]),
            contamination=float(if_raw["contamination"]),
            random_state=int(if_raw["random_state"]),
        ),
    )

    mlat_raw = raw["mlat"]
    mlat = MLATConfig(
        receivers=[ReceiverConfig(**r) for r in mlat_raw["receivers"]],
        reference_receiver_index=int(mlat_raw["reference_receiver_index"]),
        speed_of_light_mps=float(mlat_raw["speed_of_light_mps"]),
        timing_noise_std_ns=float(mlat_raw["timing_noise_std_ns"]),
        disagreement_threshold_m=float(mlat_raw["disagreement_threshold_m"]),
        random_seed=int(mlat_raw["random_seed"]),
    )

    radar_raw = raw["radar"]
    radar = RadarConfig(
        site=RadarSiteConfig(**radar_raw["site"]),
        plot_interval_s=float(radar_raw["plot_interval_s"]),
        range_noise_std_m=float(radar_raw["range_noise_std_m"]),
        azimuth_noise_std_deg=float(radar_raw["azimuth_noise_std_deg"]),
        disagreement_threshold_m=float(radar_raw["disagreement_threshold_m"]),
        random_seed=int(radar_raw["random_seed"]),
    )

    attacks_raw = raw["attacks"]
    attacks = AttacksConfig(
        min_substrate_track_length=int(attacks_raw["min_substrate_track_length"]),
        variants_per_class=int(attacks_raw["variants_per_class"]),
        train_fraction=float(attacks_raw["train_fraction"]),
        holdout_class=attacks_raw["holdout_class"],
        random_seed=int(attacks_raw["random_seed"]),
        severity_ranges={
            k: SeverityRange(min=float(v["min"]), max=float(v["max"]))
            for k, v in attacks_raw["severity_ranges"].items()
        },
    )

    rule_based_raw = raw["rule_based"]
    rule_based = RuleBasedConfig(
        max_speed_mps=float(rule_based_raw["max_speed_mps"]),
        max_turn_rate_deg_s=float(rule_based_raw["max_turn_rate_deg_s"]),
    )

    eval_raw = raw["evaluation"]
    evaluation = EvaluationConfig(
        clean_train_track_count=int(eval_raw["clean_train_track_count"]),
        clean_test_track_count=int(eval_raw["clean_test_track_count"]),
        fusion_random_state=int(eval_raw["fusion_random_state"]),
        random_seed=int(eval_raw["random_seed"]),
    )

    jz_raw = raw["jamming_zone"]
    jamming_zone = JammingZoneConfig(
        bbox=BBox(**jz_raw["bbox"]),
        source_tag=jz_raw["source_tag"],
        collection_minutes=float(jz_raw["collection_minutes"]),
        poll_interval_seconds=float(jz_raw["poll_interval_seconds"]),
        mlat_receivers=[ReceiverConfig(**r) for r in jz_raw["mlat_receivers"]],
        radar_site=RadarSiteConfig(**jz_raw["radar_site"]),
        control_window_minutes=float(jz_raw["control_window_minutes"]),
        control_random_seed=int(jz_raw["control_random_seed"]),
    )

    return AppConfig(
        opensky=opensky, database=database, kalman=kalman, ml=ml, mlat=mlat, radar=radar,
        attacks=attacks, rule_based=rule_based, evaluation=evaluation, jamming_zone=jamming_zone,
        logging=logging_cfg,
    )
