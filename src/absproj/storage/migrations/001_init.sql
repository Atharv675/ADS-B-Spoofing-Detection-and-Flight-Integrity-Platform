-- Phase 1 schema: raw messages, tracks, per-update track state, and a stub
-- detections table (populated starting Phase 2+).
-- Safe to re-run: every statement is idempotent.

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS postgis;

-- Raw OpenSky batch responses, kept verbatim for replay/audit. Not meant to be
-- queried in the hot path -- track_state below is the normalized read path.
CREATE TABLE IF NOT EXISTS raw_messages (
    ingested_at TIMESTAMPTZ NOT NULL,
    batch_time  TIMESTAMPTZ NOT NULL,
    payload     JSONB NOT NULL
);

SELECT create_hypertable('raw_messages', 'ingested_at', if_not_exists => TRUE);

-- One row per aircraft (ICAO24) ever seen.
CREATE TABLE IF NOT EXISTS tracks (
    icao24        TEXT PRIMARY KEY,
    first_seen    TIMESTAMPTZ NOT NULL,
    last_seen     TIMESTAMPTZ NOT NULL,
    last_callsign TEXT,
    -- Raw OpenSky aircraft category code (0-19); bucketed into dynamics classes
    -- (light / transport / high-performance, etc.) starting in Phase 2.
    category      SMALLINT
);

-- Per-update normalized state, one row per (time, icao24). This is what the
-- Kalman filter (Phase 2) and everything downstream reads.
CREATE TABLE IF NOT EXISTS track_state (
    time            TIMESTAMPTZ NOT NULL,
    icao24          TEXT NOT NULL REFERENCES tracks (icao24),
    longitude       DOUBLE PRECISION NOT NULL,
    latitude        DOUBLE PRECISION NOT NULL,
    geom            GEOGRAPHY(Point, 4326) NOT NULL,
    baro_altitude   DOUBLE PRECISION,
    geo_altitude    DOUBLE PRECISION,
    velocity        DOUBLE PRECISION,
    true_track      DOUBLE PRECISION,
    vertical_rate   DOUBLE PRECISION,
    on_ground       BOOLEAN NOT NULL DEFAULT FALSE,
    squawk          TEXT,
    spi             BOOLEAN NOT NULL DEFAULT FALSE,
    position_source SMALLINT,
    source          TEXT NOT NULL DEFAULT 'opensky',
    PRIMARY KEY (time, icao24)
);

SELECT create_hypertable('track_state', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS track_state_icao24_time_idx ON track_state (icao24, time DESC);
CREATE INDEX IF NOT EXISTS track_state_geom_idx ON track_state USING GIST (geom);

-- Detection/alert results. Schema stubbed now; populated from Phase 2 onward
-- (method = 'nis' | 'ml' | 'mlat' | 'radar' | 'fused').
CREATE TABLE IF NOT EXISTS detections (
    time        TIMESTAMPTZ NOT NULL,
    icao24      TEXT NOT NULL,
    method      TEXT NOT NULL,
    score       DOUBLE PRECISION,
    is_anomaly  BOOLEAN,
    details     JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

SELECT create_hypertable('detections', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS detections_icao24_time_idx ON detections (icao24, time DESC);
