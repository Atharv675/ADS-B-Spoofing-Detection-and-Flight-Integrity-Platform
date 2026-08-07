-- Phase 2 schema: per-update Kalman filter output (innovation + innovation
-- covariance + NIS). This is the feature feed the ML layer (Phase 3) reads
-- from -- it must never contain raw ADS-B fields, only filter-derived signals.

CREATE TABLE IF NOT EXISTS kalman_updates (
    time              TIMESTAMPTZ NOT NULL,
    icao24            TEXT NOT NULL,
    category          TEXT NOT NULL,
    dt_seconds        DOUBLE PRECISION NOT NULL,
    innovation_x      DOUBLE PRECISION NOT NULL,
    innovation_y      DOUBLE PRECISION NOT NULL,
    innovation_z      DOUBLE PRECISION NOT NULL,
    innovation_cov    JSONB NOT NULL,
    nis               DOUBLE PRECISION NOT NULL,
    chi2_threshold    DOUBLE PRECISION NOT NULL,
    is_anomalous      BOOLEAN NOT NULL,
    vx                DOUBLE PRECISION NOT NULL,
    vy                DOUBLE PRECISION NOT NULL,
    vz                DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (time, icao24)
);

SELECT create_hypertable('kalman_updates', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS kalman_updates_icao24_time_idx ON kalman_updates (icao24, time DESC);
