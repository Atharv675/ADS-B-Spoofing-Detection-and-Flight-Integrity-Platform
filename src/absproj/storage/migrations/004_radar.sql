-- Phase 5 schema: SIMULATED primary radar check results. See
-- verification/radar.py for the "simulated, not a real sensor" caveat --
-- it applies to every row in this table.

CREATE TABLE IF NOT EXISTS radar_checks (
    time                     TIMESTAMPTZ NOT NULL,
    icao24                   TEXT NOT NULL,
    radar_latitude           DOUBLE PRECISION NOT NULL,
    radar_longitude          DOUBLE PRECISION NOT NULL,
    radar_vx                 DOUBLE PRECISION NOT NULL,
    radar_vy                 DOUBLE PRECISION NOT NULL,
    disagreement_m           DOUBLE PRECISION NOT NULL,
    velocity_disagreement_ms DOUBLE PRECISION,
    is_anomalous             BOOLEAN NOT NULL,
    PRIMARY KEY (time, icao24)
);

SELECT create_hypertable('radar_checks', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS radar_checks_icao24_time_idx ON radar_checks (icao24, time DESC);
