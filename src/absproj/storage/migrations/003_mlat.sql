-- Phase 4 schema: SIMULATED multilateration (MLAT) check results. See
-- verification/mlat.py for the "simulated, not a real sensor network" caveat
-- -- it applies to every row in this table.

CREATE TABLE IF NOT EXISTS mlat_checks (
    time            TIMESTAMPTZ NOT NULL,
    icao24          TEXT NOT NULL,
    mlat_latitude   DOUBLE PRECISION NOT NULL,
    mlat_longitude  DOUBLE PRECISION NOT NULL,
    mlat_altitude   DOUBLE PRECISION NOT NULL,
    disagreement_m  DOUBLE PRECISION NOT NULL,
    residual        DOUBLE PRECISION NOT NULL,
    is_anomalous    BOOLEAN NOT NULL,
    PRIMARY KEY (time, icao24)
);

SELECT create_hypertable('mlat_checks', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS mlat_checks_icao24_time_idx ON mlat_checks (icao24, time DESC);
