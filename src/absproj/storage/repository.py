"""Insert/query helpers used by ingestion (and later phases). All SQL lives here
so callers never write raw queries inline."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterable, Optional

import psycopg2.extensions
from psycopg2.extras import execute_values

from absproj.ingestion.normalize import StateVector
from absproj.tracking.track_manager import KalmanUpdateRecord
from absproj.verification.mlat import MLATResult
from absproj.verification.radar import RadarResult


def insert_raw_message(conn: psycopg2.extensions.connection, batch_time: datetime, payload: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO raw_messages (ingested_at, batch_time, payload) VALUES (%s, %s, %s)",
            (datetime.now(timezone.utc), batch_time, json.dumps(payload)),
        )
    conn.commit()


def upsert_tracks(conn: psycopg2.extensions.connection, states: Iterable[StateVector]) -> int:
    rows = [
        (s.icao24, s.observed_at, s.observed_at, s.callsign, s.category)
        for s in states
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO tracks (icao24, first_seen, last_seen, last_callsign, category)
            VALUES %s
            ON CONFLICT (icao24) DO UPDATE SET
                last_seen = EXCLUDED.last_seen,
                last_callsign = COALESCE(EXCLUDED.last_callsign, tracks.last_callsign),
                category = COALESCE(EXCLUDED.category, tracks.category)
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def insert_track_states(conn: psycopg2.extensions.connection, states: Iterable[StateVector], source: str = "opensky") -> int:
    rows = [
        (
            s.observed_at,
            s.icao24,
            s.longitude,
            s.latitude,
            s.longitude,
            s.latitude,
            s.baro_altitude,
            s.geo_altitude,
            s.velocity,
            s.true_track,
            s.vertical_rate,
            s.on_ground,
            s.squawk,
            s.spi,
            s.position_source,
            source,
            s.callsign,
        )
        for s in states
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO track_state (
                time, icao24, longitude, latitude, geom,
                baro_altitude, geo_altitude, velocity, true_track, vertical_rate,
                on_ground, squawk, spi, position_source, source, callsign
            )
            VALUES %s
            ON CONFLICT (time, icao24) DO NOTHING
            """,
            rows,
            template="(%s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, "
                      "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        )
    conn.commit()
    return len(rows)



def fetch_track_state_history_for_kalman(
    conn: psycopg2.extensions.connection,
    source: str = "opensky",
    time_range: Optional[tuple[datetime, datetime]] = None,
) -> Iterable[StateVector]:
    """Yields every *airborne* track_state row (joined with tracks.category),
    ordered by (icao24, time) -- i.e. grouped into per-track chronological
    sequences, ready to feed straight into KalmanTrackManager.process() in order.

    on_ground=true rows are excluded: ground vehicles/taxiing aircraft are a
    different dynamics regime (near-stationary, erratic, multipath-prone
    positions near buildings/hangars) that a flight-dynamics constant-velocity
    model isn't meant to represent, and this project's threat model is
    airborne aircraft spoofing, not airport ground vehicle tracking. Verified
    on real traffic: ground-vehicle position glitches (e.g. a single wildly
    wrong MLAT fix for a taxiing vehicle) produced NIS values in the millions
    that would have swamped the false-positive-rate measurement without this.

    source: defaults to 'opensky' -- the main clean-traffic pipeline (Phases
    1-7) only ever sees that, keeping it isolated from Phase 8's separately
    tagged jamming-zone rows even though they live in the same table.
    time_range: optional (start, end) to scope to a specific window, e.g.
    Phase 8's comparable-duration control sample.
    """
    query = """
        SELECT
            ts.icao24, ts.time, ts.longitude, ts.latitude,
            ts.baro_altitude, ts.geo_altitude, ts.velocity, ts.true_track,
            ts.vertical_rate, ts.on_ground, ts.squawk, ts.spi, ts.position_source,
            t.category, ts.callsign
        FROM track_state ts
        JOIN tracks t ON t.icao24 = ts.icao24
        WHERE ts.on_ground = FALSE AND ts.source = %(source)s
    """
    params: dict = {"source": source}
    if time_range is not None:
        query += " AND ts.time >= %(start)s AND ts.time < %(end)s"
        params["start"], params["end"] = time_range
    query += " ORDER BY ts.icao24, ts.time"
    # A plain (client-side) cursor, not a named server-side one: this project's
    # data volumes are small enough to fetch in one go, and it sidesteps
    # server-side cursors being invalidated by commits on writes interleaved
    # on the same connection (which run_kalman.py does, in batches).
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        for row in rows:
            (icao24, time, longitude, latitude, baro_altitude, geo_altitude,
             velocity, true_track, vertical_rate, on_ground, squawk, spi,
             position_source, category, callsign) = row
            yield StateVector(
                icao24=icao24,
                callsign=callsign,
                origin_country=None,
                time_position=None,
                last_contact=int(time.timestamp()),
                longitude=longitude,
                latitude=latitude,
                baro_altitude=baro_altitude,
                on_ground=on_ground,
                velocity=velocity,
                true_track=true_track,
                vertical_rate=vertical_rate,
                geo_altitude=geo_altitude,
                squawk=squawk,
                spi=spi,
                position_source=position_source,
                category=category,
                observed_at=time,
            )


def insert_kalman_updates(conn: psycopg2.extensions.connection, records: Iterable[KalmanUpdateRecord]) -> int:
    rows = [
        (
            r.time, r.icao24, r.category.value, r.dt_seconds,
            r.innovation[0], r.innovation[1], r.innovation[2],
            json.dumps(r.innovation_cov), r.nis, r.chi2_threshold, r.is_anomalous,
            r.vx, r.vy, r.vz,
        )
        for r in records
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO kalman_updates (
                time, icao24, category, dt_seconds,
                innovation_x, innovation_y, innovation_z,
                innovation_cov, nis, chi2_threshold, is_anomalous,
                vx, vy, vz
            )
            VALUES %s
            ON CONFLICT (time, icao24) DO NOTHING
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def insert_nis_detections(conn: psycopg2.extensions.connection, records: Iterable[KalmanUpdateRecord]) -> int:
    """Writes the NIS/chi-square test's own anomaly calls into `detections`
    (method='nis') -- it is benchmarked as a standalone detector, not just an
    ML feature source."""
    rows = [
        (
            r.time, r.icao24, "nis", r.nis, r.is_anomalous,
            json.dumps({"chi2_threshold": r.chi2_threshold, "category": r.category.value}),
        )
        for r in records
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO detections (time, icao24, method, score, is_anomaly, details)
            VALUES %s
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def truncate_kalman_outputs(conn: psycopg2.extensions.connection) -> None:
    """Clears kalman_updates and this run's NIS detections so run_kalman.py can
    reprocess the full accumulated history idempotently."""
    with conn.cursor() as cur:
        cur.execute("TRUNCATE kalman_updates")
        cur.execute("DELETE FROM detections WHERE method = 'nis'")
    conn.commit()


def fetch_kalman_updates_for_ml(conn: psycopg2.extensions.connection) -> list[dict]:
    """Returns every kalman_updates row as a plain dict, the only input the ML
    feature layer is allowed to read from -- filter-derived signals, no raw
    ADS-B fields."""
    query = """
        SELECT time, icao24, category, dt_seconds,
               innovation_x, innovation_y, innovation_z, nis, vx, vy, vz,
               is_anomalous AS nis_is_anomalous
        FROM kalman_updates
        ORDER BY icao24, time
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def insert_ml_detections(conn: psycopg2.extensions.connection, rows: list[tuple]) -> int:
    """rows: (time, icao24, score, is_anomaly) tuples, method='ml'."""
    if not rows:
        return 0
    payload = [(time, icao24, "ml", float(score), bool(is_anom), None) for time, icao24, score, is_anom in rows]
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO detections (time, icao24, method, score, is_anomaly, details) VALUES %s",
            payload,
        )
    conn.commit()
    return len(payload)


def truncate_ml_detections(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM detections WHERE method = 'ml'")
    conn.commit()


def insert_mlat_checks(conn: psycopg2.extensions.connection, records: Iterable[MLATResult]) -> int:
    rows = [
        (
            r.time, r.icao24, r.mlat_latitude, r.mlat_longitude, r.mlat_altitude,
            r.disagreement_m, r.residual, r.is_anomalous,
        )
        for r in records
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO mlat_checks (
                time, icao24, mlat_latitude, mlat_longitude, mlat_altitude,
                disagreement_m, residual, is_anomalous
            )
            VALUES %s
            ON CONFLICT (time, icao24) DO NOTHING
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def insert_mlat_detections(conn: psycopg2.extensions.connection, records: Iterable[MLATResult]) -> int:
    rows = [
        (r.time, r.icao24, "mlat", r.disagreement_m, r.is_anomalous, json.dumps({"residual": r.residual}))
        for r in records
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO detections (time, icao24, method, score, is_anomaly, details) VALUES %s",
            rows,
        )
    conn.commit()
    return len(rows)


def truncate_mlat_outputs(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE mlat_checks")
        cur.execute("DELETE FROM detections WHERE method = 'mlat'")
    conn.commit()


def insert_radar_checks(conn: psycopg2.extensions.connection, records: Iterable[RadarResult]) -> int:
    rows = [
        (
            r.time, r.icao24, r.radar_latitude, r.radar_longitude, r.radar_vx, r.radar_vy,
            r.disagreement_m, r.velocity_disagreement_mps, r.is_anomalous,
        )
        for r in records
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO radar_checks (
                time, icao24, radar_latitude, radar_longitude, radar_vx, radar_vy,
                disagreement_m, velocity_disagreement_ms, is_anomalous
            )
            VALUES %s
            ON CONFLICT (time, icao24) DO NOTHING
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def insert_radar_detections(conn: psycopg2.extensions.connection, records: Iterable[RadarResult]) -> int:
    rows = [
        (
            r.time, r.icao24, "radar", r.disagreement_m, r.is_anomalous,
            json.dumps({"velocity_disagreement_mps": r.velocity_disagreement_mps}),
        )
        for r in records
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO detections (time, icao24, method, score, is_anomaly, details) VALUES %s",
            rows,
        )
    conn.commit()
    return len(rows)


def truncate_radar_outputs(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE radar_checks")
        cur.execute("DELETE FROM detections WHERE method = 'radar'")
    conn.commit()


def fetch_track_state_time_bounds(conn: psycopg2.extensions.connection, source: str = "opensky") -> tuple[datetime, datetime]:
    """Min/max observed time for a given source's airborne track_state rows --
    used to pick a random comparable-duration control window."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT min(time), max(time) FROM track_state WHERE on_ground = FALSE AND source = %s",
            (source,),
        )
        start, end = cur.fetchone()
        return start, end
