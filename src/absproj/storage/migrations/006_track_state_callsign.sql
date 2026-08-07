-- Phase 10: track_state never persisted callsign per-update -- only
-- tracks.last_callsign (latest value only) existed, so any consumer
-- reading track_state (every batch script since Phase 2) silently got
-- callsign=None for every row, even though StateVector carries it from
-- ingestion onward. This matters starting now because an identity
-- consistency check needs the callsign *at each update* to detect a
-- mid-track change (e.g. an ICAO24 collision, where the intruder's
-- callsign differs from the victim's) -- the latest-only value in `tracks`
-- can't do that. Existing rows will have NULL callsign (not backfilled);
-- only newly-ingested data after this migration has it.

ALTER TABLE track_state ADD COLUMN IF NOT EXISTS callsign TEXT;
