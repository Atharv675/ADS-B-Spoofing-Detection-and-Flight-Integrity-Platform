-- Phase 8: index on track_state.source, since Phase 8's jamming-zone rows
-- now share the table with the main clean-traffic pipeline's rows,
-- distinguished only by this column.

CREATE INDEX IF NOT EXISTS track_state_source_idx ON track_state (source);
