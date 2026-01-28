-- Pool-level worker stats with 24hr time-series ring buffer.
-- Re-keys worker_status from (pool_id, worker_id) to pool_id only.

-- Drop old per-worker primary key and add pool-level key
ALTER TABLE worker_status DROP CONSTRAINT worker_status_pkey;

-- Drop worker_id (no longer needed — one row per host pool)
ALTER TABLE worker_status DROP COLUMN worker_id;

ALTER TABLE worker_status ADD PRIMARY KEY (pool_id);

-- Add new columns for pool-level stats
ALTER TABLE worker_status ADD COLUMN active_workers INT NOT NULL DEFAULT 0;
ALTER TABLE worker_status ADD COLUMN actions_per_sec DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE worker_status ADD COLUMN avg_instance_duration_secs DOUBLE PRECISION;
ALTER TABLE worker_status ADD COLUMN time_series BYTEA;
