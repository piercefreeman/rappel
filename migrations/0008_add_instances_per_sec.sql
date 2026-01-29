-- Add instances per second throughput metric to worker status.
ALTER TABLE worker_status ADD COLUMN instances_per_sec DOUBLE PRECISION NOT NULL DEFAULT 0;
