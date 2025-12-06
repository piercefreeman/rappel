-- Parallel aggregator counters
-- Used for atomic counting of parallel action completions.
-- Each parallel block has one aggregator that waits for all actions to complete.
-- This table tracks how many have completed with atomic increment.

CREATE TABLE parallel_counters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instance_id UUID NOT NULL REFERENCES workflow_instances(id) ON DELETE CASCADE,
    -- The aggregator node that is waiting for completions
    aggregator_node_id TEXT NOT NULL,
    -- Counter value (number of completed parallel actions)
    count INTEGER NOT NULL DEFAULT 1,
    -- Timestamp
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Unique constraint for atomic upsert
    UNIQUE (instance_id, aggregator_node_id)
);

-- Index for cleanup
CREATE INDEX idx_parallel_counters_instance ON parallel_counters(instance_id);
