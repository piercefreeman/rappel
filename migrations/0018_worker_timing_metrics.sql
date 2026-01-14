-- Add worker timing metrics: track which worker handled each action and compute medians.

-- Add worker tracking columns to action_logs
-- worker_id: identifies which worker processed this action
-- enqueued_at: copied from action_queue at dispatch time to compute dequeue latency
ALTER TABLE action_logs ADD COLUMN IF NOT EXISTS worker_id BIGINT;
ALTER TABLE action_logs ADD COLUMN IF NOT EXISTS pool_id UUID;
ALTER TABLE action_logs ADD COLUMN IF NOT EXISTS enqueued_at TIMESTAMPTZ;

-- Index for aggregating by worker
CREATE INDEX IF NOT EXISTS idx_action_logs_worker
    ON action_logs (pool_id, worker_id)
    WHERE worker_id IS NOT NULL;

-- Add median timing columns to worker_status
ALTER TABLE worker_status ADD COLUMN IF NOT EXISTS median_dequeue_ms BIGINT;
ALTER TABLE worker_status ADD COLUMN IF NOT EXISTS median_handling_ms BIGINT;
