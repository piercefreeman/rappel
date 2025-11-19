ALTER TABLE daemon_action_ledger
    ADD COLUMN attempt_number INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN max_retries INTEGER NOT NULL DEFAULT 3,
    ADD COLUMN timeout_seconds INTEGER NOT NULL DEFAULT 300,
    ADD COLUMN last_error TEXT,
    ADD COLUMN deadline_at TIMESTAMPTZ,
    ADD COLUMN scheduled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN delivery_token UUID;

CREATE INDEX IF NOT EXISTS idx_action_deadline
    ON daemon_action_ledger (deadline_at)
    WHERE status = 'dispatched' AND deadline_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_action_retry
    ON daemon_action_ledger (partition_id, status, attempt_number)
    WHERE status IN ('failed', 'timed_out');

CREATE INDEX IF NOT EXISTS idx_action_scheduled
    ON daemon_action_ledger (scheduled_at)
    WHERE status = 'queued';

COMMENT ON COLUMN daemon_action_ledger.attempt_number IS
    'Zero-indexed retry counter. 0 = first attempt, 1 = first retry, etc.';

COMMENT ON COLUMN daemon_action_ledger.deadline_at IS
    'Timestamp when the action should be considered timed out. Set when dispatched.';

COMMENT ON COLUMN daemon_action_ledger.max_retries IS
    'Maximum attempts allowed for this action, cached from the DAG node metadata.';

COMMENT ON COLUMN daemon_action_ledger.timeout_seconds IS
    'Timeout duration in seconds for this action, cached from the DAG node metadata.';

COMMENT ON COLUMN daemon_action_ledger.scheduled_at IS
    'Earliest timestamp when this action becomes eligible for dispatch.';
