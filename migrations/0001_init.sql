CREATE TABLE IF NOT EXISTS workflow_instances (
    id BIGSERIAL PRIMARY KEY,
    partition_id INTEGER NOT NULL,
    workflow_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    next_action_seq INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS daemon_action_ledger (
    id BIGSERIAL PRIMARY KEY,
    instance_id BIGINT NOT NULL REFERENCES workflow_instances(id) ON DELETE CASCADE,
    partition_id INTEGER NOT NULL,
    action_seq INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload BYTEA NOT NULL,
    delivery_id BIGINT,
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dispatched_at TIMESTAMPTZ,
    acked_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    success BOOLEAN,
    result_payload BYTEA,
    UNIQUE(instance_id, action_seq)
);

CREATE INDEX IF NOT EXISTS idx_action_partition_status
    ON daemon_action_ledger (partition_id, status, action_seq);
