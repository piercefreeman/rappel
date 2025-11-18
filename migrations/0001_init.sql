CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE workflow_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_name TEXT NOT NULL,
    dag_hash TEXT NOT NULL,
    dag_proto BYTEA NOT NULL,
    concurrent BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (workflow_name, dag_hash)
);

CREATE TABLE workflow_instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partition_id INTEGER NOT NULL,
    workflow_name TEXT NOT NULL,
    workflow_version_id UUID REFERENCES workflow_versions(id),
    next_action_seq INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    input_payload BYTEA
);

CREATE TABLE daemon_action_ledger (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instance_id UUID NOT NULL REFERENCES workflow_instances(id) ON DELETE CASCADE,
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
    workflow_node_id TEXT,
    UNIQUE (instance_id, action_seq)
);

CREATE INDEX idx_action_partition_status
    ON daemon_action_ledger (partition_id, status, action_seq);

CREATE UNIQUE INDEX idx_daemon_action_instance_node
    ON daemon_action_ledger (instance_id, workflow_node_id)
    WHERE workflow_node_id IS NOT NULL;
