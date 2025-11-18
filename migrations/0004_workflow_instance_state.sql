ALTER TABLE workflow_instances
    ADD COLUMN workflow_version_id BIGINT REFERENCES workflow_versions(id),
    ADD COLUMN input_payload BYTEA;

ALTER TABLE daemon_action_ledger
    ADD COLUMN workflow_node_id TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_daemon_action_instance_node
    ON daemon_action_ledger (instance_id, workflow_node_id)
    WHERE workflow_node_id IS NOT NULL;
