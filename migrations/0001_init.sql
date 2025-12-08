-- Initial schema for rappel workflow engine.
--
-- Two main tables:
-- 1. workflow_instances: Tracks workflow execution state
-- 2. actions: Queue of actions per instance (inbox pattern)

-- Instance status enum
CREATE TYPE instance_status AS ENUM ('running', 'waiting_for_actions', 'completed', 'failed');

-- Action status enum
CREATE TYPE action_status AS ENUM ('queued', 'dispatched', 'completed', 'failed');

-- Workflow instances
CREATE TABLE workflow_instances (
    id UUID PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    module_name TEXT NOT NULL,
    status instance_status NOT NULL DEFAULT 'running',

    -- Initial arguments for the workflow
    initial_args JSONB NOT NULL DEFAULT '{}',

    -- Final result when completed
    result JSONB,

    -- How many actions have been completed (used for replay index)
    actions_until_index INTEGER NOT NULL DEFAULT 0,

    -- When to next schedule this instance for execution
    -- NULL means ready immediately, non-NULL means wait until then
    scheduled_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for claiming ready instances
CREATE INDEX idx_instances_ready ON workflow_instances (status, scheduled_at, created_at)
    WHERE status = 'waiting_for_actions';

-- Actions queue (inbox per instance)
CREATE TABLE actions (
    id UUID PRIMARY KEY,
    instance_id UUID NOT NULL REFERENCES workflow_instances(id) ON DELETE CASCADE,

    -- Position in the action sequence for this instance
    sequence INTEGER NOT NULL,

    -- Action details
    action_name TEXT NOT NULL,
    module_name TEXT NOT NULL,
    kwargs JSONB NOT NULL DEFAULT '{}',

    -- Execution state
    status action_status NOT NULL DEFAULT 'queued',
    result JSONB,
    error_message TEXT,

    -- Dispatch token for idempotent completion
    dispatch_token UUID,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Ensure unique sequence per instance
    UNIQUE (instance_id, sequence)
);

-- Index for claiming queued actions
CREATE INDEX idx_actions_queued ON actions (status, created_at)
    WHERE status = 'queued';

-- Index for getting completed actions by instance
CREATE INDEX idx_actions_instance_completed ON actions (instance_id, sequence)
    WHERE status = 'completed';

-- Trigger to update updated_at on row changes
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER workflow_instances_updated_at
    BEFORE UPDATE ON workflow_instances
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER actions_updated_at
    BEFORE UPDATE ON actions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
