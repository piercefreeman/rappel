-- Core schema for Rappel workflow execution engine
-- Uses Postgres-specific features: UUID, TIMESTAMPTZ, BYTEA, SKIP LOCKED support

-- Workflow version definitions (compiled IR programs)
CREATE TABLE workflow_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_name TEXT NOT NULL,
    dag_hash TEXT NOT NULL,
    -- Serialized AST proto (rappel.ast.Program)
    program_proto BYTEA NOT NULL,
    concurrent BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (workflow_name, dag_hash)
);

CREATE INDEX idx_workflow_versions_name ON workflow_versions(workflow_name);

-- Workflow instances (running or completed executions)
CREATE TABLE workflow_instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    partition_id INTEGER NOT NULL DEFAULT 0,
    workflow_name TEXT NOT NULL,
    workflow_version_id UUID REFERENCES workflow_versions(id),
    -- Sequence counter for actions within this instance
    next_action_seq INTEGER NOT NULL DEFAULT 0,
    -- Serialized input arguments (WorkflowArguments proto)
    input_payload BYTEA,
    -- Serialized result (WorkflowArguments proto) - set when complete
    result_payload BYTEA,
    -- Status: 'running', 'completed', 'failed'
    status TEXT NOT NULL DEFAULT 'running',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_workflow_instances_status ON workflow_instances(status);
CREATE INDEX idx_workflow_instances_version ON workflow_instances(workflow_version_id);

-- Action queue (the core work queue for distributed execution)
CREATE TABLE action_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instance_id UUID NOT NULL REFERENCES workflow_instances(id) ON DELETE CASCADE,
    partition_id INTEGER NOT NULL DEFAULT 0,
    -- Sequence number within the instance (for ordering)
    action_seq INTEGER NOT NULL,
    -- Action identification
    module_name TEXT NOT NULL,
    action_name TEXT NOT NULL,
    -- Serialized dispatch payload (ActionDispatch proto or similar)
    dispatch_payload BYTEA NOT NULL,

    -- Status: 'queued', 'dispatched', 'completed', 'failed', 'timed_out'
    status TEXT NOT NULL DEFAULT 'queued',

    -- Retry configuration
    attempt_number INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    timeout_seconds INTEGER NOT NULL DEFAULT 300,
    timeout_retry_limit INTEGER NOT NULL DEFAULT 3,
    -- 'failure' or 'timeout' - determines which retry limit applies
    retry_kind TEXT NOT NULL DEFAULT 'failure',

    -- Backoff configuration
    -- 'none', 'linear', 'exponential'
    backoff_kind TEXT NOT NULL DEFAULT 'none',
    backoff_base_delay_ms INTEGER NOT NULL DEFAULT 0,
    backoff_multiplier DOUBLE PRECISION NOT NULL DEFAULT 2.0,

    -- Dispatch tracking
    delivery_id BIGINT,
    -- Unique token for idempotent completion marking
    delivery_token UUID,

    -- Timestamps
    scheduled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deadline_at TIMESTAMPTZ,
    enqueued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dispatched_at TIMESTAMPTZ,
    acked_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- Result
    success BOOLEAN,
    result_payload BYTEA,
    last_error TEXT,

    -- DAG node reference (for tracking which IR node this action represents)
    node_id TEXT,

    UNIQUE (instance_id, action_seq)
);

-- Critical index for SKIP LOCKED dispatch query
-- Composite index on (scheduled_at, action_seq) for efficient ordering
CREATE INDEX idx_action_queue_dispatch
    ON action_queue(scheduled_at, action_seq)
    WHERE status = 'queued';

-- Index for finding timed-out actions
CREATE INDEX idx_action_queue_deadline
    ON action_queue(deadline_at)
    WHERE status = 'dispatched' AND deadline_at IS NOT NULL;

-- Index for retry queries
CREATE INDEX idx_action_queue_retry
    ON action_queue(partition_id, status, attempt_number)
    WHERE status IN ('failed', 'timed_out');

-- Index for instance-based queries
CREATE INDEX idx_action_queue_instance ON action_queue(instance_id);

-- Execution context (accumulated variable bindings during workflow execution)
CREATE TABLE instance_context (
    instance_id UUID PRIMARY KEY REFERENCES workflow_instances(id) ON DELETE CASCADE,
    -- JSON object mapping variable names to serialized values
    context_json JSONB NOT NULL DEFAULT '{}',
    -- Exception information for error handling
    exceptions_json JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Loop iteration state (for tracking progress through loops)
CREATE TABLE loop_state (
    instance_id UUID NOT NULL REFERENCES workflow_instances(id) ON DELETE CASCADE,
    loop_id TEXT NOT NULL,
    current_index INTEGER NOT NULL DEFAULT 0,
    -- Serialized accumulators (map of name -> collected values)
    accumulators BYTEA,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (instance_id, loop_id)
);
