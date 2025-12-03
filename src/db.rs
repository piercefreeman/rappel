//! Database layer for Rappel workflow execution.
//!
//! Uses PostgreSQL with sqlx for type-safe queries. Key features:
//! - Distributed work queue with `SKIP LOCKED` for non-blocking dispatch
//! - Workflow version caching
//! - Atomic action completion with delivery token validation
//!
//! # Connection
//!
//! Set the `DATABASE_URL` environment variable to your PostgreSQL connection string:
//! ```text
//! DATABASE_URL=postgresql://user:password@localhost:5432/rappel
//! ```

use chrono::{DateTime, Utc};
use sqlx::{
    FromRow, PgPool, Row,
    postgres::PgPoolOptions,
};
use thiserror::Error;
use uuid::Uuid;

// ============================================================================
// Type Aliases & Newtypes
// ============================================================================

/// Unique identifier for a workflow version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkflowVersionId(pub Uuid);

impl WorkflowVersionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for WorkflowVersionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for WorkflowVersionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a workflow instance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkflowInstanceId(pub Uuid);

impl WorkflowInstanceId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for WorkflowInstanceId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for WorkflowInstanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for an action in the queue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionId(pub Uuid);

impl ActionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ActionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ActionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ============================================================================
// Status Enums
// ============================================================================

/// Status of a workflow instance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceStatus {
    Running,
    Completed,
    Failed,
}

impl InstanceStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "running" => Some(Self::Running),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

/// Status of an action in the queue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionStatus {
    Queued,
    Dispatched,
    Completed,
    Failed,
    TimedOut,
}

impl ActionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Dispatched => "dispatched",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "queued" => Some(Self::Queued),
            "dispatched" => Some(Self::Dispatched),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "timed_out" => Some(Self::TimedOut),
            _ => None,
        }
    }
}

/// Type of retry being attempted
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryKind {
    Failure,
    Timeout,
}

impl RetryKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Failure => "failure",
            Self::Timeout => "timeout",
        }
    }
}

/// Backoff strategy for retries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackoffKind {
    None,
    Linear,
    Exponential,
}

impl BackoffKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Exponential => "exponential",
        }
    }
}

// ============================================================================
// Model Structs
// ============================================================================

/// A workflow version (compiled program definition)
#[derive(Debug, Clone, FromRow)]
pub struct WorkflowVersion {
    pub id: Uuid,
    pub workflow_name: String,
    pub dag_hash: String,
    pub program_proto: Vec<u8>,
    pub concurrent: bool,
    pub created_at: DateTime<Utc>,
}

/// Summary of a workflow version (without the proto payload)
#[derive(Debug, Clone, FromRow)]
pub struct WorkflowVersionSummary {
    pub id: Uuid,
    pub workflow_name: String,
    pub dag_hash: String,
    pub concurrent: bool,
    pub created_at: DateTime<Utc>,
}

/// A workflow instance (execution)
#[derive(Debug, Clone, FromRow)]
pub struct WorkflowInstance {
    pub id: Uuid,
    pub partition_id: i32,
    pub workflow_name: String,
    pub workflow_version_id: Option<Uuid>,
    pub next_action_seq: i32,
    pub input_payload: Option<Vec<u8>>,
    pub result_payload: Option<Vec<u8>>,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// An action ready for dispatch (returned from dispatch_actions)
#[derive(Debug, Clone)]
pub struct QueuedAction {
    pub id: Uuid,
    pub instance_id: Uuid,
    pub partition_id: i32,
    pub action_seq: i32,
    pub module_name: String,
    pub action_name: String,
    pub dispatch_payload: Vec<u8>,
    pub timeout_seconds: i32,
    pub max_retries: i32,
    pub attempt_number: i32,
    pub delivery_token: Uuid,
    pub timeout_retry_limit: i32,
    pub retry_kind: String,
}

/// Record for completing an action
#[derive(Debug, Clone)]
pub struct CompletionRecord {
    pub action_id: ActionId,
    pub success: bool,
    pub result_payload: Vec<u8>,
    pub delivery_token: Uuid,
    pub error_message: Option<String>,
}

/// New action to enqueue
#[derive(Debug, Clone)]
pub struct NewAction {
    pub instance_id: WorkflowInstanceId,
    pub module_name: String,
    pub action_name: String,
    pub dispatch_payload: Vec<u8>,
    pub timeout_seconds: i32,
    pub max_retries: i32,
    pub backoff_kind: BackoffKind,
    pub backoff_base_delay_ms: i32,
    pub node_id: Option<String>,
}

/// Instance execution context
#[derive(Debug, Clone, FromRow)]
pub struct InstanceContext {
    pub instance_id: Uuid,
    pub context_json: serde_json::Value,
    pub exceptions_json: serde_json::Value,
    pub updated_at: DateTime<Utc>,
}

/// Loop iteration state
#[derive(Debug, Clone, FromRow)]
pub struct LoopState {
    pub instance_id: Uuid,
    pub loop_id: String,
    pub current_index: i32,
    pub accumulators: Option<Vec<u8>>,
    pub updated_at: DateTime<Utc>,
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum DbError {
    #[error("Database error: {0}")]
    Sqlx(#[from] sqlx::Error),

    #[error("Migration error: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid delivery token")]
    InvalidDeliveryToken,
}

pub type DbResult<T> = Result<T, DbError>;

// ============================================================================
// Database
// ============================================================================

/// Main database handle
#[derive(Clone)]
pub struct Database {
    pool: PgPool,
}

impl Database {
    /// Connect to the database and run migrations
    pub async fn connect(database_url: &str) -> DbResult<Self> {
        Self::connect_with_pool_size(database_url, 10).await
    }

    /// Connect with a custom pool size
    pub async fn connect_with_pool_size(database_url: &str, max_connections: u32) -> DbResult<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(max_connections)
            .connect(database_url)
            .await?;

        // Run migrations
        sqlx::migrate!("./migrations").run(&pool).await?;

        Ok(Self { pool })
    }

    /// Get a reference to the connection pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    // ========================================================================
    // Workflow Versions
    // ========================================================================

    /// Create or update a workflow version
    /// Returns the version ID (existing if hash matches, new otherwise)
    pub async fn upsert_workflow_version(
        &self,
        workflow_name: &str,
        dag_hash: &str,
        program_proto: &[u8],
        concurrent: bool,
    ) -> DbResult<WorkflowVersionId> {
        let row = sqlx::query(
            r#"
            INSERT INTO workflow_versions (workflow_name, dag_hash, program_proto, concurrent)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (workflow_name, dag_hash) DO UPDATE SET workflow_name = EXCLUDED.workflow_name
            RETURNING id
            "#
        )
        .bind(workflow_name)
        .bind(dag_hash)
        .bind(program_proto)
        .bind(concurrent)
        .fetch_one(&self.pool)
        .await?;

        let id: Uuid = row.get("id");
        Ok(WorkflowVersionId(id))
    }

    /// Load a workflow version by ID
    pub async fn get_workflow_version(&self, id: WorkflowVersionId) -> DbResult<WorkflowVersion> {
        let version = sqlx::query_as::<_, WorkflowVersion>(
            r#"
            SELECT id, workflow_name, dag_hash, program_proto, concurrent, created_at
            FROM workflow_versions
            WHERE id = $1
            "#
        )
        .bind(id.0)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| DbError::NotFound(format!("workflow version {}", id)))?;

        Ok(version)
    }

    /// List all workflow versions
    pub async fn list_workflow_versions(&self) -> DbResult<Vec<WorkflowVersionSummary>> {
        let versions = sqlx::query_as::<_, WorkflowVersionSummary>(
            r#"
            SELECT id, workflow_name, dag_hash, concurrent, created_at
            FROM workflow_versions
            ORDER BY created_at DESC
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        Ok(versions)
    }

    // ========================================================================
    // Workflow Instances
    // ========================================================================

    /// Create a new workflow instance
    pub async fn create_instance(
        &self,
        workflow_name: &str,
        version_id: WorkflowVersionId,
        input_payload: Option<&[u8]>,
    ) -> DbResult<WorkflowInstanceId> {
        let row = sqlx::query(
            r#"
            INSERT INTO workflow_instances (workflow_name, workflow_version_id, input_payload)
            VALUES ($1, $2, $3)
            RETURNING id
            "#
        )
        .bind(workflow_name)
        .bind(version_id.0)
        .bind(input_payload)
        .fetch_one(&self.pool)
        .await?;

        let id: Uuid = row.get("id");

        // Initialize context
        sqlx::query(
            r#"
            INSERT INTO instance_context (instance_id)
            VALUES ($1)
            "#
        )
        .bind(id)
        .execute(&self.pool)
        .await?;

        Ok(WorkflowInstanceId(id))
    }

    /// Get a workflow instance by ID
    pub async fn get_instance(&self, id: WorkflowInstanceId) -> DbResult<WorkflowInstance> {
        let instance = sqlx::query_as::<_, WorkflowInstance>(
            r#"
            SELECT id, partition_id, workflow_name, workflow_version_id,
                   next_action_seq, input_payload, result_payload, status,
                   created_at, completed_at
            FROM workflow_instances
            WHERE id = $1
            "#
        )
        .bind(id.0)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| DbError::NotFound(format!("workflow instance {}", id)))?;

        Ok(instance)
    }

    /// Mark an instance as completed
    pub async fn complete_instance(
        &self,
        id: WorkflowInstanceId,
        result_payload: Option<&[u8]>,
    ) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE workflow_instances
            SET status = 'completed', result_payload = $2, completed_at = NOW()
            WHERE id = $1
            "#
        )
        .bind(id.0)
        .bind(result_payload)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Mark an instance as failed
    pub async fn fail_instance(&self, id: WorkflowInstanceId) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE workflow_instances
            SET status = 'failed', completed_at = NOW()
            WHERE id = $1
            "#
        )
        .bind(id.0)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    // ========================================================================
    // Action Queue
    // ========================================================================

    /// Enqueue a new action
    pub async fn enqueue_action(&self, action: NewAction) -> DbResult<ActionId> {
        // Get and increment sequence number atomically
        let row = sqlx::query(
            r#"
            UPDATE workflow_instances
            SET next_action_seq = next_action_seq + 1
            WHERE id = $1
            RETURNING next_action_seq - 1
            "#
        )
        .bind(action.instance_id.0)
        .fetch_one(&self.pool)
        .await?;

        let seq: i32 = row.get(0);

        let row = sqlx::query(
            r#"
            INSERT INTO action_queue (
                instance_id, action_seq, module_name, action_name,
                dispatch_payload, timeout_seconds, max_retries,
                backoff_kind, backoff_base_delay_ms, node_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            "#
        )
        .bind(action.instance_id.0)
        .bind(seq)
        .bind(&action.module_name)
        .bind(&action.action_name)
        .bind(&action.dispatch_payload)
        .bind(action.timeout_seconds)
        .bind(action.max_retries)
        .bind(action.backoff_kind.as_str())
        .bind(action.backoff_base_delay_ms)
        .bind(&action.node_id)
        .fetch_one(&self.pool)
        .await?;

        let id: Uuid = row.get("id");
        Ok(ActionId(id))
    }

    /// Dispatch actions from the queue using SKIP LOCKED
    ///
    /// This is the core distributed queue operation. It atomically:
    /// 1. Selects up to `limit` queued actions that are ready
    /// 2. Locks them with FOR UPDATE SKIP LOCKED (non-blocking)
    /// 3. Updates their status to 'dispatched'
    /// 4. Sets deadline and delivery token
    /// 5. Returns the actions for execution
    pub async fn dispatch_actions(&self, limit: i32) -> DbResult<Vec<QueuedAction>> {
        let rows = sqlx::query(
            r#"
            WITH next_actions AS (
                SELECT id
                FROM action_queue
                WHERE status = 'queued'
                  AND scheduled_at <= NOW()
                ORDER BY scheduled_at, action_seq
                FOR UPDATE SKIP LOCKED
                LIMIT $1
            )
            UPDATE action_queue aq
            SET status = 'dispatched',
                dispatched_at = NOW(),
                deadline_at = CASE
                    WHEN timeout_seconds > 0
                    THEN NOW() + (timeout_seconds || ' seconds')::interval
                    ELSE NULL
                END,
                delivery_token = gen_random_uuid()
            FROM next_actions
            WHERE aq.id = next_actions.id
            RETURNING
                aq.id,
                aq.instance_id,
                aq.partition_id,
                aq.action_seq,
                aq.module_name,
                aq.action_name,
                aq.dispatch_payload,
                aq.timeout_seconds,
                aq.max_retries,
                aq.attempt_number,
                aq.delivery_token,
                aq.timeout_retry_limit,
                aq.retry_kind
            "#
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        let actions = rows.into_iter().map(|row| {
            QueuedAction {
                id: row.get("id"),
                instance_id: row.get("instance_id"),
                partition_id: row.get("partition_id"),
                action_seq: row.get("action_seq"),
                module_name: row.get("module_name"),
                action_name: row.get("action_name"),
                dispatch_payload: row.get("dispatch_payload"),
                timeout_seconds: row.get("timeout_seconds"),
                max_retries: row.get("max_retries"),
                attempt_number: row.get("attempt_number"),
                delivery_token: row.get("delivery_token"),
                timeout_retry_limit: row.get("timeout_retry_limit"),
                retry_kind: row.get("retry_kind"),
            }
        }).collect();

        Ok(actions)
    }

    /// Complete an action with its result
    ///
    /// Uses delivery_token for idempotent completion - if the token doesn't match,
    /// the action was already completed by another worker or timed out.
    pub async fn complete_action(&self, record: CompletionRecord) -> DbResult<bool> {
        let result = sqlx::query(
            r#"
            UPDATE action_queue
            SET status = CASE WHEN $2 THEN 'completed' ELSE 'failed' END,
                success = $2,
                result_payload = $3,
                last_error = $4,
                completed_at = NOW()
            WHERE id = $1 AND delivery_token = $5 AND status = 'dispatched'
            "#
        )
        .bind(record.action_id.0)
        .bind(record.success)
        .bind(&record.result_payload)
        .bind(&record.error_message)
        .bind(record.delivery_token)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Find and mark timed-out actions for retry
    ///
    /// Returns the number of actions that were marked for retry
    pub async fn mark_timed_out_actions(&self, limit: i32) -> DbResult<i64> {
        let result = sqlx::query(
            r#"
            WITH overdue AS (
                SELECT id
                FROM action_queue
                WHERE status = 'dispatched'
                  AND deadline_at IS NOT NULL
                  AND deadline_at < NOW()
                FOR UPDATE SKIP LOCKED
                LIMIT $1
            )
            UPDATE action_queue aq
            SET status = 'timed_out',
                retry_kind = 'timeout'
            FROM overdue
            WHERE aq.id = overdue.id
            "#
        )
        .bind(limit)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() as i64)
    }

    /// Requeue failed or timed-out actions for retry
    ///
    /// Calculates backoff delay based on policy and schedules retry
    pub async fn requeue_failed_actions(&self, limit: i32) -> DbResult<i64> {
        let result = sqlx::query(
            r#"
            WITH retryable AS (
                SELECT id
                FROM action_queue
                WHERE (
                    (status = 'failed' AND retry_kind = 'failure' AND attempt_number < max_retries)
                    OR
                    (status = 'timed_out' AND retry_kind = 'timeout' AND attempt_number < timeout_retry_limit)
                )
                FOR UPDATE SKIP LOCKED
                LIMIT $1
            )
            UPDATE action_queue aq
            SET status = 'queued',
                attempt_number = aq.attempt_number + 1,
                scheduled_at = NOW() + (
                    CASE aq.backoff_kind
                        WHEN 'linear' THEN (aq.backoff_base_delay_ms * (aq.attempt_number + 1))
                        WHEN 'exponential' THEN (aq.backoff_base_delay_ms * POWER(aq.backoff_multiplier, aq.attempt_number))
                        ELSE 0
                    END || ' milliseconds'
                )::interval,
                deadline_at = NULL,
                delivery_token = NULL
            FROM retryable
            WHERE aq.id = retryable.id
            "#
        )
        .bind(limit)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() as i64)
    }

    /// Get all actions for an instance
    pub async fn get_instance_actions(&self, instance_id: WorkflowInstanceId) -> DbResult<Vec<QueuedAction>> {
        let rows = sqlx::query(
            r#"
            SELECT
                id,
                instance_id,
                partition_id,
                action_seq,
                module_name,
                action_name,
                dispatch_payload,
                timeout_seconds,
                max_retries,
                attempt_number,
                COALESCE(delivery_token, gen_random_uuid()) as delivery_token,
                timeout_retry_limit,
                retry_kind
            FROM action_queue
            WHERE instance_id = $1
            ORDER BY action_seq
            "#
        )
        .bind(instance_id.0)
        .fetch_all(&self.pool)
        .await?;

        let actions = rows.into_iter().map(|row| {
            QueuedAction {
                id: row.get("id"),
                instance_id: row.get("instance_id"),
                partition_id: row.get("partition_id"),
                action_seq: row.get("action_seq"),
                module_name: row.get("module_name"),
                action_name: row.get("action_name"),
                dispatch_payload: row.get("dispatch_payload"),
                timeout_seconds: row.get("timeout_seconds"),
                max_retries: row.get("max_retries"),
                attempt_number: row.get("attempt_number"),
                delivery_token: row.get("delivery_token"),
                timeout_retry_limit: row.get("timeout_retry_limit"),
                retry_kind: row.get("retry_kind"),
            }
        }).collect();

        Ok(actions)
    }

    // ========================================================================
    // Instance Context
    // ========================================================================

    /// Get the execution context for an instance
    pub async fn get_instance_context(&self, instance_id: WorkflowInstanceId) -> DbResult<InstanceContext> {
        let ctx = sqlx::query_as::<_, InstanceContext>(
            r#"
            SELECT instance_id, context_json, exceptions_json, updated_at
            FROM instance_context
            WHERE instance_id = $1
            "#
        )
        .bind(instance_id.0)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| DbError::NotFound(format!("context for instance {}", instance_id)))?;

        Ok(ctx)
    }

    /// Update the execution context for an instance
    pub async fn update_instance_context(
        &self,
        instance_id: WorkflowInstanceId,
        context_json: serde_json::Value,
    ) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE instance_context
            SET context_json = $2, updated_at = NOW()
            WHERE instance_id = $1
            "#
        )
        .bind(instance_id.0)
        .bind(context_json)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    // ========================================================================
    // Loop State
    // ========================================================================

    /// Get or create loop state
    pub async fn get_or_create_loop_state(
        &self,
        instance_id: WorkflowInstanceId,
        loop_id: &str,
    ) -> DbResult<LoopState> {
        let state = sqlx::query_as::<_, LoopState>(
            r#"
            INSERT INTO loop_state (instance_id, loop_id)
            VALUES ($1, $2)
            ON CONFLICT (instance_id, loop_id) DO UPDATE SET loop_id = EXCLUDED.loop_id
            RETURNING instance_id, loop_id, current_index, accumulators, updated_at
            "#
        )
        .bind(instance_id.0)
        .bind(loop_id)
        .fetch_one(&self.pool)
        .await?;

        Ok(state)
    }

    /// Increment loop index and return new value
    pub async fn increment_loop_index(
        &self,
        instance_id: WorkflowInstanceId,
        loop_id: &str,
    ) -> DbResult<i32> {
        let row = sqlx::query(
            r#"
            UPDATE loop_state
            SET current_index = current_index + 1, updated_at = NOW()
            WHERE instance_id = $1 AND loop_id = $2
            RETURNING current_index
            "#
        )
        .bind(instance_id.0)
        .bind(loop_id)
        .fetch_one(&self.pool)
        .await?;

        let new_index: i32 = row.get(0);
        Ok(new_index)
    }

    /// Update loop accumulators
    pub async fn update_loop_accumulators(
        &self,
        instance_id: WorkflowInstanceId,
        loop_id: &str,
        accumulators: &[u8],
    ) -> DbResult<()> {
        sqlx::query(
            r#"
            UPDATE loop_state
            SET accumulators = $3, updated_at = NOW()
            WHERE instance_id = $1 AND loop_id = $2
            "#
        )
        .bind(instance_id.0)
        .bind(loop_id)
        .bind(accumulators)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_roundtrip() {
        assert_eq!(ActionStatus::from_str(ActionStatus::Queued.as_str()), Some(ActionStatus::Queued));
        assert_eq!(ActionStatus::from_str(ActionStatus::Dispatched.as_str()), Some(ActionStatus::Dispatched));
        assert_eq!(ActionStatus::from_str(ActionStatus::Completed.as_str()), Some(ActionStatus::Completed));
        assert_eq!(ActionStatus::from_str(ActionStatus::Failed.as_str()), Some(ActionStatus::Failed));
        assert_eq!(ActionStatus::from_str(ActionStatus::TimedOut.as_str()), Some(ActionStatus::TimedOut));
    }

    #[test]
    fn test_instance_status_roundtrip() {
        assert_eq!(InstanceStatus::from_str(InstanceStatus::Running.as_str()), Some(InstanceStatus::Running));
        assert_eq!(InstanceStatus::from_str(InstanceStatus::Completed.as_str()), Some(InstanceStatus::Completed));
        assert_eq!(InstanceStatus::from_str(InstanceStatus::Failed.as_str()), Some(InstanceStatus::Failed));
    }

    #[test]
    fn test_id_display() {
        let id = WorkflowInstanceId::new();
        let s = id.to_string();
        assert!(!s.is_empty());
    }
}
