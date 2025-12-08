//! Database layer for workflow persistence.
//!
//! Uses PostgreSQL with:
//! - Workflow instances table
//! - Action inbox/queue per instance
//! - SELECT FOR UPDATE for queue locking

use chrono::{DateTime, Utc};
use sqlx::postgres::PgPool;
use sqlx::FromRow;
use uuid::Uuid;

/// Action status in the queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "action_status", rename_all = "lowercase")]
pub enum ActionStatus {
    Queued,
    Dispatched,
    Completed,
    Failed,
}

/// Workflow instance status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "instance_status", rename_all = "lowercase")]
pub enum InstanceStatus {
    Running,
    WaitingForActions,
    Completed,
    Failed,
}

/// A workflow instance record.
#[derive(Debug, Clone, FromRow)]
pub struct WorkflowInstance {
    pub id: Uuid,
    pub workflow_name: String,
    pub module_name: String,
    pub status: InstanceStatus,
    pub initial_args: serde_json::Value,
    pub result: Option<serde_json::Value>,
    pub actions_until_index: i32,
    pub scheduled_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// An action in the instance's inbox/queue.
#[derive(Debug, Clone, FromRow)]
pub struct ActionRecord {
    pub id: Uuid,
    pub instance_id: Uuid,
    pub sequence: i32,
    pub action_name: String,
    pub module_name: String,
    pub kwargs: serde_json::Value,
    pub status: ActionStatus,
    pub result: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub dispatch_token: Option<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Database operations for workflow management.
pub struct Database {
    pool: PgPool,
}

impl Database {
    /// Create a new database connection.
    pub async fn connect(database_url: &str) -> anyhow::Result<Self> {
        let pool = PgPool::connect(database_url).await?;
        Ok(Database { pool })
    }

    /// Run database migrations.
    pub async fn migrate(&self) -> anyhow::Result<()> {
        sqlx::migrate!("./migrations").run(&self.pool).await?;
        Ok(())
    }

    /// Create a new workflow instance.
    pub async fn create_instance(
        &self,
        workflow_name: &str,
        module_name: &str,
        initial_args: serde_json::Value,
    ) -> anyhow::Result<Uuid> {
        let id = Uuid::new_v4();
        sqlx::query(
            r#"
            INSERT INTO workflow_instances (id, workflow_name, module_name, status, initial_args)
            VALUES ($1, $2, $3, 'running', $4)
            "#,
        )
        .bind(id)
        .bind(workflow_name)
        .bind(module_name)
        .bind(&initial_args)
        .execute(&self.pool)
        .await?;

        Ok(id)
    }

    /// Get an instance by ID.
    pub async fn get_instance(&self, id: Uuid) -> anyhow::Result<Option<WorkflowInstance>> {
        let row: Option<WorkflowInstance> = sqlx::query_as(
            r#"
            SELECT
                id, workflow_name, module_name,
                status,
                initial_args, result,
                actions_until_index,
                scheduled_at,
                created_at, updated_at
            FROM workflow_instances
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        Ok(row)
    }

    /// Get instances that are waiting for actions and ready to run.
    /// Uses FOR UPDATE SKIP LOCKED for distributed queue semantics.
    pub async fn claim_ready_instances(&self, limit: i32) -> anyhow::Result<Vec<WorkflowInstance>> {
        // Claim instances where:
        // 1. Status is 'waiting_for_actions'
        // 2. scheduled_at is NULL or <= now
        let rows: Vec<WorkflowInstance> = sqlx::query_as(
            r#"
            SELECT
                id, workflow_name, module_name,
                status,
                initial_args, result,
                actions_until_index,
                scheduled_at,
                created_at, updated_at
            FROM workflow_instances
            WHERE status = 'waiting_for_actions'
              AND (scheduled_at IS NULL OR scheduled_at <= NOW())
            ORDER BY scheduled_at ASC NULLS FIRST, created_at ASC
            LIMIT $1
            FOR UPDATE SKIP LOCKED
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Update instance status and schedule next run.
    pub async fn update_instance_status(
        &self,
        id: Uuid,
        status: InstanceStatus,
        actions_until_index: i32,
        scheduled_at: Option<DateTime<Utc>>,
    ) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            UPDATE workflow_instances
            SET status = $2, actions_until_index = $3, scheduled_at = $4, updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(status)
        .bind(actions_until_index)
        .bind(scheduled_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Complete an instance with a result.
    pub async fn complete_instance(
        &self,
        id: Uuid,
        result: serde_json::Value,
    ) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            UPDATE workflow_instances
            SET status = 'completed', result = $2, updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(&result)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Fail an instance with an error message.
    pub async fn fail_instance(&self, id: Uuid, error: &str) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            UPDATE workflow_instances
            SET status = 'failed', result = $2, updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(serde_json::json!({ "error": error }))
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Add actions to an instance's inbox.
    pub async fn add_actions(
        &self,
        instance_id: Uuid,
        actions: Vec<(String, String, serde_json::Value)>, // (action_name, module_name, kwargs)
        start_sequence: i32,
    ) -> anyhow::Result<Vec<Uuid>> {
        let mut ids = Vec::with_capacity(actions.len());

        for (i, (action_name, module_name, kwargs)) in actions.into_iter().enumerate() {
            let id = Uuid::new_v4();
            let sequence = start_sequence + i as i32;

            sqlx::query(
                r#"
                INSERT INTO actions (id, instance_id, sequence, action_name, module_name, kwargs, status)
                VALUES ($1, $2, $3, $4, $5, $6, 'queued')
                "#,
            )
            .bind(id)
            .bind(instance_id)
            .bind(sequence)
            .bind(&action_name)
            .bind(&module_name)
            .bind(&kwargs)
            .execute(&self.pool)
            .await?;

            ids.push(id);
        }

        Ok(ids)
    }

    /// Claim queued actions for execution.
    /// Uses FOR UPDATE SKIP LOCKED for distributed queue semantics.
    pub async fn claim_queued_actions(&self, limit: i32) -> anyhow::Result<Vec<ActionRecord>> {
        let rows: Vec<ActionRecord> = sqlx::query_as(
            r#"
            SELECT
                id, instance_id, sequence, action_name, module_name, kwargs,
                status,
                result, error_message, dispatch_token,
                created_at, updated_at
            FROM actions
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT $1
            FOR UPDATE SKIP LOCKED
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Mark an action as dispatched.
    pub async fn dispatch_action(&self, id: Uuid, dispatch_token: Uuid) -> anyhow::Result<()> {
        sqlx::query(
            r#"
            UPDATE actions
            SET status = 'dispatched', dispatch_token = $2, updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(dispatch_token)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Complete an action with a result.
    pub async fn complete_action(
        &self,
        id: Uuid,
        dispatch_token: Uuid,
        result: serde_json::Value,
    ) -> anyhow::Result<bool> {
        // Use dispatch_token to ensure idempotent completion
        let result = sqlx::query(
            r#"
            UPDATE actions
            SET status = 'completed', result = $3, updated_at = NOW()
            WHERE id = $1 AND dispatch_token = $2 AND status = 'dispatched'
            "#,
        )
        .bind(id)
        .bind(dispatch_token)
        .bind(&result)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Fail an action with an error.
    pub async fn fail_action(
        &self,
        id: Uuid,
        dispatch_token: Uuid,
        error_message: &str,
    ) -> anyhow::Result<bool> {
        let result = sqlx::query(
            r#"
            UPDATE actions
            SET status = 'failed', error_message = $3, updated_at = NOW()
            WHERE id = $1 AND dispatch_token = $2 AND status = 'dispatched'
            "#,
        )
        .bind(id)
        .bind(dispatch_token)
        .bind(error_message)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Get all completed actions for an instance up to a sequence number.
    pub async fn get_completed_actions(
        &self,
        instance_id: Uuid,
        up_to_sequence: i32,
    ) -> anyhow::Result<Vec<ActionRecord>> {
        let rows: Vec<ActionRecord> = sqlx::query_as(
            r#"
            SELECT
                id, instance_id, sequence, action_name, module_name, kwargs,
                status,
                result, error_message, dispatch_token,
                created_at, updated_at
            FROM actions
            WHERE instance_id = $1 AND sequence < $2 AND status = 'completed'
            ORDER BY sequence ASC
            "#,
        )
        .bind(instance_id)
        .bind(up_to_sequence)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Check if all actions up to a sequence are completed.
    pub async fn all_actions_completed(
        &self,
        instance_id: Uuid,
        up_to_sequence: i32,
    ) -> anyhow::Result<bool> {
        let row: (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*)
            FROM actions
            WHERE instance_id = $1 AND sequence < $2 AND status != 'completed'
            "#,
        )
        .bind(instance_id)
        .bind(up_to_sequence)
        .fetch_one(&self.pool)
        .await?;

        Ok(row.0 == 0)
    }

    /// Get the pool for direct access (e.g., transactions).
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}
