//! Database operations for the scheduler.

use chrono::{DateTime, Utc};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::db;
use crate::rappel_core::backends::{BackendError, BackendResult};

use super::types::{CreateScheduleParams, ScheduleId, ScheduleType, WorkflowSchedule};
use super::utils::{apply_jitter, next_cron_run, next_interval_run};

/// Database handle for scheduler operations.
#[derive(Clone)]
pub struct SchedulerDatabase {
    pool: PgPool,
}

// TODO: Consolidate objects into the regular db manager. Consolidate the functional logic
// of the work we do on an individual "tick" into the backends/postgres.rs implementation. Come
// up with a very simple in-memory representation of the same basic logic in backends/memory.

impl SchedulerDatabase {
    /// Create a new scheduler database handle from an existing pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Connect to the database.
    pub async fn connect(dsn: &str) -> BackendResult<Self> {
        let pool = PgPool::connect(dsn).await?;
        db::run_migrations(&pool).await?;
        Ok(Self { pool })
    }

    /// Get the underlying pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Ensure the schedule schema exists.
    pub async fn ensure_schema(&self) -> BackendResult<()> {
        db::run_migrations(&self.pool).await
    }

    // ========================================================================
    // Schedule Management
    // ========================================================================

    /// Create or update a schedule.
    pub async fn upsert_schedule(
        &self,
        params: &CreateScheduleParams,
    ) -> BackendResult<ScheduleId> {
        // Calculate initial next_run_at
        let next_run_at = self.compute_next_run(
            params.schedule_type,
            params.cron_expression.as_deref(),
            params.interval_seconds,
            params.jitter_seconds,
            None,
        )?;

        let row = sqlx::query(
            r#"
            INSERT INTO workflow_schedules
                (workflow_name, schedule_name, schedule_type, cron_expression, interval_seconds,
                 jitter_seconds, input_payload, next_run_at, priority, allow_duplicate)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (workflow_name, schedule_name)
            DO UPDATE SET
                schedule_type = EXCLUDED.schedule_type,
                cron_expression = EXCLUDED.cron_expression,
                interval_seconds = EXCLUDED.interval_seconds,
                jitter_seconds = EXCLUDED.jitter_seconds,
                input_payload = EXCLUDED.input_payload,
                next_run_at = EXCLUDED.next_run_at,
                priority = EXCLUDED.priority,
                allow_duplicate = EXCLUDED.allow_duplicate,
                status = 'active',
                updated_at = NOW()
            RETURNING id
            "#,
        )
        .bind(&params.workflow_name)
        .bind(&params.schedule_name)
        .bind(params.schedule_type.as_str())
        .bind(&params.cron_expression)
        .bind(params.interval_seconds)
        .bind(params.jitter_seconds)
        .bind(&params.input_payload)
        .bind(next_run_at)
        .bind(params.priority)
        .bind(params.allow_duplicate)
        .fetch_one(&self.pool)
        .await?;

        let id: Uuid = row.get("id");
        Ok(ScheduleId(id))
    }

    /// Get a schedule by ID.
    pub async fn get_schedule(&self, id: ScheduleId) -> BackendResult<WorkflowSchedule> {
        let schedule = sqlx::query_as::<_, ScheduleRow>(
            r#"
            SELECT id, workflow_name, schedule_name, schedule_type, cron_expression, interval_seconds,
                   jitter_seconds, input_payload, status, next_run_at, last_run_at, last_instance_id,
                   created_at, updated_at, priority, allow_duplicate
            FROM workflow_schedules
            WHERE id = $1
            "#,
        )
        .bind(id.0)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| BackendError::Message(format!("schedule not found: {}", id)))?;

        Ok(schedule.into())
    }

    /// Get a schedule by workflow name and schedule name.
    pub async fn get_schedule_by_name(
        &self,
        workflow_name: &str,
        schedule_name: &str,
    ) -> BackendResult<Option<WorkflowSchedule>> {
        let schedule = sqlx::query_as::<_, ScheduleRow>(
            r#"
            SELECT id, workflow_name, schedule_name, schedule_type, cron_expression, interval_seconds,
                   jitter_seconds, input_payload, status, next_run_at, last_run_at, last_instance_id,
                   created_at, updated_at, priority, allow_duplicate
            FROM workflow_schedules
            WHERE workflow_name = $1 AND schedule_name = $2 AND status != 'deleted'
            "#,
        )
        .bind(workflow_name)
        .bind(schedule_name)
        .fetch_optional(&self.pool)
        .await?;

        Ok(schedule.map(|s| s.into()))
    }

    /// List all schedules (excluding deleted).
    pub async fn list_schedules(
        &self,
        limit: i64,
        offset: i64,
    ) -> BackendResult<Vec<WorkflowSchedule>> {
        let rows = sqlx::query_as::<_, ScheduleRow>(
            r#"
            SELECT id, workflow_name, schedule_name, schedule_type, cron_expression, interval_seconds,
                   jitter_seconds, input_payload, status, next_run_at, last_run_at, last_instance_id,
                   created_at, updated_at, priority, allow_duplicate
            FROM workflow_schedules
            WHERE status != 'deleted'
            ORDER BY workflow_name, schedule_name
            LIMIT $1 OFFSET $2
            "#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| r.into()).collect())
    }

    /// Count schedules (excluding deleted).
    pub async fn count_schedules(&self) -> BackendResult<i64> {
        let count = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM workflow_schedules WHERE status != 'deleted'",
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(count)
    }

    /// Update schedule status.
    pub async fn update_schedule_status(
        &self,
        id: ScheduleId,
        status: &str,
    ) -> BackendResult<bool> {
        let result = sqlx::query(
            r#"
            UPDATE workflow_schedules
            SET status = $2, updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(id.0)
        .bind(status)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Delete a schedule (soft delete).
    pub async fn delete_schedule(&self, id: ScheduleId) -> BackendResult<bool> {
        self.update_schedule_status(id, "deleted").await
    }

    // ========================================================================
    // Scheduler Loop Operations
    // ========================================================================

    /// Find schedules that are due to run.
    /// Uses FOR UPDATE SKIP LOCKED for multi-runner safety.
    pub async fn find_due_schedules(&self, limit: i32) -> BackendResult<Vec<WorkflowSchedule>> {
        let rows = sqlx::query_as::<_, ScheduleRow>(
            r#"
            SELECT id, workflow_name, schedule_name, schedule_type, cron_expression, interval_seconds,
                   jitter_seconds, input_payload, status, next_run_at, last_run_at, last_instance_id,
                   created_at, updated_at, priority, allow_duplicate
            FROM workflow_schedules
            WHERE status = 'active'
              AND next_run_at IS NOT NULL
              AND next_run_at <= NOW()
            ORDER BY next_run_at
            FOR UPDATE SKIP LOCKED
            LIMIT $1
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|r| r.into()).collect())
    }

    /// Check if there's a running instance for this schedule (for duplicate check).
    pub async fn has_running_instance(&self, _schedule_id: ScheduleId) -> BackendResult<bool> {
        // Check in runner_instances for any instance that was queued for this schedule
        // We don't have schedule_id in runner_instances, so this is a simplified check
        // In production you'd want to track this properly
        Ok(false)
    }

    /// Mark a schedule as executed and compute the next run time.
    pub async fn mark_schedule_executed(
        &self,
        schedule_id: ScheduleId,
        instance_id: Uuid,
    ) -> BackendResult<()> {
        // Get the schedule to compute next run time
        let schedule = self.get_schedule(schedule_id).await?;
        let schedule_type = ScheduleType::parse(&schedule.schedule_type)
            .ok_or_else(|| BackendError::Message("invalid schedule type".to_string()))?;

        let next_run_at = self.compute_next_run(
            schedule_type,
            schedule.cron_expression.as_deref(),
            schedule.interval_seconds,
            schedule.jitter_seconds,
            Some(Utc::now()),
        )?;

        sqlx::query(
            r#"
            UPDATE workflow_schedules
            SET last_run_at = NOW(),
                last_instance_id = $2,
                next_run_at = $3,
                updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(schedule_id.0)
        .bind(instance_id)
        .bind(next_run_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Update next_run_at without creating an instance.
    /// Used when skipping a scheduled run.
    pub async fn skip_schedule_run(&self, schedule_id: ScheduleId) -> BackendResult<()> {
        let schedule = self.get_schedule(schedule_id).await?;
        let schedule_type = ScheduleType::parse(&schedule.schedule_type)
            .ok_or_else(|| BackendError::Message("invalid schedule type".to_string()))?;

        let next_run_at = self.compute_next_run(
            schedule_type,
            schedule.cron_expression.as_deref(),
            schedule.interval_seconds,
            schedule.jitter_seconds,
            Some(Utc::now()),
        )?;

        sqlx::query(
            r#"
            UPDATE workflow_schedules
            SET next_run_at = $2, updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(schedule_id.0)
        .bind(next_run_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    fn compute_next_run(
        &self,
        schedule_type: ScheduleType,
        cron_expression: Option<&str>,
        interval_seconds: Option<i64>,
        jitter_seconds: i64,
        last_run_at: Option<DateTime<Utc>>,
    ) -> BackendResult<DateTime<Utc>> {
        let base = match schedule_type {
            ScheduleType::Cron => {
                let expr = cron_expression
                    .ok_or_else(|| BackendError::Message("cron expression required".to_string()))?;
                next_cron_run(expr).map_err(BackendError::Message)?
            }
            ScheduleType::Interval => {
                let seconds = interval_seconds.ok_or_else(|| {
                    BackendError::Message("interval_seconds required".to_string())
                })?;
                next_interval_run(seconds, last_run_at)
            }
        };

        apply_jitter(base, jitter_seconds).map_err(BackendError::Message)
    }
}

// Internal row type for sqlx
#[derive(sqlx::FromRow)]
struct ScheduleRow {
    id: Uuid,
    workflow_name: String,
    schedule_name: String,
    schedule_type: String,
    cron_expression: Option<String>,
    interval_seconds: Option<i64>,
    jitter_seconds: i64,
    input_payload: Option<Vec<u8>>,
    status: String,
    next_run_at: Option<DateTime<Utc>>,
    last_run_at: Option<DateTime<Utc>>,
    last_instance_id: Option<Uuid>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    priority: i32,
    allow_duplicate: bool,
}

impl From<ScheduleRow> for WorkflowSchedule {
    fn from(row: ScheduleRow) -> Self {
        Self {
            id: row.id,
            workflow_name: row.workflow_name,
            schedule_name: row.schedule_name,
            schedule_type: row.schedule_type,
            cron_expression: row.cron_expression,
            interval_seconds: row.interval_seconds,
            jitter_seconds: row.jitter_seconds,
            input_payload: row.input_payload,
            status: row.status,
            next_run_at: row.next_run_at,
            last_run_at: row.last_run_at,
            last_instance_id: row.last_instance_id,
            created_at: row.created_at,
            updated_at: row.updated_at,
            priority: row.priority,
            allow_duplicate: row.allow_duplicate,
        }
    }
}
