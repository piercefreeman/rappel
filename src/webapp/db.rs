//! Database operations for the webapp.
//!
//! These are read-heavy operations for displaying workflow information in the dashboard.

use chrono::{DateTime, Utc};
use sqlx::{PgPool, Row};
use uuid::Uuid;

use crate::rappel_core::backends::{BackendError, BackendResult, GraphUpdate};
use crate::rappel_core::runner::state::NodeStatus;

use super::types::{
    ExecutionEdgeView, ExecutionGraphView, ExecutionNodeView, InstanceDetail, InstanceStatus,
    InstanceSummary, ScheduleDetail, ScheduleSummary, TimelineEntry,
};

/// Database handle for webapp queries.
#[derive(Clone)]
pub struct WebappDatabase {
    pool: PgPool,
}

impl WebappDatabase {
    /// Create a new webapp database handle from an existing pool.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Connect to the database.
    pub async fn connect(dsn: &str) -> BackendResult<Self> {
        let pool = PgPool::connect(dsn).await?;
        Ok(Self { pool })
    }

    /// Get the underlying pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    // ========================================================================
    // Instance Listing
    // ========================================================================

    /// Count total instances, optionally filtered by search.
    pub async fn count_instances(&self, search: Option<&str>) -> BackendResult<i64> {
        let count = if let Some(_search) = search {
            // For now, simple search not implemented - would need to decode state
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM runner_instances")
                .fetch_one(&self.pool)
                .await?
        } else {
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM runner_instances")
                .fetch_one(&self.pool)
                .await?
        };
        Ok(count)
    }

    /// List instances with pagination.
    pub async fn list_instances(
        &self,
        _search: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> BackendResult<Vec<InstanceSummary>> {
        let rows = sqlx::query(
            r#"
            SELECT instance_id, entry_node, created_at, state, result, error
            FROM runner_instances
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            "#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let mut instances = Vec::new();
        for row in rows {
            let instance_id: Uuid = row.get("instance_id");
            let entry_node: Uuid = row.get("entry_node");
            let created_at: DateTime<Utc> = row.get("created_at");
            let state_bytes: Option<Vec<u8>> = row.get("state");
            let result_bytes: Option<Vec<u8>> = row.get("result");
            let error_bytes: Option<Vec<u8>> = row.get("error");

            let status = determine_status(&state_bytes, &result_bytes, &error_bytes);
            let workflow_name = extract_workflow_name(&state_bytes);
            let input_preview = extract_input_preview(&state_bytes);

            instances.push(InstanceSummary {
                id: instance_id,
                entry_node,
                created_at,
                status,
                workflow_name,
                input_preview,
            });
        }

        Ok(instances)
    }

    /// Get a single instance by ID.
    pub async fn get_instance(&self, instance_id: Uuid) -> BackendResult<InstanceDetail> {
        let row = sqlx::query(
            r#"
            SELECT instance_id, entry_node, created_at, state, result, error
            FROM runner_instances
            WHERE instance_id = $1
            "#,
        )
        .bind(instance_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| BackendError::Message(format!("instance not found: {}", instance_id)))?;

        let instance_id: Uuid = row.get("instance_id");
        let entry_node: Uuid = row.get("entry_node");
        let created_at: DateTime<Utc> = row.get("created_at");
        let state_bytes: Option<Vec<u8>> = row.get("state");
        let result_bytes: Option<Vec<u8>> = row.get("result");
        let error_bytes: Option<Vec<u8>> = row.get("error");

        let status = determine_status(&state_bytes, &result_bytes, &error_bytes);
        let workflow_name = extract_workflow_name(&state_bytes);
        let input_payload = format_state_preview(&state_bytes);
        let result_payload = format_result(&result_bytes);
        let error_payload = format_error(&error_bytes);

        Ok(InstanceDetail {
            id: instance_id,
            entry_node,
            created_at,
            status,
            workflow_name,
            input_payload,
            result_payload,
            error_payload,
        })
    }

    /// Get the execution graph for an instance.
    pub async fn get_execution_graph(
        &self,
        instance_id: Uuid,
    ) -> BackendResult<Option<ExecutionGraphView>> {
        let row = sqlx::query(
            r#"
            SELECT state FROM runner_instances WHERE instance_id = $1
            "#,
        )
        .bind(instance_id)
        .fetch_optional(&self.pool)
        .await?;

        let Some(row) = row else {
            return Ok(None);
        };

        let state_bytes: Option<Vec<u8>> = row.get("state");
        let Some(state_bytes) = state_bytes else {
            return Ok(None);
        };

        let graph_update: GraphUpdate = rmp_serde::from_slice(&state_bytes)
            .map_err(|e| BackendError::Message(format!("failed to decode state: {}", e)))?;

        let nodes: Vec<ExecutionNodeView> = graph_update
            .nodes
            .values()
            .map(|node| ExecutionNodeView {
                id: node.node_id.to_string(),
                node_type: node.node_type.clone(),
                label: node.label.clone(),
                status: format_node_status(&node.status),
                action_name: node.action.as_ref().map(|a| a.action_name.clone()),
                module_name: node.action.as_ref().and_then(|a| a.module_name.clone()),
            })
            .collect();

        let edges: Vec<ExecutionEdgeView> = graph_update
            .edges
            .iter()
            .map(|edge| ExecutionEdgeView {
                source: edge.source.to_string(),
                target: edge.target.to_string(),
                edge_type: format!("{:?}", edge.edge_type),
            })
            .collect();

        Ok(Some(ExecutionGraphView { nodes, edges }))
    }

    /// Get action results for an instance.
    pub async fn get_action_results(&self, instance_id: Uuid) -> BackendResult<Vec<TimelineEntry>> {
        // First get the graph state to map node_ids to action names
        let graph = self.get_execution_graph(instance_id).await?;
        let node_map: std::collections::HashMap<String, &ExecutionNodeView> = graph
            .as_ref()
            .map(|g| g.nodes.iter().map(|n| (n.id.clone(), n)).collect())
            .unwrap_or_default();

        // Get action results from runner_actions_done
        let rows = sqlx::query(
            r#"
            SELECT id, created_at, node_id, action_name, attempt, result
            FROM runner_actions_done
            ORDER BY created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let mut entries = Vec::new();
        for row in rows {
            let node_id: Uuid = row.get("node_id");
            let action_name: String = row.get("action_name");
            let attempt: i32 = row.get("attempt");
            let created_at: DateTime<Utc> = row.get("created_at");
            let result_bytes: Option<Vec<u8>> = row.get("result");

            let node_id_str = node_id.to_string();
            let node = node_map.get(&node_id_str);

            let (response_preview, error) = if let Some(bytes) = &result_bytes {
                match rmp_serde::from_slice::<serde_json::Value>(bytes) {
                    Ok(value) => {
                        let preview = serde_json::to_string_pretty(&value)
                            .unwrap_or_else(|_| "{}".to_string());
                        // Check if it's an error result
                        if value.get("error").is_some() {
                            let err = value
                                .get("error")
                                .and_then(|e| e.as_str())
                                .map(|s| s.to_string());
                            (preview, err)
                        } else {
                            (preview, None)
                        }
                    }
                    Err(_) => ("(decode error)".to_string(), None),
                }
            } else {
                ("(no result)".to_string(), None)
            };

            let status = if error.is_some() {
                "failed".to_string()
            } else {
                "completed".to_string()
            };

            entries.push(TimelineEntry {
                action_id: node_id_str,
                action_name: action_name.clone(),
                module_name: node.and_then(|n| n.module_name.clone()),
                status,
                attempt_number: attempt,
                dispatched_at: Some(created_at.to_rfc3339()),
                completed_at: Some(created_at.to_rfc3339()),
                duration_ms: None, // We don't track start time separately
                request_preview: "{}".to_string(), // Not stored
                response_preview,
                error,
            });
        }

        Ok(entries)
    }

    /// Get distinct workflow names for filtering.
    pub async fn get_distinct_workflows(&self) -> BackendResult<Vec<String>> {
        // In the new schema, workflow names would need to be extracted from state
        // For now, return empty since we don't have a dedicated column
        Ok(Vec::new())
    }

    /// Get distinct statuses for filtering.
    pub async fn get_distinct_statuses(&self) -> BackendResult<Vec<String>> {
        Ok(vec![
            "queued".to_string(),
            "running".to_string(),
            "completed".to_string(),
            "failed".to_string(),
        ])
    }

    // ========================================================================
    // Schedule Queries
    // ========================================================================

    /// Count schedules (excluding deleted).
    pub async fn count_schedules(&self) -> BackendResult<i64> {
        let count = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM workflow_schedules WHERE status != 'deleted'",
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(count)
    }

    /// List schedules with pagination.
    pub async fn list_schedules(
        &self,
        limit: i64,
        offset: i64,
    ) -> BackendResult<Vec<ScheduleSummary>> {
        let rows = sqlx::query(
            r#"
            SELECT id, workflow_name, schedule_name, schedule_type, cron_expression, interval_seconds,
                   status, next_run_at, last_run_at, created_at
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

        let mut schedules = Vec::new();
        for row in rows {
            schedules.push(ScheduleSummary {
                id: row.get::<Uuid, _>("id").to_string(),
                workflow_name: row.get("workflow_name"),
                schedule_name: row.get("schedule_name"),
                schedule_type: row.get("schedule_type"),
                cron_expression: row.get("cron_expression"),
                interval_seconds: row.get("interval_seconds"),
                status: row.get("status"),
                next_run_at: row
                    .get::<Option<DateTime<Utc>>, _>("next_run_at")
                    .map(|dt| dt.to_rfc3339()),
                last_run_at: row
                    .get::<Option<DateTime<Utc>>, _>("last_run_at")
                    .map(|dt| dt.to_rfc3339()),
                created_at: row.get::<DateTime<Utc>, _>("created_at").to_rfc3339(),
            });
        }

        Ok(schedules)
    }

    /// Get a schedule by ID.
    pub async fn get_schedule(&self, schedule_id: Uuid) -> BackendResult<ScheduleDetail> {
        let row = sqlx::query(
            r#"
            SELECT id, workflow_name, schedule_name, schedule_type, cron_expression, interval_seconds,
                   jitter_seconds, input_payload, status, next_run_at, last_run_at, last_instance_id,
                   created_at, updated_at, priority, allow_duplicate
            FROM workflow_schedules
            WHERE id = $1
            "#,
        )
        .bind(schedule_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| BackendError::Message(format!("schedule not found: {}", schedule_id)))?;

        // Try to decode input_payload as JSON string
        let input_payload: Option<String> = row
            .get::<Option<Vec<u8>>, _>("input_payload")
            .and_then(|bytes| {
                rmp_serde::from_slice::<serde_json::Value>(&bytes)
                    .ok()
                    .map(|v| serde_json::to_string_pretty(&v).unwrap_or_default())
            });

        Ok(ScheduleDetail {
            id: row.get::<Uuid, _>("id").to_string(),
            workflow_name: row.get("workflow_name"),
            schedule_name: row.get("schedule_name"),
            schedule_type: row.get("schedule_type"),
            cron_expression: row.get("cron_expression"),
            interval_seconds: row.get("interval_seconds"),
            jitter_seconds: row.get("jitter_seconds"),
            status: row.get("status"),
            next_run_at: row
                .get::<Option<DateTime<Utc>>, _>("next_run_at")
                .map(|dt| dt.to_rfc3339()),
            last_run_at: row
                .get::<Option<DateTime<Utc>>, _>("last_run_at")
                .map(|dt| dt.to_rfc3339()),
            last_instance_id: row
                .get::<Option<Uuid>, _>("last_instance_id")
                .map(|id| id.to_string()),
            created_at: row.get::<DateTime<Utc>, _>("created_at").to_rfc3339(),
            updated_at: row.get::<DateTime<Utc>, _>("updated_at").to_rfc3339(),
            priority: row.get("priority"),
            allow_duplicate: row.get("allow_duplicate"),
            input_payload,
        })
    }

    /// Update schedule status.
    pub async fn update_schedule_status(
        &self,
        schedule_id: Uuid,
        status: &str,
    ) -> BackendResult<bool> {
        let result = sqlx::query(
            r#"
            UPDATE workflow_schedules
            SET status = $2, updated_at = NOW()
            WHERE id = $1
            "#,
        )
        .bind(schedule_id)
        .bind(status)
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    /// Get distinct schedule statuses for filtering.
    pub async fn get_distinct_schedule_statuses(&self) -> BackendResult<Vec<String>> {
        Ok(vec!["active".to_string(), "paused".to_string()])
    }

    /// Get distinct schedule types for filtering.
    pub async fn get_distinct_schedule_types(&self) -> BackendResult<Vec<String>> {
        Ok(vec!["cron".to_string(), "interval".to_string()])
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn determine_status(
    state_bytes: &Option<Vec<u8>>,
    result_bytes: &Option<Vec<u8>>,
    error_bytes: &Option<Vec<u8>>,
) -> InstanceStatus {
    if error_bytes.is_some() {
        return InstanceStatus::Failed;
    }
    if result_bytes.is_some() {
        return InstanceStatus::Completed;
    }
    if state_bytes.is_some() {
        // Has state but no result/error - still running
        return InstanceStatus::Running;
    }
    InstanceStatus::Queued
}

fn extract_workflow_name(state_bytes: &Option<Vec<u8>>) -> Option<String> {
    let bytes = state_bytes.as_ref()?;
    let graph: GraphUpdate = rmp_serde::from_slice(bytes).ok()?;

    // Try to find the first action node and use its action name as workflow name
    for node in graph.nodes.values() {
        if let Some(action) = &node.action {
            return Some(action.action_name.clone());
        }
    }
    None
}

fn extract_input_preview(state_bytes: &Option<Vec<u8>>) -> String {
    let Some(bytes) = state_bytes else {
        return "{}".to_string();
    };

    match rmp_serde::from_slice::<GraphUpdate>(bytes) {
        Ok(graph) => {
            // Try to find entry node's input
            let count = graph.nodes.len();
            format!("{{nodes: {}}}", count)
        }
        Err(_) => "{}".to_string(),
    }
}

fn format_state_preview(state_bytes: &Option<Vec<u8>>) -> String {
    let Some(bytes) = state_bytes else {
        return "(no state)".to_string();
    };

    match rmp_serde::from_slice::<GraphUpdate>(bytes) {
        Ok(graph) => {
            let summary = serde_json::json!({
                "node_count": graph.nodes.len(),
                "edge_count": graph.edges.len(),
            });
            serde_json::to_string_pretty(&summary).unwrap_or_else(|_| "{}".to_string())
        }
        Err(_) => "(decode error)".to_string(),
    }
}

fn format_result(result_bytes: &Option<Vec<u8>>) -> String {
    let Some(bytes) = result_bytes else {
        return "(pending)".to_string();
    };

    match rmp_serde::from_slice::<serde_json::Value>(bytes) {
        Ok(value) => serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string()),
        Err(_) => "(decode error)".to_string(),
    }
}

fn format_error(error_bytes: &Option<Vec<u8>>) -> Option<String> {
    let bytes = error_bytes.as_ref()?;

    match rmp_serde::from_slice::<serde_json::Value>(bytes) {
        Ok(value) => {
            Some(serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".to_string()))
        }
        Err(_) => Some("(decode error)".to_string()),
    }
}

fn format_node_status(status: &NodeStatus) -> String {
    match status {
        NodeStatus::Queued => "queued".to_string(),
        NodeStatus::Running => "running".to_string(),
        NodeStatus::Completed => "completed".to_string(),
        NodeStatus::Failed => "failed".to_string(),
    }
}
