//! Shared types for the webapp.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Configuration for the webapp server.
#[derive(Debug, Clone)]
pub struct WebappConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
}

impl Default for WebappConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            host: "0.0.0.0".to_string(),
            port: 24119,
        }
    }
}

impl WebappConfig {
    /// Create config from environment variables.
    pub fn from_env() -> Self {
        let enabled = std::env::var("RAPPEL_WEBAPP_ENABLED")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        let (host, port) = std::env::var("RAPPEL_WEBAPP_ADDR")
            .ok()
            .and_then(|addr| {
                let parts: Vec<&str> = addr.split(':').collect();
                if parts.len() == 2 {
                    let host = parts[0].to_string();
                    let port = parts[1].parse().ok()?;
                    Some((host, port))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| ("0.0.0.0".to_string(), 24119));

        Self {
            enabled,
            host,
            port,
        }
    }

    /// Get the bind address.
    pub fn bind_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Instance status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InstanceStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

impl std::fmt::Display for InstanceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Summary of a workflow instance for listing.
#[derive(Debug, Clone, Serialize)]
pub struct InstanceSummary {
    pub id: Uuid,
    pub entry_node: Uuid,
    pub created_at: DateTime<Utc>,
    pub status: InstanceStatus,
    pub workflow_name: Option<String>,
    pub input_preview: String,
}

/// Full details of a workflow instance.
#[derive(Debug, Clone, Serialize)]
pub struct InstanceDetail {
    pub id: Uuid,
    pub entry_node: Uuid,
    pub created_at: DateTime<Utc>,
    pub status: InstanceStatus,
    pub workflow_name: Option<String>,
    pub input_payload: String,
    pub result_payload: String,
    pub error_payload: Option<String>,
}

/// Node in the execution graph for display.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionNodeView {
    pub id: String,
    pub node_type: String,
    pub label: String,
    pub status: String,
    pub action_name: Option<String>,
    pub module_name: Option<String>,
}

/// Edge in the execution graph for display.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionEdgeView {
    pub source: String,
    pub target: String,
    pub edge_type: String,
}

/// Execution graph data for rendering.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionGraphView {
    pub nodes: Vec<ExecutionNodeView>,
    pub edges: Vec<ExecutionEdgeView>,
}

/// Timeline entry for an action execution.
#[derive(Debug, Clone, Serialize)]
pub struct TimelineEntry {
    pub action_id: String,
    pub action_name: String,
    pub module_name: Option<String>,
    pub status: String,
    pub attempt_number: i32,
    pub dispatched_at: Option<String>,
    pub completed_at: Option<String>,
    pub duration_ms: Option<i64>,
    pub request_preview: String,
    pub response_preview: String,
    pub error: Option<String>,
}

/// Action log entry with full details.
#[derive(Debug, Clone, Serialize)]
pub struct ActionLogEntry {
    pub action_id: String,
    pub action_name: String,
    pub module_name: Option<String>,
    pub status: String,
    pub attempt_number: i32,
    pub dispatched_at: Option<String>,
    pub completed_at: Option<String>,
    pub duration_ms: Option<i64>,
    pub request: String,
    pub response: String,
    pub error: Option<String>,
}

/// Response for the workflow run data API.
#[derive(Debug, Serialize)]
pub struct WorkflowRunDataResponse {
    pub nodes: Vec<ExecutionNodeView>,
    pub timeline: Vec<TimelineEntry>,
    pub page: i64,
    pub per_page: i64,
    pub total: i64,
    pub has_more: bool,
}

/// Response for action logs API.
#[derive(Debug, Serialize)]
pub struct ActionLogsResponse {
    pub logs: Vec<ActionLogEntry>,
}

/// Filter values response.
#[derive(Debug, Serialize)]
pub struct FilterValuesResponse {
    pub values: Vec<String>,
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub service: &'static str,
}

/// Export format for a workflow instance.
#[derive(Debug, Serialize)]
pub struct WorkflowInstanceExport {
    pub export_version: &'static str,
    pub exported_at: String,
    pub instance: InstanceExportInfo,
    pub nodes: Vec<ExecutionNodeView>,
    pub timeline: Vec<TimelineEntry>,
}

/// Instance info for export.
#[derive(Debug, Serialize)]
pub struct InstanceExportInfo {
    pub id: String,
    pub status: String,
    pub created_at: String,
    pub input_payload: String,
    pub result_payload: String,
}

/// Schedule summary for listing.
#[derive(Debug, Clone, Serialize)]
pub struct ScheduleSummary {
    pub id: String,
    pub workflow_name: String,
    pub schedule_name: String,
    pub schedule_type: String,
    pub cron_expression: Option<String>,
    pub interval_seconds: Option<i64>,
    pub status: String,
    pub next_run_at: Option<String>,
    pub last_run_at: Option<String>,
    pub created_at: String,
}

/// Full schedule details.
#[derive(Debug, Clone, Serialize)]
pub struct ScheduleDetail {
    pub id: String,
    pub workflow_name: String,
    pub schedule_name: String,
    pub schedule_type: String,
    pub cron_expression: Option<String>,
    pub interval_seconds: Option<i64>,
    pub jitter_seconds: i64,
    pub status: String,
    pub next_run_at: Option<String>,
    pub last_run_at: Option<String>,
    pub last_instance_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub priority: i32,
    pub allow_duplicate: bool,
    pub input_payload: Option<String>,
}
