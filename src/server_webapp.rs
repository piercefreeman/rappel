//! Web application server for the Rappel workflow dashboard.
//!
//! This module provides a human-readable web UI for inspecting workflows,
//! viewing workflow versions, and monitoring workflow instances.
//!
//! The webapp is disabled by default and can be enabled via environment variables:
//! - `CARABINER_WEBAPP_ENABLED`: Set to "true" or "1" to enable
//! - `CARABINER_WEBAPP_PORT`: Port to bind (default: 24119)

use std::{net::SocketAddr, sync::Arc};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::get,
};
use serde::Serialize;
use tera::{Context as TeraContext, Tera};
use tokio::net::TcpListener;
use tracing::{error, info};
use uuid::Uuid;

use crate::db::{Database, WorkflowVersionId, WorkflowVersionSummary};

/// Default port for the webapp server
pub const DEFAULT_WEBAPP_PORT: u16 = 24119;

/// Webapp server configuration
#[derive(Debug, Clone)]
pub struct WebappConfig {
    /// Whether the webapp is enabled
    pub enabled: bool,
    /// Port to bind to
    pub port: u16,
}

impl Default for WebappConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: DEFAULT_WEBAPP_PORT,
        }
    }
}

impl WebappConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let enabled = std::env::var("CARABINER_WEBAPP_ENABLED")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        let port = std::env::var("CARABINER_WEBAPP_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_WEBAPP_PORT);

        Self { enabled, port }
    }

    /// Get the socket address to bind to
    pub fn bind_addr(&self) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], self.port))
    }
}

/// Webapp server handle
pub struct WebappServer {
    addr: SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl WebappServer {
    /// Start the webapp server
    ///
    /// Returns None if the webapp is disabled via configuration.
    pub async fn start(config: WebappConfig, database: Arc<Database>) -> Result<Option<Self>> {
        if !config.enabled {
            info!("webapp disabled (set CARABINER_WEBAPP_ENABLED=true to enable)");
            return Ok(None);
        }

        let bind_addr = config.bind_addr();
        let listener = TcpListener::bind(bind_addr)
            .await
            .with_context(|| format!("failed to bind webapp listener on {bind_addr}"))?;

        let actual_addr = listener.local_addr()?;

        // Initialize templates
        let templates = init_templates()?;

        let state = WebappState {
            database,
            templates: Arc::new(templates),
        };

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

        // Spawn the server task
        tokio::spawn(run_server(listener, state, shutdown_rx));

        info!(addr = %actual_addr, "webapp server started");

        Ok(Some(Self {
            addr: actual_addr,
            shutdown_tx,
        }))
    }

    /// Get the address the server is bound to
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// Shutdown the server
    pub async fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
    }
}

/// Initialize Tera templates
fn init_templates() -> Result<Tera> {
    let mut tera = Tera::new("templates/**/*.html")
        .context("failed to initialize templates from templates/ directory")?;
    tera.autoescape_on(vec![".html", ".tera"]);
    Ok(tera)
}

// ============================================================================
// Internal Server State
// ============================================================================

#[derive(Clone)]
struct WebappState {
    database: Arc<Database>,
    templates: Arc<Tera>,
}

async fn run_server(
    listener: TcpListener,
    state: WebappState,
    shutdown_rx: tokio::sync::oneshot::Receiver<()>,
) {
    let app = Router::new()
        .route("/", get(list_workflows))
        .route("/workflow/{workflow_version_id}", get(workflow_detail))
        .route(
            "/workflow/{workflow_version_id}/run/{instance_id}",
            get(workflow_run_detail),
        )
        .route("/healthz", get(healthz))
        .with_state(state);

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        })
        .await
        .ok();
}

// ============================================================================
// Handlers
// ============================================================================

async fn healthz() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "rappel-webapp",
    })
}

async fn list_workflows(State(state): State<WebappState>) -> impl IntoResponse {
    match state.database.list_workflow_versions().await {
        Ok(workflows) => Html(render_home_page(&state.templates, &workflows)),
        Err(err) => {
            error!(?err, "failed to load workflow summaries");
            Html(render_error_page(
                &state.templates,
                "Unable to load workflows",
                "We couldn't fetch workflow versions. Please check the database connection.",
            ))
        }
    }
}

async fn workflow_detail(
    State(state): State<WebappState>,
    Path(version_id): Path<Uuid>,
) -> impl IntoResponse {
    // Load workflow version
    let version = match state
        .database
        .get_workflow_version(WorkflowVersionId(version_id))
        .await
    {
        Ok(v) => v,
        Err(err) => {
            error!(?err, %version_id, "failed to load workflow version");
            return Html(render_error_page(
                &state.templates,
                "Workflow not found",
                "The requested workflow version could not be located.",
            ));
        }
    };

    // Load recent instances for this version
    let instances = state
        .database
        .list_instances_for_version(WorkflowVersionId(version_id), 20)
        .await
        .unwrap_or_default();

    Html(render_workflow_detail_page(
        &state.templates,
        &version,
        &instances,
    ))
}

async fn workflow_run_detail(
    State(state): State<WebappState>,
    Path((version_id, instance_id)): Path<(Uuid, Uuid)>,
) -> impl IntoResponse {
    // Load workflow version
    let version = match state
        .database
        .get_workflow_version(WorkflowVersionId(version_id))
        .await
    {
        Ok(v) => v,
        Err(err) => {
            error!(?err, %version_id, "failed to load workflow version");
            return Html(render_error_page(
                &state.templates,
                "Workflow not found",
                "The requested workflow version could not be located.",
            ));
        }
    };

    // Load instance
    let instance = match state
        .database
        .get_instance(crate::db::WorkflowInstanceId(instance_id))
        .await
    {
        Ok(i) => i,
        Err(err) => {
            error!(?err, %instance_id, "failed to load instance");
            return Html(render_error_page(
                &state.templates,
                "Instance not found",
                "The requested workflow instance could not be located.",
            ));
        }
    };

    // Load actions for this instance
    let actions = state
        .database
        .get_instance_actions(crate::db::WorkflowInstanceId(instance_id))
        .await
        .unwrap_or_default();

    Html(render_workflow_run_page(
        &state.templates,
        &version,
        &instance,
        &actions,
    ))
}

// ============================================================================
// Response Types
// ============================================================================

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
}

#[derive(Debug)]
struct HttpError {
    status: StatusCode,
    message: String,
}

impl HttpError {
    #[allow(dead_code)]
    fn internal(err: anyhow::Error) -> Self {
        error!(?err, "request failed");
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: "internal server error".to_string(),
        }
    }
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        let body = Json(serde_json::json!({ "message": self.message }));
        (self.status, body).into_response()
    }
}

// ============================================================================
// Template Rendering
// ============================================================================

#[derive(Serialize)]
struct HomePageContext {
    title: String,
    workflows: Vec<HomeWorkflowContext>,
}

#[derive(Serialize)]
struct HomeWorkflowContext {
    id: String,
    name: String,
    hash: String,
    created_at: String,
}

fn render_home_page(templates: &Tera, workflows: &[WorkflowVersionSummary]) -> String {
    let workflows: Vec<HomeWorkflowContext> = workflows
        .iter()
        .map(|w| HomeWorkflowContext {
            id: w.id.to_string(),
            name: w.workflow_name.clone(),
            hash: truncate_hash(&w.dag_hash),
            created_at: w.created_at.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        })
        .collect();

    let context = HomePageContext {
        title: "Registered Workflow Versions".to_string(),
        workflows,
    };

    render_template(templates, "home.html", &context)
}

#[derive(Serialize)]
struct WorkflowDetailPageContext {
    title: String,
    workflow: WorkflowDetailMetadata,
    nodes: Vec<WorkflowNodeContext>,
    has_nodes: bool,
    recent_runs: Vec<WorkflowRunSummary>,
    has_runs: bool,
    graph_data: WorkflowGraphData,
}

#[derive(Serialize)]
struct WorkflowDetailMetadata {
    id: String,
    name: String,
    hash: String,
    created_at: String,
    concurrency_label: String,
}

#[derive(Serialize)]
struct WorkflowNodeContext {
    id: String,
    module: String,
    action: String,
    guard: String,
    depends_on_display: String,
    waits_for_display: String,
}

#[derive(Serialize)]
struct WorkflowRunSummary {
    id: String,
    created_at: String,
    status: String,
    progress: String,
    url: String,
}

#[derive(Serialize)]
struct WorkflowGraphData {
    nodes: Vec<WorkflowGraphNode>,
}

#[derive(Serialize)]
struct WorkflowGraphNode {
    id: String,
    action: String,
    module: String,
    depends_on: Vec<String>,
}

fn render_workflow_detail_page(
    templates: &Tera,
    version: &crate::db::WorkflowVersion,
    instances: &[crate::db::WorkflowInstance],
) -> String {
    // Decode the DAG from the program proto
    let dag = decode_dag_from_proto(&version.program_proto);

    let nodes: Vec<WorkflowNodeContext> = dag
        .iter()
        .map(|node| WorkflowNodeContext {
            id: node.id.clone(),
            module: if node.module.is_empty() {
                "workflow".to_string()
            } else {
                node.module.clone()
            },
            action: if node.action.is_empty() {
                "action".to_string()
            } else {
                node.action.clone()
            },
            guard: node.guard.clone().unwrap_or_else(|| "None".to_string()),
            depends_on_display: format_dependencies(&node.depends_on),
            waits_for_display: format_dependencies(&node.waits_for),
        })
        .collect();

    let graph_data = WorkflowGraphData {
        nodes: dag
            .iter()
            .map(|node| WorkflowGraphNode {
                id: node.id.clone(),
                action: if node.action.is_empty() {
                    "action".to_string()
                } else {
                    node.action.clone()
                },
                module: if node.module.is_empty() {
                    "workflow".to_string()
                } else {
                    node.module.clone()
                },
                depends_on: node.depends_on.clone(),
            })
            .collect(),
    };

    let recent_runs: Vec<WorkflowRunSummary> = instances
        .iter()
        .map(|i| WorkflowRunSummary {
            id: i.id.to_string(),
            created_at: i.created_at.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
            status: i.status.clone(),
            progress: format!("seq {}", i.next_action_seq),
            url: format!("/workflow/{}/run/{}", version.id, i.id),
        })
        .collect();

    let workflow = WorkflowDetailMetadata {
        id: version.id.to_string(),
        name: version.workflow_name.clone(),
        hash: truncate_hash(&version.dag_hash),
        created_at: version
            .created_at
            .format("%Y-%m-%d %H:%M:%S UTC")
            .to_string(),
        concurrency_label: if version.concurrent {
            "Concurrent".to_string()
        } else {
            "Serial".to_string()
        },
    };

    let context = WorkflowDetailPageContext {
        title: format!("{} - Workflow Detail", version.workflow_name),
        workflow,
        has_nodes: !nodes.is_empty(),
        nodes,
        has_runs: !recent_runs.is_empty(),
        recent_runs,
        graph_data,
    };

    render_template(templates, "workflow.html", &context)
}

#[derive(Serialize)]
struct WorkflowRunPageContext {
    title: String,
    workflow: WorkflowDetailMetadata,
    instance: InstanceContext,
    nodes: Vec<NodeExecutionContext>,
}

#[derive(Serialize)]
struct InstanceContext {
    id: String,
    created_at: String,
    status: String,
    progress: String,
    input_payload: String,
    result_payload: String,
}

#[derive(Serialize)]
struct NodeExecutionContext {
    id: String,
    module: String,
    action: String,
    status: String,
    request_payload: String,
    response_payload: String,
}

fn render_workflow_run_page(
    templates: &Tera,
    version: &crate::db::WorkflowVersion,
    instance: &crate::db::WorkflowInstance,
    actions: &[crate::db::QueuedAction],
) -> String {
    let workflow = WorkflowDetailMetadata {
        id: version.id.to_string(),
        name: version.workflow_name.clone(),
        hash: truncate_hash(&version.dag_hash),
        created_at: version
            .created_at
            .format("%Y-%m-%d %H:%M:%S UTC")
            .to_string(),
        concurrency_label: if version.concurrent {
            "Concurrent".to_string()
        } else {
            "Serial".to_string()
        },
    };

    let instance_ctx = InstanceContext {
        id: instance.id.to_string(),
        created_at: instance
            .created_at
            .format("%Y-%m-%d %H:%M:%S UTC")
            .to_string(),
        status: instance.status.clone(),
        progress: format!("seq {}", instance.next_action_seq),
        input_payload: format_payload(&instance.input_payload),
        result_payload: format_payload(&instance.result_payload),
    };

    let nodes: Vec<NodeExecutionContext> = actions
        .iter()
        .map(|a| NodeExecutionContext {
            id: a.node_id.clone().unwrap_or_else(|| a.id.to_string()),
            module: a.module_name.clone(),
            action: a.action_name.clone(),
            status: a.node_type.clone(),
            request_payload: format_binary_payload(&a.dispatch_payload),
            response_payload: "(see result)".to_string(),
        })
        .collect();

    let context = WorkflowRunPageContext {
        title: format!("Run {} - {}", instance.id, version.workflow_name),
        workflow,
        instance: instance_ctx,
        nodes,
    };

    render_template(templates, "workflow_run.html", &context)
}

#[derive(Serialize)]
struct ErrorPageContext {
    title: String,
    message: String,
}

fn render_error_page(templates: &Tera, title: &str, message: &str) -> String {
    let context = ErrorPageContext {
        title: title.to_string(),
        message: message.to_string(),
    };
    render_template(templates, "error.html", &context)
}

fn render_template<T: Serialize>(templates: &Tera, template: &str, data: &T) -> String {
    let context = match TeraContext::from_serialize(data) {
        Ok(ctx) => ctx,
        Err(err) => {
            error!(?err, "failed to serialize template context");
            TeraContext::new()
        }
    };
    match templates.render(template, &context) {
        Ok(html) => html,
        Err(err) => {
            error!(?err, template = template, "failed to render template");
            "<!DOCTYPE html><html lang=\"en\"><body><h1>Template error</h1></body></html>"
                .to_string()
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Simple DAG node extracted from proto for display
struct SimpleDagNode {
    id: String,
    module: String,
    action: String,
    guard: Option<String>,
    depends_on: Vec<String>,
    waits_for: Vec<String>,
}

fn decode_dag_from_proto(proto_bytes: &[u8]) -> Vec<SimpleDagNode> {
    use prost::Message;

    // Try to decode the program proto (uses parser::ast which is the proto type)
    let program = match crate::parser::ast::Program::decode(proto_bytes) {
        Ok(p) => p,
        Err(_) => return vec![],
    };

    // Convert to DAG using the existing converter
    let dag = crate::dag::convert_to_dag(&program);

    dag.nodes
        .values()
        .map(|node| {
            // Find depends_on edges (StateMachine type = control flow)
            let depends_on: Vec<String> = dag
                .edges
                .iter()
                .filter(|e| e.target == node.id && e.edge_type == crate::dag::EdgeType::StateMachine)
                .map(|e| e.source.clone())
                .collect();

            // Find waits_for edges (DataFlow type)
            let waits_for: Vec<String> = dag
                .edges
                .iter()
                .filter(|e| e.target == node.id && e.edge_type == crate::dag::EdgeType::DataFlow)
                .map(|e| e.source.clone())
                .collect();

            SimpleDagNode {
                id: node.id.clone(),
                module: node.module_name.clone().unwrap_or_default(),
                action: node.action_name.clone().unwrap_or_default(),
                guard: node.guard_expr.as_ref().map(|e| crate::print_expr(e)),
                depends_on,
                waits_for,
            }
        })
        .collect()
}

fn truncate_hash(hash: &str) -> String {
    if hash.len() > 12 {
        format!("{}...", &hash[..12])
    } else {
        hash.to_string()
    }
}

fn format_dependencies(items: &[String]) -> String {
    if items.is_empty() {
        "None".to_string()
    } else {
        items.join(", ")
    }
}

fn format_payload(payload: &Option<Vec<u8>>) -> String {
    match payload {
        Some(bytes) if !bytes.is_empty() => format_binary_payload(bytes),
        _ => "(empty)".to_string(),
    }
}

fn format_binary_payload(bytes: &[u8]) -> String {
    // Try to decode as UTF-8 first
    if let Ok(s) = std::str::from_utf8(bytes) {
        // Try to pretty-print as JSON
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(s) {
            if let Ok(pretty) = serde_json::to_string_pretty(&json) {
                return pretty;
            }
        }
        return s.to_string();
    }

    // Fall back to hex representation for binary data
    format!("({} bytes)", bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webapp_config_default() {
        let config = WebappConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.port, DEFAULT_WEBAPP_PORT);
    }

    #[test]
    fn test_truncate_hash() {
        assert_eq!(truncate_hash("abc"), "abc");
        assert_eq!(truncate_hash("abcdefghijklmnop"), "abcdefghijkl...");
    }

    #[test]
    fn test_format_dependencies() {
        assert_eq!(format_dependencies(&[]), "None");
        assert_eq!(
            format_dependencies(&["a".to_string(), "b".to_string()]),
            "a, b"
        );
    }
}
