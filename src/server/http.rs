//! HTTP server for workflow management.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::db::Database;

/// Shared state for HTTP handlers.
#[derive(Clone)]
pub struct HttpState {
    pub db: Arc<Database>,
}

/// Health check response.
#[derive(Serialize)]
pub struct HealthResponse {
    pub service: String,
    pub status: String,
}

/// Register workflow request.
#[derive(Deserialize)]
pub struct RegisterWorkflowRequest {
    pub workflow_name: String,
    pub module_name: String,
    pub initial_args: serde_json::Value,
}

/// Register workflow response.
#[derive(Serialize)]
pub struct RegisterWorkflowResponse {
    pub instance_id: String,
}

/// Wait for instance request.
#[derive(Deserialize)]
pub struct WaitForInstanceRequest {
    pub instance_id: String,
    pub poll_interval_secs: Option<f64>,
}

/// Wait for instance response.
#[derive(Serialize)]
pub struct WaitForInstanceResponse {
    pub status: String,
    pub result: Option<serde_json::Value>,
}

/// Create the HTTP router.
pub fn create_router(state: HttpState) -> Router {
    Router::new()
        .route("/healthz", get(health_check))
        .route("/v1/workflows/register", post(register_workflow))
        .route("/v1/workflows/wait", post(wait_for_instance))
        .with_state(state)
}

/// Health check endpoint.
async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        service: "rappel".to_string(),
        status: "healthy".to_string(),
    })
}

/// Register a workflow and start an instance.
async fn register_workflow(
    State(state): State<HttpState>,
    Json(request): Json<RegisterWorkflowRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    info!(
        workflow_name = %request.workflow_name,
        module_name = %request.module_name,
        "Registering workflow"
    );

    let instance_id = state
        .db
        .create_instance(&request.workflow_name, &request.module_name, request.initial_args)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(RegisterWorkflowResponse {
        instance_id: instance_id.to_string(),
    }))
}

/// Wait for an instance to complete.
async fn wait_for_instance(
    State(state): State<HttpState>,
    Json(request): Json<WaitForInstanceRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let instance_id: uuid::Uuid = request
        .instance_id
        .parse()
        .map_err(|_| (StatusCode::BAD_REQUEST, "Invalid instance ID".to_string()))?;

    let poll_interval = request.poll_interval_secs.unwrap_or(1.0);

    loop {
        let instance = state
            .db
            .get_instance(instance_id)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .ok_or((StatusCode::NOT_FOUND, "Instance not found".to_string()))?;

        match instance.status {
            crate::db::InstanceStatus::Completed => {
                return Ok(Json(WaitForInstanceResponse {
                    status: "completed".to_string(),
                    result: instance.result,
                }));
            }
            crate::db::InstanceStatus::Failed => {
                return Ok(Json(WaitForInstanceResponse {
                    status: "failed".to_string(),
                    result: instance.result,
                }));
            }
            _ => {
                tokio::time::sleep(tokio::time::Duration::from_secs_f64(poll_interval)).await;
            }
        }
    }
}
