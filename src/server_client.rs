//! Server/Client infrastructure for the Rappel workflow engine.
//!
//! This module provides:
//! - HTTP server for workflow registration and status
//! - gRPC server for Python worker connections (worker bridge)
//! - Configuration and server lifecycle management

use std::{net::SocketAddr, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use prost::Message;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use tracing::{error, info};

use crate::{db::Database, messages::proto};

/// Service name for health checks
pub const SERVICE_NAME: &str = "rappel";

/// Health check endpoint path
pub const HEALTH_PATH: &str = "/healthz";

/// Workflow registration endpoint path
pub const REGISTER_PATH: &str = "/v1/workflows/register";

/// Wait for instance endpoint path
pub const WAIT_PATH: &str = "/v1/workflows/wait";

/// Construct a health check URL
pub fn health_url(host: &str, port: u16) -> String {
    format!("http://{host}:{port}{HEALTH_PATH}")
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub http_addr: SocketAddr,
    pub grpc_addr: SocketAddr,
    pub database_url: String,
}

/// Run both HTTP and gRPC servers
pub async fn run_servers(config: ServerConfig) -> Result<()> {
    let ServerConfig {
        http_addr,
        grpc_addr,
        database_url,
    } = config;

    let database_url = Arc::new(database_url);
    let database = Database::connect(database_url.as_ref())
        .await
        .with_context(|| format!("failed to connect to database at {}", database_url.as_ref()))?;

    let http_listener = TcpListener::bind(http_addr)
        .await
        .with_context(|| format!("failed to bind http listener on {http_addr}"))?;
    let grpc_listener = TcpListener::bind(grpc_addr)
        .await
        .with_context(|| format!("failed to bind grpc listener on {grpc_addr}"))?;

    info!(?http_addr, ?grpc_addr, "rappel server listening");

    let http_state = HttpState::new(
        SERVICE_NAME,
        http_addr,
        grpc_addr,
        Arc::clone(&database_url),
        database,
    );

    let http_task = tokio::spawn(run_http_server(http_listener, http_state));
    let grpc_task = tokio::spawn(run_grpc_server(grpc_listener, database_url));

    tokio::select! {
        result = http_task => result??,
        result = grpc_task => result??,
    }

    Ok(())
}

// ============================================================================
// HTTP Server
// ============================================================================

#[derive(Clone)]
struct HttpState {
    service_name: &'static str,
    http_addr: SocketAddr,
    grpc_addr: SocketAddr,
    #[allow(dead_code)]
    database_url: Arc<String>,
    #[allow(dead_code)]
    database: Database,
}

impl HttpState {
    fn new(
        service_name: &'static str,
        http_addr: SocketAddr,
        grpc_addr: SocketAddr,
        database_url: Arc<String>,
        database: Database,
    ) -> Self {
        Self {
            service_name,
            http_addr,
            grpc_addr,
            database_url,
            database,
        }
    }
}

async fn run_http_server(listener: TcpListener, state: HttpState) -> Result<()> {
    let app = Router::new()
        .route(HEALTH_PATH, get(healthz))
        .route(REGISTER_PATH, post(register_workflow_http))
        .route(WAIT_PATH, post(wait_for_instance_http))
        .with_state(state);

    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
    http_port: u16,
    grpc_port: u16,
}

async fn healthz(State(state): State<HttpState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: state.service_name,
        http_port: state.http_addr.port(),
        grpc_port: state.grpc_addr.port(),
    })
}

#[derive(Debug, Deserialize)]
struct RegisterWorkflowHttpRequest {
    registration_b64: String,
}

#[derive(Debug, Serialize)]
struct RegisterWorkflowHttpResponse {
    workflow_version_id: String,
    workflow_instance_id: String,
}

#[derive(Debug, Deserialize)]
struct WaitForInstanceHttpRequest {
    instance_id: String,
    poll_interval_secs: Option<f64>,
}

#[derive(Debug, Serialize)]
struct WaitForInstanceHttpResponse {
    payload_b64: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponseBody {
    message: String,
}

#[derive(Debug)]
struct HttpError {
    status: StatusCode,
    message: String,
}

impl HttpError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    #[allow(dead_code)]
    fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.into(),
        }
    }

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
        let body = Json(ErrorResponseBody {
            message: self.message,
        });
        (self.status, body).into_response()
    }
}

async fn register_workflow_http(
    State(_state): State<HttpState>,
    Json(payload): Json<RegisterWorkflowHttpRequest>,
) -> Result<Json<RegisterWorkflowHttpResponse>, HttpError> {
    use base64::{Engine as _, engine::general_purpose};

    if payload.registration_b64.is_empty() {
        return Err(HttpError::bad_request("registration payload missing"));
    }

    let bytes = general_purpose::STANDARD
        .decode(&payload.registration_b64)
        .map_err(|_| HttpError::bad_request("invalid base64 payload"))?;

    // Decode the registration
    let _registration = proto::WorkflowRegistration::decode(&bytes[..])
        .map_err(|e| HttpError::bad_request(format!("invalid registration: {e}")))?;

    // TODO: Register the workflow and create instance
    // For now, return placeholder IDs
    let version_id = uuid::Uuid::new_v4();
    let instance_id = uuid::Uuid::new_v4();

    Ok(Json(RegisterWorkflowHttpResponse {
        workflow_version_id: version_id.to_string(),
        workflow_instance_id: instance_id.to_string(),
    }))
}

async fn wait_for_instance_http(
    State(_state): State<HttpState>,
    Json(request): Json<WaitForInstanceHttpRequest>,
) -> Result<Json<WaitForInstanceHttpResponse>, HttpError> {
    use base64::{Engine as _, engine::general_purpose};

    let _interval = sanitize_interval(request.poll_interval_secs);
    let _instance_id: uuid::Uuid = request
        .instance_id
        .parse()
        .map_err(|_| HttpError::bad_request("invalid instance_id"))?;

    // TODO: Poll for instance completion
    // For now, return empty payload
    let encoded = general_purpose::STANDARD.encode([]);

    Ok(Json(WaitForInstanceHttpResponse {
        payload_b64: encoded,
    }))
}

pub(crate) fn sanitize_interval(value: Option<f64>) -> Duration {
    let raw = value.unwrap_or(1.0);
    let clamped = raw.clamp(0.1, 30.0);
    Duration::from_secs_f64(clamped)
}

// ============================================================================
// gRPC Server
// ============================================================================

async fn run_grpc_server(listener: TcpListener, database_url: Arc<String>) -> Result<()> {
    use proto::workflow_service_server::WorkflowServiceServer;

    let incoming = TcpListenerStream::new(listener);
    let service = WorkflowGrpcService::new(database_url);

    // TODO: Also add the worker bridge server here
    Server::builder()
        .add_service(WorkflowServiceServer::new(service))
        .serve_with_incoming(incoming)
        .await?;

    Ok(())
}

#[derive(Clone)]
struct WorkflowGrpcService {
    #[allow(dead_code)]
    database_url: Arc<String>,
}

impl WorkflowGrpcService {
    fn new(database_url: Arc<String>) -> Self {
        Self { database_url }
    }
}

#[tonic::async_trait]
impl proto::workflow_service_server::WorkflowService for WorkflowGrpcService {
    async fn register_workflow(
        &self,
        request: tonic::Request<proto::RegisterWorkflowRequest>,
    ) -> Result<tonic::Response<proto::RegisterWorkflowResponse>, tonic::Status> {
        let inner = request.into_inner();
        let _registration = inner
            .registration
            .ok_or_else(|| tonic::Status::invalid_argument("registration missing"))?;

        // TODO: Register the workflow and create instance
        let version_id = uuid::Uuid::new_v4();
        let instance_id = uuid::Uuid::new_v4();

        Ok(tonic::Response::new(proto::RegisterWorkflowResponse {
            workflow_version_id: version_id.to_string(),
            workflow_instance_id: instance_id.to_string(),
        }))
    }

    async fn wait_for_instance(
        &self,
        request: tonic::Request<proto::WaitForInstanceRequest>,
    ) -> Result<tonic::Response<proto::WaitForInstanceResponse>, tonic::Status> {
        let inner = request.into_inner();
        let _instance_id: uuid::Uuid = inner.instance_id.parse().map_err(|err| {
            tonic::Status::invalid_argument(format!("invalid instance_id: {err}"))
        })?;

        let _interval = sanitize_interval(Some(inner.poll_interval_secs));

        // TODO: Poll for instance completion
        Ok(tonic::Response::new(proto::WaitForInstanceResponse {
            payload: vec![],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_url() {
        assert_eq!(
            health_url("127.0.0.1", 24117),
            "http://127.0.0.1:24117/healthz"
        );
    }

    #[test]
    fn test_sanitize_interval() {
        // Test default
        assert_eq!(sanitize_interval(None), Duration::from_secs_f64(1.0));

        // Test clamping
        assert_eq!(sanitize_interval(Some(0.01)), Duration::from_secs_f64(0.1));
        assert_eq!(
            sanitize_interval(Some(100.0)),
            Duration::from_secs_f64(30.0)
        );
        assert_eq!(sanitize_interval(Some(5.0)), Duration::from_secs_f64(5.0));
    }
}
