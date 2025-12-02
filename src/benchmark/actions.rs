//! Action benchmark harness using the new Store/Scheduler architecture.
//!
//! This benchmark tests raw action dispatch throughput by running a simple
//! echo workflow that exercises the full action dispatch path without
//! complex workflow logic.

use anyhow::{Context, Result, anyhow};
use futures::{StreamExt, future::BoxFuture, stream::FuturesUnordered};
use prost::Message;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response as GrpcResponse, Status, async_trait, transport::Server};
use tracing::{info, warn};

use crate::benchmark::common::{BenchmarkResult, BenchmarkSummary};
use crate::benchmark::fixtures;
use crate::messages::proto::{self, NodeDispatch, workflow_service_server::WorkflowServiceServer};
use crate::server_worker::WorkerBridgeServer;
use crate::store::{ActionCompletion, QueuedAction, Store};
use crate::worker::{ActionDispatchPayload, PythonWorkerConfig, PythonWorkerPool};

/// Type alias for dispatch result
type DispatchResult = Result<BenchmarkResult>;

/// Completion record for async processing
struct CompletionRecord {
    action_id: uuid::Uuid,
    node_id: String,
    instance_id: uuid::Uuid,
    success: bool,
    result_payload: Vec<u8>,
}

/// Configuration for action benchmark
#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub total_messages: usize,
    pub in_flight: usize,
    pub payload_size: usize,
    pub progress_interval: Option<Duration>,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            total_messages: 10_000,
            in_flight: 32,
            payload_size: 4096,
            progress_interval: None,
        }
    }
}

/// Action benchmark harness
pub struct BenchmarkHarness {
    store: Arc<Store>,
    worker_server: Arc<WorkerBridgeServer>,
    pool: Arc<PythonWorkerPool>,
    _python_env: TempDir,
    _registration_server: BenchmarkRegistrationServer,
    workflow_id: uuid::Uuid,
}

impl BenchmarkHarness {
    /// Create a new benchmark harness
    pub async fn new(
        worker_count: usize,
        store: Store,
    ) -> Result<Self> {
        let store = Arc::new(store);
        store
            .init_schema()
            .await
            .context("Failed to initialize schema")?;

        // Clean up any existing data
        cleanup_store(&store).await?;

        // Start GRPC registration server
        let registration_server = BenchmarkRegistrationServer::start(Arc::clone(&store)).await?;
        let grpc_port = registration_server.port();

        // Set up Python environment and run registration
        let python_env = setup_python_env(grpc_port).await?;

        // Wait for registration to complete
        sleep(Duration::from_millis(500)).await;

        // Load the registered workflow
        let (workflow_id, _dag) = store
            .get_workflow_by_name("benchmark.echo_action")
            .await?
            .ok_or_else(|| anyhow!("Benchmark workflow not registered"))?;

        // Clear action queue and instances from registration
        // (registration creates an initial instance with empty input that we don't want)
        sqlx::query("DELETE FROM action_queue")
            .execute(store.pool())
            .await
            .ok();
        sqlx::query("DELETE FROM instances")
            .execute(store.pool())
            .await
            .ok();

        // Set up workers with extra_python_paths including the temp directory
        let worker_server = WorkerBridgeServer::start(None).await?;
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let worker_script = repo_root
            .join("python")
            .join(".venv")
            .join("bin")
            .join("rappel-worker");
        let worker_config = PythonWorkerConfig {
            script_path: worker_script,
            script_args: Vec::new(),
            user_module: "benchmark_actions".to_string(),
            extra_python_paths: vec![python_env.path().to_path_buf()],
        };
        let pool = Arc::new(
            PythonWorkerPool::new(worker_config, worker_count, Arc::clone(&worker_server)).await?,
        );

        Ok(Self {
            store,
            worker_server,
            pool,
            _python_env: python_env,
            _registration_server: registration_server,
            workflow_id,
        })
    }

    /// Run the benchmark with the given configuration
    pub async fn run(&self, config: &HarnessConfig) -> Result<BenchmarkSummary> {
        let total = config.total_messages;
        let mut completed = Vec::with_capacity(total);
        let start = Instant::now();
        let mut last_report = start;

        // Build payload for all instances
        let payload_str = "x".repeat(config.payload_size);
        let workflow_input = encode_workflow_input(&[("payload", &payload_str)]);

        // Create all workflow instances up front
        let mut pending_instances = Vec::with_capacity(total);
        for _ in 0..total {
            let instance_id = self
                .store
                .create_instance_with_input(self.workflow_id, &workflow_input)
                .await?;
            pending_instances.push(instance_id);
        }

        // Set up completion channel for async completion processing
        let (completion_tx, completion_rx) = mpsc::channel::<CompletionRecord>(config.in_flight * 4);
        let store_for_completion = Arc::clone(&self.store);
        let completion_handle = tokio::spawn(async move {
            Self::completion_loop(store_for_completion, completion_rx).await;
        });

        // Process actions with parallel dispatch
        let worker_count = self.pool.len().max(1);
        let max_inflight = config.in_flight.max(1) * worker_count;
        let batch_size = max_inflight.min(100) as i64;
        let mut inflight: FuturesUnordered<BoxFuture<'_, DispatchResult>> = FuturesUnordered::new();
        let mut dispatched = 0usize;

        while completed.len() < total {
            // Dispatch more actions if we have capacity
            while inflight.len() < max_inflight {
                let available = max_inflight - inflight.len();
                let actions = self.store.dequeue_actions(available.min(batch_size as usize) as i64).await?;
                if actions.is_empty() {
                    break;
                }

                for action in actions {
                    let pool = Arc::clone(&self.pool);
                    let tx = completion_tx.clone();
                    let store = Arc::clone(&self.store);
                    let fut: BoxFuture<'_, DispatchResult> = Box::pin(async move {
                        Self::dispatch_single_action(store, pool, tx, action).await
                    });
                    inflight.push(fut);
                    dispatched += 1;
                }
            }

            // Wait for at least one action to complete
            match inflight.next().await {
                Some(Ok(result)) => {
                    completed.push(result);

                    // Progress reporting
                    if let Some(interval) = config.progress_interval {
                        let now = Instant::now();
                        if now.duration_since(last_report) >= interval {
                            let elapsed = now.duration_since(start);
                            let throughput = completed.len() as f64 / elapsed.as_secs_f64().max(1e-9);
                            info!(
                                processed = completed.len(),
                                total,
                                elapsed = %format!("{:.1}s", elapsed.as_secs_f64()),
                                throughput = %format!("{:.0} msg/s", throughput),
                                in_flight = inflight.len(),
                                dispatched,
                                "benchmark progress",
                            );
                            last_report = now;
                        }
                    }
                }
                Some(Err(err)) => {
                    warn!(?err, "dispatch error");
                }
                None => {
                    // No more inflight tasks, check if we need more
                    if dispatched >= total {
                        break;
                    }
                    // Small sleep to avoid busy-loop when waiting for new actions
                    sleep(Duration::from_millis(1)).await;
                }
            }
        }

        // Wait for completion processing to finish
        drop(completion_tx);
        let _ = completion_handle.await;

        let elapsed = start.elapsed();
        Ok(BenchmarkSummary::from_results(completed, elapsed))
    }

    /// Dispatch a single action and return the result
    async fn dispatch_single_action(
        store: Arc<Store>,
        pool: Arc<PythonWorkerPool>,
        completion_tx: mpsc::Sender<CompletionRecord>,
        action: QueuedAction,
    ) -> Result<BenchmarkResult> {
        // Decode and dispatch to worker
        let dispatch: NodeDispatch =
            serde_json::from_str(&action.dispatch_json).context("Failed to decode dispatch")?;

        // Check if this is a sleep action (no-op)
        let is_sleep = dispatch
            .node
            .as_ref()
            .map(|n| n.action == "__sleep__")
            .unwrap_or(false);

        if is_sleep {
            let completion = ActionCompletion {
                action_id: action.id,
                node_id: action.node_id.clone(),
                instance_id: action.instance_id,
                success: true,
                result: None,
                exception_type: None,
                exception_module: None,
            };
            store.complete_action(completion).await?;
            // Return a minimal result for sleep actions
            return Ok(BenchmarkResult {
                sequence: 0,
                ack_latency: Duration::ZERO,
                round_trip: Duration::ZERO,
                worker_duration: Duration::ZERO,
            });
        }

        let payload = ActionDispatchPayload {
            action_id: action.id,
            instance_id: action.instance_id,
            sequence: action.attempt,
            dispatch,
            timeout_seconds: action.timeout_seconds,
            max_retries: action.max_retries,
            attempt_number: action.attempt,
            dispatch_token: action.id,
        };

        let worker = pool.next_worker();
        let metrics = worker.send_action(payload).await?;

        // Send completion record to async processor
        let record = CompletionRecord {
            action_id: metrics.action_id,
            node_id: action.node_id,
            instance_id: action.instance_id,
            success: metrics.success,
            result_payload: metrics.response_payload.clone(),
        };
        if let Err(err) = completion_tx.send(record).await {
            warn!(?err, "completion channel closed");
        }

        Ok(metrics.into())
    }

    /// Background loop to process completions (parallel processing)
    async fn completion_loop(store: Arc<Store>, mut rx: mpsc::Receiver<CompletionRecord>) {
        use tokio::sync::Semaphore;

        // Process completions in parallel with bounded concurrency
        let semaphore = Arc::new(Semaphore::new(32)); // Allow 32 concurrent completion tasks
        let mut handles = Vec::new();

        while let Some(record) = rx.recv().await {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let store = Arc::clone(&store);

            let handle = tokio::spawn(async move {
                let decoded = decode_result(&record.result_payload);
                let completion = ActionCompletion {
                    action_id: record.action_id,
                    node_id: record.node_id,
                    instance_id: record.instance_id,
                    success: record.success,
                    result: decoded.result,
                    exception_type: decoded.exception_type,
                    exception_module: decoded.exception_module,
                };

                if let Err(err) = store.complete_action(completion).await {
                    warn!(?err, "failed to complete action");
                }
                drop(permit);
            });
            handles.push(handle);
        }

        // Wait for all completion tasks to finish
        for handle in handles {
            let _ = handle.await;
        }
    }

    /// Shutdown the harness
    pub async fn shutdown(self) -> Result<()> {
        match Arc::try_unwrap(self.pool) {
            Ok(pool) => pool.shutdown().await?,
            Err(_) => warn!("pool still has references, skipping shutdown"),
        }
        self.worker_server.shutdown().await;
        Ok(())
    }
}

/// Clean up database tables
async fn cleanup_store(store: &Store) -> Result<()> {
    sqlx::query("DELETE FROM action_queue")
        .execute(store.pool())
        .await
        .ok();
    sqlx::query("DELETE FROM instances")
        .execute(store.pool())
        .await
        .ok();
    sqlx::query("DELETE FROM workflows")
        .execute(store.pool())
        .await
        .ok();
    Ok(())
}

/// Set up Python environment with benchmark fixtures
async fn setup_python_env(grpc_port: u16) -> Result<TempDir> {
    use std::fs;
    use std::process::Command;

    let temp_dir = TempDir::new()?;

    // Write fixture files
    fs::write(temp_dir.path().join("__init__.py"), fixtures::INIT_PY)?;
    fs::write(
        temp_dir.path().join("benchmark_common.py"),
        fixtures::BENCHMARK_COMMON,
    )?;
    fs::write(
        temp_dir.path().join("benchmark_actions.py"),
        fixtures::BENCHMARK_ACTIONS,
    )?;

    // Write entrypoint script that registers the workflow by calling run()
    let entrypoint = r#"
import asyncio
from benchmark_actions import EchoActionWorkflow

if __name__ == "__main__":
    # Running the workflow triggers registration with the gRPC server
    asyncio.run(EchoActionWorkflow().run())
"#;
    let entrypoint_path = temp_dir.path().join("__entrypoint__.py");
    fs::write(&entrypoint_path, entrypoint)?;

    // Get Python paths
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let python_src = repo_root.join("python").join("src");
    let python_dir = repo_root.join("python");
    let proto_dir = python_dir.join("proto");

    let python_path = format!(
        "{}:{}:{}:{}",
        temp_dir.path().display(),
        python_src.display(),
        python_dir.display(),
        proto_dir.display()
    );

    // Run the entrypoint
    let python_bin = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "uv".to_string());
    let mut cmd = Command::new(&python_bin);

    if python_bin.ends_with("uv") || python_bin == "uv" {
        cmd.arg("run").arg("python");
    }

    cmd.arg(&entrypoint_path)
        .env("PYTHONPATH", &python_path)
        .env("CARABINER_SKIP_WAIT_FOR_INSTANCE", "1")
        .env("CARABINER_GRPC_ADDR", format!("127.0.0.1:{}", grpc_port))
        .env("CARABINER_SERVER_PORT", (grpc_port - 1).to_string())
        .current_dir(&python_dir);

    let child = cmd.spawn().context("Failed to spawn Python entrypoint")?;
    let output = tokio::time::timeout(
        Duration::from_secs(30),
        tokio::task::spawn_blocking(move || child.wait_with_output()),
    )
    .await
    .context("Python entrypoint timed out after 30s")?
    .context("Task join failed")?
    .context("Failed to run Python entrypoint")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(anyhow!(
            "Python entrypoint failed:\nstdout: {}\nstderr: {}",
            stdout,
            stderr
        ));
    }

    Ok(temp_dir)
}

/// Encode workflow inputs as protobuf
fn encode_workflow_input(inputs: &[(&str, &str)]) -> Vec<u8> {
    let arguments: Vec<proto::WorkflowArgument> = inputs
        .iter()
        .map(|(key, value)| proto::WorkflowArgument {
            key: (*key).to_string(),
            value: Some(proto::WorkflowArgumentValue {
                kind: Some(proto::workflow_argument_value::Kind::Primitive(
                    proto::PrimitiveWorkflowArgument {
                        kind: Some(proto::primitive_workflow_argument::Kind::StringValue(
                            (*value).to_string(),
                        )),
                    },
                )),
            }),
        })
        .collect();

    let args = proto::WorkflowArguments { arguments };
    args.encode_to_vec()
}

/// Result of decoding a worker response payload
struct DecodedResult {
    result: Option<serde_json::Value>,
    exception_type: Option<String>,
    exception_module: Option<String>,
}

/// Decode result payload from worker response
fn decode_result(payload: &[u8]) -> DecodedResult {
    if payload.is_empty() {
        return DecodedResult {
            result: None,
            exception_type: None,
            exception_module: None,
        };
    }

    let args = match proto::WorkflowArguments::decode(payload) {
        Ok(a) => a,
        Err(_) => {
            return DecodedResult {
                result: None,
                exception_type: None,
                exception_module: None,
            };
        }
    };

    let result = args
        .arguments
        .iter()
        .find(|a| a.key == "result")
        .and_then(|arg| arg.value.as_ref())
        .map(proto_value_to_json);

    let (exception_type, exception_module) = args
        .arguments
        .iter()
        .find(|a| a.key == "error")
        .and_then(|arg| arg.value.as_ref())
        .and_then(|v| {
            use proto::workflow_argument_value::Kind;
            if let Some(Kind::Exception(e)) = &v.kind {
                Some((Some(e.r#type.clone()), Some(e.module.clone())))
            } else {
                None
            }
        })
        .unwrap_or((None, None));

    DecodedResult {
        result,
        exception_type,
        exception_module,
    }
}

/// Convert proto value to JSON
fn proto_value_to_json(value: &proto::WorkflowArgumentValue) -> serde_json::Value {
    use proto::primitive_workflow_argument::Kind as PrimitiveKind;
    use proto::workflow_argument_value::Kind;

    match &value.kind {
        Some(Kind::Primitive(p)) => match &p.kind {
            Some(PrimitiveKind::StringValue(s)) => serde_json::Value::String(s.clone()),
            Some(PrimitiveKind::IntValue(i)) => serde_json::Value::Number((*i).into()),
            Some(PrimitiveKind::DoubleValue(d)) => serde_json::Number::from_f64(*d)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Some(PrimitiveKind::BoolValue(b)) => serde_json::Value::Bool(*b),
            Some(PrimitiveKind::NullValue(_)) => serde_json::Value::Null,
            None => serde_json::Value::Null,
        },
        Some(Kind::ListValue(list)) => {
            serde_json::Value::Array(list.items.iter().map(proto_value_to_json).collect())
        }
        Some(Kind::TupleValue(tuple)) => {
            serde_json::Value::Array(tuple.items.iter().map(proto_value_to_json).collect())
        }
        Some(Kind::DictValue(dict)) => {
            let map: serde_json::Map<String, serde_json::Value> = dict
                .entries
                .iter()
                .filter_map(|entry| {
                    entry
                        .value
                        .as_ref()
                        .map(|v| (entry.key.clone(), proto_value_to_json(v)))
                })
                .collect();
            serde_json::Value::Object(map)
        }
        Some(Kind::Basemodel(model)) => {
            if let Some(data) = &model.data {
                let data_map: serde_json::Map<String, serde_json::Value> = data
                    .entries
                    .iter()
                    .filter_map(|entry| {
                        entry
                            .value
                            .as_ref()
                            .map(|v| (entry.key.clone(), proto_value_to_json(v)))
                    })
                    .collect();
                serde_json::Value::Object(data_map)
            } else {
                serde_json::Value::Null
            }
        }
        _ => serde_json::Value::Null,
    }
}

/// GRPC server for workflow registration during benchmarks
struct BenchmarkRegistrationServer {
    addr: std::net::SocketAddr,
    handle: JoinHandle<()>,
}

impl BenchmarkRegistrationServer {
    async fn start(store: Arc<Store>) -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let incoming = TcpListenerStream::new(listener);

        let service = BenchmarkWorkflowService { store };
        let handle = tokio::spawn(async move {
            let _ = Server::builder()
                .add_service(WorkflowServiceServer::new(service))
                .serve_with_incoming(incoming)
                .await;
        });

        Ok(Self { addr, handle })
    }

    fn port(&self) -> u16 {
        self.addr.port()
    }
}

impl Drop for BenchmarkRegistrationServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

#[derive(Clone)]
struct BenchmarkWorkflowService {
    store: Arc<Store>,
}

#[async_trait]
impl proto::workflow_service_server::WorkflowService for BenchmarkWorkflowService {
    async fn register_workflow(
        &self,
        request: Request<proto::RegisterWorkflowRequest>,
    ) -> Result<GrpcResponse<proto::RegisterWorkflowResponse>, Status> {
        let inner = request.into_inner();
        let registration = inner
            .registration
            .ok_or_else(|| Status::invalid_argument("registration missing"))?;

        let (version_id, instance_id) = self
            .store
            .register_workflow(&registration)
            .await
            .map_err(|e| Status::internal(format!("{:?}", e)))?;

        Ok(GrpcResponse::new(proto::RegisterWorkflowResponse {
            workflow_version_id: version_id.to_string(),
            workflow_instance_id: instance_id.to_string(),
        }))
    }

    async fn wait_for_instance(
        &self,
        _request: Request<proto::WaitForInstanceRequest>,
    ) -> Result<GrpcResponse<proto::WaitForInstanceResponse>, Status> {
        Err(Status::unimplemented("not used in benchmarks"))
    }
}
