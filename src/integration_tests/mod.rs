use std::{
    env,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use crate::{
    Database, PythonWorkerConfig, PythonWorkerPool,
    db::CompletionRecord,
    messages::proto,
    server_client::{self, ServerConfig},
    server_worker::WorkerBridgeServer,
    worker::{ActionDispatchPayload, RoundTripMetrics},
};
use anyhow::{Context, Result, anyhow};
use once_cell::sync::Lazy;
use prost::Message;
use prost_types::{Value as ProstValue, value::Kind as ProstValueKind};
use reqwest::Client;
use tokio::{sync::Mutex, task::JoinHandle, time::sleep};
mod common;
use self::common::run_in_env;
const PARTITION_ID: i32 = 91;
const INTEGRATION_MODULE: &str = "integration_module";
const INTEGRATION_MODULE_SOURCE: &str = include_str!("fixtures/integration_module.py");
const INTEGRATION_COMPLEX_MODULE: &str = include_str!("fixtures/integration_complex.py");
const INTEGRATION_EXCEPTION_MODULE: &str = include_str!("fixtures/integration_exception.py");

const REGISTER_SCRIPT: &str = r#"
import asyncio
from integration_module import IntegrationWorkflow

async def main():
    wf = IntegrationWorkflow()
    await wf.run()

asyncio.run(main())
"#;

const REGISTER_COMPLEX_SCRIPT: &str = r#"
import asyncio
from integration_complex import ComplexWorkflow

async def main():
    wf = ComplexWorkflow()
    await wf.run()

asyncio.run(main())
"#;

const REGISTER_EXCEPTION_SCRIPT: &str = r#"
import asyncio
from integration_exception import ExceptionWorkflow

async def main():
    wf = ExceptionWorkflow()
    await wf.run()

asyncio.run(main())
"#;

struct TestServer {
    http_addr: SocketAddr,
    grpc_addr: SocketAddr,
    handle: JoinHandle<Result<()>>,
}

impl TestServer {
    async fn spawn(database_url: String) -> Result<Self> {
        let http_port = reserve_port()?;
        let grpc_port = http_port + 1;
        let http_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), http_port);
        let grpc_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), grpc_port);
        let config = ServerConfig {
            http_addr,
            grpc_addr,
            database_url,
        };
        let handle = tokio::spawn(async move { server_client::run_servers(config).await });
        Ok(Self {
            http_addr,
            grpc_addr,
            handle,
        })
    }

    async fn shutdown(self) {
        self.handle.abort();
        let _ = self.handle.await;
    }
}

fn reserve_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

async fn wait_for_health(http_addr: SocketAddr) -> Result<()> {
    let client = Client::new();
    let url = format!("http://{http_addr}{}", server_client::HEALTH_PATH);
    for attempt in 0..100 {
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            Ok(resp) => {
                eprintln!("health attempt {attempt}: status {}", resp.status());
            }
            Err(err) => {
                eprintln!("health attempt {attempt} failed: {err}");
            }
        }
        sleep(Duration::from_millis(100)).await;
    }
    Err(anyhow!("server health endpoint not responding"))
}

async fn cleanup_database(db: &Database) -> Result<()> {
    sqlx::query("TRUNCATE daemon_action_ledger, workflow_instances, workflow_versions CASCADE")
        .execute(db.pool())
        .await?;
    Ok(())
}

async fn dispatch_all_actions(
    database: &Database,
    pool: &PythonWorkerPool,
    expected_actions: usize,
) -> Result<Vec<RoundTripMetrics>> {
    let mut completed = Vec::new();
    while completed.len() < expected_actions {
        let actions = database.dispatch_actions(PARTITION_ID, 16).await?;
        if actions.is_empty() {
            sleep(Duration::from_millis(50)).await;
            continue;
        }
        let mut batch_records = Vec::new();
        let mut batch_metrics = Vec::new();
        for action in actions {
            let dispatch = proto::WorkflowNodeDispatch::decode(action.dispatch_payload.as_slice())
                .context("failed to decode workflow dispatch")?;
            let payload = ActionDispatchPayload {
                action_id: action.id,
                instance_id: action.instance_id,
                sequence: action.action_seq,
                dispatch,
            };
            let worker = pool.next_worker();
            let metrics = worker.send_action(payload).await?;
            batch_records.push(to_completion_record(metrics.clone()));
            batch_metrics.push(metrics);
        }
        database.mark_actions_batch(&batch_records).await?;
        completed.extend(batch_metrics);
    }
    Ok(completed)
}

fn to_completion_record(metrics: RoundTripMetrics) -> CompletionRecord {
    CompletionRecord {
        action_id: metrics.action_id,
        success: metrics.success,
        delivery_id: metrics.delivery_id,
        result_payload: metrics.response_payload,
    }
}

fn encode_workflow_input(pairs: &[(&str, &str)]) -> Vec<u8> {
    let mut arguments = proto::WorkflowArguments {
        arguments: Vec::new(),
    };
    for (key, value) in pairs {
        arguments.arguments.push(proto::WorkflowArgument {
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
        });
    }
    arguments.encode_to_vec()
}

fn parse_result(payload: &[u8]) -> Result<Option<String>> {
    if payload.is_empty() {
        return Ok(None);
    }
    let arguments = proto::WorkflowArguments::decode(payload)
        .map_err(|err| anyhow!("decode workflow arguments: {err}"))?;
    for argument in arguments.arguments {
        if argument.key == "result"
            && let Some(value) = argument.value.as_ref()
        {
            return decode_argument_value(value);
        }
        if argument.key == "error"
            && let Some(value) = argument.value.as_ref()
        {
            return Ok(extract_string_from_value(value));
        }
    }
    Err(anyhow!("missing result payload"))
}

fn decode_argument_value(value: &proto::WorkflowArgumentValue) -> Result<Option<String>> {
    use proto::workflow_argument_value::Kind;
    match value.kind.as_ref() {
        Some(Kind::Primitive(primitive)) => Ok(primitive_value_to_string(primitive)),
        Some(Kind::Basemodel(model)) => {
            if let Some(struct_data) = model.data.as_ref()
                && let Some(variables) = struct_data.fields.get("variables")
                && let Some(ProstValueKind::StructValue(struct_value)) = variables.kind.as_ref()
            {
                for entry in struct_value.fields.values() {
                    if let Some(result) = extract_string_from_prost(entry) {
                        return Ok(Some(result));
                    }
                }
            }
            Ok(None)
        }
        Some(Kind::Exception(err)) => Ok(Some(err.message.clone())),
        Some(Kind::ListValue(list)) => {
            for entry in &list.items {
                if let Some(result) = decode_argument_value(entry)? {
                    return Ok(Some(result));
                }
            }
            Ok(None)
        }
        Some(Kind::TupleValue(list)) => {
            for entry in &list.items {
                if let Some(result) = decode_argument_value(entry)? {
                    return Ok(Some(result));
                }
            }
            Ok(None)
        }
        Some(Kind::DictValue(dict)) => {
            for entry in &dict.entries {
                if let Some(value) = entry.value.as_ref()
                    && let Some(result) = decode_argument_value(value)?
                {
                    return Ok(Some(result));
                }
            }
            Ok(None)
        }
        None => Ok(None),
    }
}

fn extract_string_from_value(value: &proto::WorkflowArgumentValue) -> Option<String> {
    decode_argument_value(value).ok().flatten()
}

fn primitive_value_to_string(value: &proto::PrimitiveWorkflowArgument) -> Option<String> {
    use proto::primitive_workflow_argument::Kind;
    match value.kind.as_ref()? {
        Kind::StringValue(text) => Some(text.clone()),
        Kind::DoubleValue(number) => Some(number.to_string()),
        Kind::IntValue(number) => Some(number.to_string()),
        Kind::BoolValue(flag) => Some(flag.to_string()),
        Kind::NullValue(_) => None,
    }
}

fn extract_string_from_prost(value: &ProstValue) -> Option<String> {
    match value.kind.as_ref()? {
        ProstValueKind::StringValue(text) => Some(text.clone()),
        ProstValueKind::StructValue(struct_value) => {
            for entry in struct_value.fields.values() {
                if let Some(result) = extract_string_from_prost(entry) {
                    return Some(result);
                }
            }
            None
        }
        ProstValueKind::ListValue(list_value) => {
            for entry in &list_value.values {
                if let Some(result) = extract_string_from_prost(entry) {
                    return Some(result);
                }
            }
            None
        }
        _ => None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn workflow_executes_end_to_end() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let _ = dotenvy::dotenv();
    // Run these integration tests serially so the shared temp python envs don't race
    // and unload worker modules mid-run. Once workers isolate their PYTHONPATH we can drop this lock.
    let _test_lock = TEST_SERIAL_GUARD.lock().await;
    let database_url = match env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            eprintln!("skipping integration test: DATABASE_URL not set");
            return Ok(());
        }
    };
    let database = Database::connect(&database_url).await?;
    cleanup_database(&database).await?;

    let server = TestServer::spawn(database_url.clone()).await?;
    wait_for_health(server.http_addr).await?;

    let files = vec![
        ("integration_module.py", INTEGRATION_MODULE_SOURCE),
        ("register.py", REGISTER_SCRIPT),
    ];
    let env_pairs = vec![
        ("CARABINER_GRPC_ADDR", server.grpc_addr.to_string()),
        ("CARABINER_SERVER_PORT", server.http_addr.port().to_string()),
        ("CARABINER_SERVER_HOST", server.http_addr.ip().to_string()),
    ];
    let python_env = run_in_env(&files, &[], &env_pairs, "register.py").await?;
    assert!(python_env.path().join("integration_module.py").exists());

    let versions = database.list_workflow_versions().await?;
    let version = versions
        .iter()
        .find(|v| v.workflow_name == "integrationworkflow")
        .context("integration workflow missing")?;
    let version_detail = database
        .load_workflow_version(version.id)
        .await?
        .context("missing workflow version detail")?;
    let expected_actions = version_detail.dag.nodes.len();

    let workflow_input = encode_workflow_input(&[("input", "world")]);
    let _instance_id = database
        .create_workflow_instance(
            PARTITION_ID,
            &version.workflow_name,
            version.id,
            Some(&workflow_input),
        )
        .await?;

    let worker_server: Arc<WorkerBridgeServer> = WorkerBridgeServer::start(None).await?;
    let worker_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join(".venv")
        .join("bin")
        .join("carabiner-worker");
    let worker_config = PythonWorkerConfig {
        script_path: worker_script,
        user_module: INTEGRATION_MODULE.to_string(),
        extra_python_paths: vec![python_env.path().to_path_buf()],
        ..PythonWorkerConfig::default()
    };
    let pool = PythonWorkerPool::new(worker_config, 1, Arc::clone(&worker_server)).await?;

    let completed = dispatch_all_actions(&database, &pool, expected_actions).await?;
    assert_eq!(completed.len(), expected_actions);

    pool.shutdown().await?;
    worker_server.shutdown().await;

    let message = completed
        .iter()
        .rev()
        .find_map(|metrics| parse_result(&metrics.response_payload).transpose())
        .transpose()?
        .context("expected primitive result")?;
    assert_eq!(message, "hello world");

    server.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn workflow_executes_complex_flow() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let _ = dotenvy::dotenv();
    let _test_lock = TEST_SERIAL_GUARD.lock().await;
    let database_url = match env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            eprintln!("skipping integration test: DATABASE_URL not set");
            return Ok(());
        }
    };
    let database = Database::connect(&database_url).await?;
    cleanup_database(&database).await?;

    let server = TestServer::spawn(database_url.clone()).await?;
    wait_for_health(server.http_addr).await?;

    let files = vec![
        ("integration_complex.py", INTEGRATION_COMPLEX_MODULE),
        ("register_complex.py", REGISTER_COMPLEX_SCRIPT),
    ];
    let env_pairs = vec![
        ("CARABINER_GRPC_ADDR", server.grpc_addr.to_string()),
        ("CARABINER_SERVER_PORT", server.http_addr.port().to_string()),
        ("CARABINER_SERVER_HOST", server.http_addr.ip().to_string()),
    ];
    let python_env = run_in_env(&files, &[], &env_pairs, "register_complex.py").await?;
    assert!(python_env.path().join("integration_complex.py").exists());

    let versions = database.list_workflow_versions().await?;
    let version = versions
        .iter()
        .find(|v| v.workflow_name == "complexworkflow")
        .context("complex workflow missing")?;
    let version_detail = database
        .load_workflow_version(version.id)
        .await?
        .context("missing complex workflow detail")?;
    let expected_actions = version_detail.dag.nodes.len();

    let complex_input = encode_workflow_input(&[("input", "unused")]);
    let instance_id = database
        .create_workflow_instance(
            PARTITION_ID,
            &version.workflow_name,
            version.id,
            Some(&complex_input),
        )
        .await?;

    let worker_server: Arc<WorkerBridgeServer> = WorkerBridgeServer::start(None).await?;
    let worker_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join(".venv")
        .join("bin")
        .join("carabiner-worker");
    let worker_config = PythonWorkerConfig {
        script_path: worker_script,
        user_module: "integration_complex".to_string(),
        extra_python_paths: vec![python_env.path().to_path_buf()],
        ..PythonWorkerConfig::default()
    };
    let pool = PythonWorkerPool::new(worker_config, 1, Arc::clone(&worker_server)).await?;

    let completed = dispatch_all_actions(&database, &pool, expected_actions).await?;
    assert_eq!(completed.len(), expected_actions);

    pool.shutdown().await?;
    worker_server.shutdown().await;

    let message = completed
        .iter()
        .rev()
        .find_map(|metrics| parse_result(&metrics.response_payload).transpose())
        .transpose()?
        .context("expected primitive result")?;
    // Workflow argument primitives travel as f64, so int math produces .0 suffix now.
    assert_eq!(message, "big:3.0,7.0");

    let stored_result: Option<Vec<u8>> =
        sqlx::query_scalar("SELECT result_payload FROM workflow_instances WHERE id = $1")
            .bind(instance_id)
            .fetch_one(database.pool())
            .await?;
    let stored_payload = stored_result.context("missing workflow result payload")?;
    let stored_message = parse_result(&stored_payload)?.context("expected primitive result")?;
    assert_eq!(stored_message, "big:3.0,7.0");

    server.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn workflow_handles_exception_flow() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let _ = dotenvy::dotenv();
    let _test_lock = TEST_SERIAL_GUARD.lock().await;
    let database_url = match env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            eprintln!("skipping integration test: DATABASE_URL not set");
            return Ok(());
        }
    };
    let database = Database::connect(&database_url).await?;
    cleanup_database(&database).await?;

    let server = TestServer::spawn(database_url.clone()).await?;
    wait_for_health(server.http_addr).await?;

    let files = vec![
        ("integration_exception.py", INTEGRATION_EXCEPTION_MODULE),
        ("register_exception.py", REGISTER_EXCEPTION_SCRIPT),
    ];
    let env_pairs = vec![
        ("CARABINER_GRPC_ADDR", server.grpc_addr.to_string()),
        ("CARABINER_SERVER_PORT", server.http_addr.port().to_string()),
        ("CARABINER_SERVER_HOST", server.http_addr.ip().to_string()),
    ];
    let python_env = run_in_env(&files, &[], &env_pairs, "register_exception.py").await?;
    assert!(python_env.path().join("integration_exception.py").exists());

    let versions = database.list_workflow_versions().await?;
    let version = versions
        .iter()
        .find(|v| v.workflow_name == "exceptionworkflow")
        .context("exception workflow missing")?;
    let version_detail = database
        .load_workflow_version(version.id)
        .await?
        .context("missing exception workflow detail")?;
    let expected_actions = version_detail.dag.nodes.len();

    let exception_input = encode_workflow_input(&[("mode", "exception")]);
    let instance_id = database
        .create_workflow_instance(
            PARTITION_ID,
            &version.workflow_name,
            version.id,
            Some(&exception_input),
        )
        .await?;

    let worker_server: Arc<WorkerBridgeServer> = WorkerBridgeServer::start(None).await?;
    let worker_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("python")
        .join(".venv")
        .join("bin")
        .join("carabiner-worker");
    let worker_config = PythonWorkerConfig {
        script_path: worker_script,
        user_module: "integration_exception".to_string(),
        extra_python_paths: vec![python_env.path().to_path_buf()],
        ..PythonWorkerConfig::default()
    };
    let pool = PythonWorkerPool::new(worker_config, 1, Arc::clone(&worker_server)).await?;

    let completed = dispatch_all_actions(&database, &pool, expected_actions).await?;
    assert_eq!(completed.len(), expected_actions);

    pool.shutdown().await?;
    worker_server.shutdown().await;

    let message = completed
        .iter()
        .rev()
        .find_map(|metrics| parse_result(&metrics.response_payload).transpose())
        .transpose()?
        .context("expected primitive result")?;
    assert_eq!(message, "handled:fallback");

    let stored_result: Option<Vec<u8>> =
        sqlx::query_scalar("SELECT result_payload FROM workflow_instances WHERE id = $1")
            .bind(instance_id)
            .fetch_one(database.pool())
            .await?;
    let stored_payload = stored_result.context("missing workflow result payload")?;
    let stored_message = parse_result(&stored_payload)?.context("expected primitive result")?;
    assert_eq!(stored_message, "handled:fallback");

    server.shutdown().await;
    Ok(())
}
static TEST_SERIAL_GUARD: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
