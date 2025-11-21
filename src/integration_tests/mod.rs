use std::{
    env,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Duration,
};

use crate::{
    Database, PythonWorkerPool,
    db::CompletionRecord,
    messages::proto,
    server_client::{self, ServerConfig},
    worker::{ActionDispatchPayload, RoundTripMetrics},
};
use anyhow::{Context, Result, anyhow};
use prost::Message;
use reqwest::Client;
use serial_test::serial;
use tokio::{task::JoinHandle, time::sleep};
mod common;
mod harness;
use self::harness::{WorkflowHarness, WorkflowHarnessConfig};
const INTEGRATION_MODULE: &str = "integration_module";
const INTEGRATION_MODULE_SOURCE: &str = include_str!("fixtures/integration_module.py");
const INTEGRATION_COMPLEX_MODULE: &str = include_str!("fixtures/integration_complex.py");
const INTEGRATION_LOOP_MODULE: &str = "integration_loop";
const INTEGRATION_LOOP_MODULE_SOURCE: &str = include_str!("fixtures/integration_loop.py");
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

const REGISTER_LOOP_SCRIPT: &str = r#"
import asyncio
from integration_loop import LoopWorkflow

async def main():
    wf = LoopWorkflow()
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

async fn purge_empty_input_instances(db: &Database) -> Result<()> {
    // In integration tests, just clear all instances and actions to start fresh
    sqlx::query("DELETE FROM daemon_action_ledger")
        .execute(db.pool())
        .await?;
    sqlx::query("DELETE FROM workflow_instances")
        .execute(db.pool())
        .await?;
    Ok(())
}

async fn dispatch_all_actions(
    database: &Database,
    pool: &PythonWorkerPool,
    target_actions: usize,
) -> Result<Vec<RoundTripMetrics>> {
    let mut completed = Vec::new();
    let mut max_iterations = target_actions.saturating_mul(20).max(100); // Safety limit to prevent infinite loops
    let mut idle_cycles = 0usize;
    while max_iterations > 0 {
        max_iterations -= 1;
        let actions = database.dispatch_actions(16).await?;
        if actions.is_empty() {
            idle_cycles = idle_cycles.saturating_add(1);
            if idle_cycles >= 3 && completed.len() >= target_actions {
                break;
            }
            sleep(Duration::from_millis(50)).await;
            continue;
        }
        idle_cycles = 0;
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
                timeout_seconds: action.timeout_seconds,
                max_retries: action.max_retries,
                attempt_number: action.attempt_number,
                dispatch_token: action.delivery_token,
            };
            let worker = pool.next_worker();
            let metrics = worker.send_action(payload).await?;
            if metrics.success
                && matches!(
                    metrics.control,
                    Some(proto::WorkflowNodeControl {
                        kind: Some(proto::workflow_node_control::Kind::Loop(_))
                    })
                )
                && database
                    .requeue_loop_iteration(
                        &to_completion_record(metrics.clone()),
                        metrics.control.as_ref().unwrap(),
                    )
                    .await?
            {
                continue;
            }
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
        dispatch_token: metrics.dispatch_token,
        control: metrics.control,
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
            if let Some(dict_data) = model.data.as_ref() {
                // Look for "variables" key in the dict
                if let Some(variables_entry) =
                    dict_data.entries.iter().find(|e| e.key == "variables")
                    && let Some(variables_value) = &variables_entry.value
                {
                    // Recursively decode the variables value
                    if let Some(result) = decode_argument_value(variables_value)? {
                        return Ok(Some(result));
                    }
                }
                // Also check other entries
                for entry in &dict_data.entries {
                    if let Some(entry_value) = &entry.value
                        && let Some(result) = decode_argument_value(entry_value)?
                    {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn workflow_executes_end_to_end() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let _ = dotenvy::dotenv();
    let Some(harness) = WorkflowHarness::new(WorkflowHarnessConfig {
        files: &[
            ("integration_module.py", INTEGRATION_MODULE_SOURCE),
            ("register.py", REGISTER_SCRIPT),
        ],
        entrypoint: "register.py",
        workflow_name: "integrationworkflow",
        user_module: INTEGRATION_MODULE,
        inputs: &[("input", "world")],
    })
    .await?
    else {
        return Ok(());
    };

    let completed = harness.dispatch_all().await?;
    assert!(
        completed.len() >= harness.expected_actions(),
        "expected at least {} completions, saw {}",
        harness.expected_actions(),
        completed.len()
    );

    let stored_payload = harness
        .stored_result()
        .await?
        .context("missing workflow result payload")?;
    let message = parse_result(&stored_payload)?.context("expected primitive result")?;
    assert_eq!(message, "hello world");

    harness.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn workflow_executes_complex_flow() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let _ = dotenvy::dotenv();
    let Some(harness) = WorkflowHarness::new(WorkflowHarnessConfig {
        files: &[
            ("integration_complex.py", INTEGRATION_COMPLEX_MODULE),
            ("register_complex.py", REGISTER_COMPLEX_SCRIPT),
        ],
        entrypoint: "register_complex.py",
        workflow_name: "complexworkflow",
        user_module: "integration_complex",
        inputs: &[("input", "unused")],
    })
    .await?
    else {
        return Ok(());
    };

    let completed = harness.dispatch_all().await?;
    assert!(
        completed.len() >= harness.expected_actions(),
        "expected at least {} completions, saw {}",
        harness.expected_actions(),
        completed.len()
    );

    let stored_payload = harness
        .stored_result()
        .await?
        .context("missing workflow result payload")?;
    let stored_message = parse_result(&stored_payload)?.context("expected primitive result")?;
    assert_eq!(stored_message, "big:3,7");

    harness.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn workflow_executes_looped_actions() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let _ = dotenvy::dotenv();
    let Some(harness) = WorkflowHarness::new(WorkflowHarnessConfig {
        files: &[
            ("integration_loop.py", INTEGRATION_LOOP_MODULE_SOURCE),
            ("register_loop.py", REGISTER_LOOP_SCRIPT),
        ],
        entrypoint: "register_loop.py",
        workflow_name: "loopworkflow",
        user_module: INTEGRATION_LOOP_MODULE,
        inputs: &[("input", "unused")],
    })
    .await?
    else {
        return Ok(());
    };

    let completed = harness.dispatch_all().await?;
    assert!(
        completed.len() >= harness.expected_actions(),
        "expected at least {} completions, saw {}",
        harness.expected_actions(),
        completed.len()
    );

    let stored_payload = harness
        .stored_result()
        .await?
        .context("missing workflow result payload")?;
    let parsed_result =
        parse_result(&stored_payload)?.context("expected primitive workflow result")?;
    assert_eq!(parsed_result, "alpha-local-decorated,beta-local-decorated");

    harness.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn workflow_handles_exception_flow() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let _ = dotenvy::dotenv();
    let Some(harness) = WorkflowHarness::new(WorkflowHarnessConfig {
        files: &[
            ("integration_exception.py", INTEGRATION_EXCEPTION_MODULE),
            ("register_exception.py", REGISTER_EXCEPTION_SCRIPT),
        ],
        entrypoint: "register_exception.py",
        workflow_name: "exceptionworkflow",
        user_module: "integration_exception",
        inputs: &[("mode", "exception")],
    })
    .await?
    else {
        return Ok(());
    };

    let completed = harness.dispatch_all().await?;
    assert_eq!(completed.len(), harness.expected_actions());

    let cleanup_node = harness
        .version_detail()
        .dag
        .nodes
        .iter()
        .find(|node| node.action == "cleanup")
        .context("cleanup node missing")?;
    let (cleanup_status, cleanup_success, cleanup_result): (String, bool, Option<Vec<u8>>) =
        sqlx::query_as(
            "SELECT status, success, result_payload FROM daemon_action_ledger WHERE instance_id = $1 AND workflow_node_id = $2",
        )
        .bind(harness.instance_id())
        .bind(&cleanup_node.id)
        .fetch_one(harness.database().pool())
        .await?;
    assert_eq!(cleanup_status, "completed");
    assert!(
        cleanup_success,
        "cleanup action did not succeed despite exception handling"
    );
    let cleanup_payload = cleanup_result.context("cleanup result payload missing")?;
    assert!(!cleanup_payload.is_empty(), "cleanup payload missing bytes");

    let stored_payload = harness
        .stored_result()
        .await?
        .context("missing workflow result payload")?;
    assert!(
        !stored_payload.is_empty(),
        "workflow result payload missing bytes"
    );

    harness.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn stale_worker_completion_is_ignored() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let _ = dotenvy::dotenv();
    let database_url = match env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            eprintln!("skipping integration test: DATABASE_URL not set");
            return Ok(());
        }
    };
    let database = Database::connect(&database_url).await?;
    cleanup_database(&database).await?;
    let dispatch = proto::WorkflowNodeDispatch {
        node: None,
        workflow_input: None,
        context: Vec::new(),
    };
    let payload = dispatch.encode_to_vec();
    database
        .seed_actions(1, "tests", "action", &payload)
        .await?;
    let mut actions = database.dispatch_actions(1).await?;
    let mut action = actions.pop().expect("dispatched action");
    let stale_token = action.delivery_token;
    database.requeue_action(action.id).await?;
    let mut redispatched = database.dispatch_actions(1).await?;
    action = redispatched.pop().expect("redispatched action");
    let fresh_token = action.delivery_token;

    let stale_record = CompletionRecord {
        action_id: action.id,
        success: true,
        delivery_id: 1,
        result_payload: Vec::new(),
        dispatch_token: Some(stale_token),
        control: None,
    };
    database.mark_actions_batch(&[stale_record]).await?;
    let (status, payload): (String, Option<Vec<u8>>) =
        sqlx::query_as("SELECT status, result_payload FROM daemon_action_ledger WHERE id = $1")
            .bind(action.id)
            .fetch_one(database.pool())
            .await?;
    assert_eq!(status, "dispatched");
    assert!(payload.is_none());

    let mut result_args = proto::WorkflowArguments {
        arguments: Vec::new(),
    };
    result_args.arguments.push(proto::WorkflowArgument {
        key: "result".to_string(),
        value: Some(proto::WorkflowArgumentValue {
            kind: Some(proto::workflow_argument_value::Kind::Primitive(
                proto::PrimitiveWorkflowArgument {
                    kind: Some(proto::primitive_workflow_argument::Kind::StringValue(
                        "ok".to_string(),
                    )),
                },
            )),
        }),
    });
    let valid_record = CompletionRecord {
        action_id: action.id,
        success: true,
        delivery_id: 2,
        result_payload: result_args.encode_to_vec(),
        dispatch_token: Some(fresh_token),
        control: None,
    };
    database.mark_actions_batch(&[valid_record]).await?;
    let (status, payload): (String, Option<Vec<u8>>) =
        sqlx::query_as("SELECT status, result_payload FROM daemon_action_ledger WHERE id = $1")
            .bind(action.id)
            .fetch_one(database.pool())
            .await?;
    assert_eq!(status, "completed");
    assert!(payload.is_some());
    Ok(())
}
