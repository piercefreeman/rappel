//! Start Workers - Runs the core runloop with Python worker pool.
//!
//! This binary starts the worker infrastructure:
//! - Connects to the database
//! - Starts the WorkerBridge gRPC server for worker connections
//! - Spawns a pool of Python workers
//! - Runs the core runloop to process queued workflow instances
//! - Optionally starts the scheduler and web dashboard
//!
//! Configuration is via environment variables:
//! - RAPPEL_DATABASE_URL: PostgreSQL connection string (required)
//! - RAPPEL_WORKER_GRPC_ADDR: gRPC server for worker connections (default: 127.0.0.1:24118)
//! - RAPPEL_USER_MODULE: Python module(s) to preload (comma-separated)
//! - RAPPEL_WORKER_COUNT: Number of workers (default: num_cpus)
//! - RAPPEL_CONCURRENT_PER_WORKER: Max concurrent actions per worker (default: 10)
//! - RAPPEL_POLL_INTERVAL_MS: Poll interval for queued instances (default: 100)
//! - RAPPEL_INSTANCE_CLAIM_BATCH_SIZE: Instances per poll (default: 50)
//! - RAPPEL_INSTANCE_DONE_BATCH_SIZE: Instance completion flush batch size (default: claim size)
//! - RAPPEL_PERSIST_INTERVAL_MS: Result persistence tick (default: 500)
//! - RAPPEL_MAX_ACTION_LIFECYCLE: Max actions per worker before recycling
//! - RAPPEL_SCHEDULER_ENABLED: Enable scheduler loop
//! - RAPPEL_SCHEDULER_POLL_INTERVAL_MS: Scheduler poll interval (default: 1000)
//! - RAPPEL_SCHEDULER_BATCH_SIZE: Scheduler batch size (default: 100)
//! - RAPPEL_WEBAPP_ENABLED / RAPPEL_WEBAPP_ADDR: Web dashboard configuration

use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use prost::Message;
use sqlx::Row;
use tokio::signal;
use tokio::sync::watch;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use rappel::messages::ast as ir;
use rappel::rappel_core::backends::PostgresBackend;
use rappel::rappel_core::dag::{DAG, convert_to_dag};
use rappel::rappel_core::runloop::RunLoop;
use rappel::{
    PythonWorkerConfig, PythonWorkerPool, RemoteWorkerPool, SchedulerConfig, SchedulerDatabase,
    WebappConfig, WebappDatabase, WebappServer, WorkerBridgeServer, spawn_scheduler,
};

const DEFAULT_WORKER_GRPC_ADDR: &str = "127.0.0.1:24118";
type DagResolver = Arc<dyn Fn(&str) -> Option<DAG> + Send + Sync>;

#[derive(Debug, Clone)]
struct WorkerConfig {
    database_url: String,
    worker_grpc_addr: SocketAddr,
    worker_count: usize,
    concurrent_per_worker: usize,
    user_modules: Vec<String>,
    max_action_lifecycle: Option<u64>,
    poll_interval: Duration,
    instance_batch_size: usize,
    instance_done_batch_size: Option<usize>,
    persistence_interval: Duration,
    scheduler_enabled: bool,
    scheduler_poll_interval: Duration,
    scheduler_batch_size: i32,
}

impl WorkerConfig {
    fn from_env() -> Result<Self> {
        let database_url =
            env::var("RAPPEL_DATABASE_URL").context("RAPPEL_DATABASE_URL must be set")?;

        let worker_grpc_addr = env::var("RAPPEL_WORKER_GRPC_ADDR")
            .unwrap_or_else(|_| DEFAULT_WORKER_GRPC_ADDR.to_string())
            .parse()
            .context("invalid RAPPEL_WORKER_GRPC_ADDR")?;

        let worker_count = env_usize("RAPPEL_WORKER_COUNT").unwrap_or_else(default_worker_count);

        let concurrent_per_worker = env_usize("RAPPEL_CONCURRENT_PER_WORKER")
            .or_else(|| env_usize("RAPPEL_WORKER_CONCURRENCY"))
            .unwrap_or(10);

        let user_modules = env::var("RAPPEL_USER_MODULE")
            .ok()
            .map(parse_modules)
            .unwrap_or_default();

        let max_action_lifecycle = env_u64("RAPPEL_MAX_ACTION_LIFECYCLE");

        let poll_interval =
            Duration::from_millis(env_u64("RAPPEL_POLL_INTERVAL_MS").unwrap_or(100));

        let instance_batch_size = env_usize("RAPPEL_INSTANCE_BATCH_SIZE")
            .or_else(|| env_usize("RAPPEL_INSTANCE_CLAIM_BATCH_SIZE"))
            .unwrap_or(50);

        let instance_done_batch_size = env_usize("RAPPEL_INSTANCE_DONE_BATCH_SIZE");

        let persistence_interval =
            Duration::from_millis(env_u64("RAPPEL_PERSIST_INTERVAL_MS").unwrap_or(500));

        let scheduler_enabled = env_bool("RAPPEL_SCHEDULER_ENABLED");
        let scheduler_poll_interval =
            Duration::from_millis(env_u64("RAPPEL_SCHEDULER_POLL_INTERVAL_MS").unwrap_or(1000));
        let scheduler_batch_size = env_i32("RAPPEL_SCHEDULER_BATCH_SIZE").unwrap_or(100);

        Ok(Self {
            database_url,
            worker_grpc_addr,
            worker_count,
            concurrent_per_worker,
            user_modules,
            max_action_lifecycle,
            poll_interval,
            instance_batch_size,
            instance_done_batch_size,
            persistence_interval,
            scheduler_enabled,
            scheduler_poll_interval,
            scheduler_batch_size,
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rappel=info,start_workers=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = WorkerConfig::from_env()?;

    info!(
        worker_count = config.worker_count,
        concurrent_per_worker = config.concurrent_per_worker,
        user_modules = ?config.user_modules,
        max_action_lifecycle = ?config.max_action_lifecycle,
        "starting worker infrastructure"
    );

    let backend = PostgresBackend::connect(&config.database_url)
        .await
        .map_err(|err| anyhow::anyhow!(err))?;
    ensure_workflow_schema(backend.pool()).await?;

    let worker_bridge = WorkerBridgeServer::start(Some(config.worker_grpc_addr)).await?;
    info!(addr = %worker_bridge.addr(), "worker bridge started");

    let mut worker_config = PythonWorkerConfig::new();
    if !config.user_modules.is_empty() {
        worker_config = worker_config.with_user_modules(config.user_modules.clone());
    }

    let worker_pool = Arc::new(
        PythonWorkerPool::new_with_concurrency(
            worker_config,
            config.worker_count,
            Arc::clone(&worker_bridge),
            config.max_action_lifecycle,
            config.concurrent_per_worker,
        )
        .await?,
    );
    info!(count = config.worker_count, "python worker pool started");

    let remote_pool = RemoteWorkerPool::new(Arc::clone(&worker_pool));

    let webapp_config = WebappConfig::from_env();
    let webapp_db = WebappDatabase::new(backend.pool().clone());
    let webapp_server = WebappServer::start(webapp_config, webapp_db).await?;

    let (scheduler_handle, scheduler_shutdown) = if config.scheduler_enabled {
        let scheduler_db = SchedulerDatabase::new(backend.pool().clone());
        scheduler_db
            .ensure_schema()
            .await
            .map_err(|err| anyhow::anyhow!(err))?;
        let scheduler_config = SchedulerConfig {
            poll_interval: config.scheduler_poll_interval,
            batch_size: config.scheduler_batch_size,
        };
        let dag_resolver = build_dag_resolver(backend.pool().clone());
        let (handle, shutdown_tx) = spawn_scheduler(
            scheduler_db,
            backend.clone(),
            scheduler_config,
            dag_resolver,
        );
        info!(
            poll_interval_ms = config.scheduler_poll_interval.as_millis(),
            batch_size = config.scheduler_batch_size,
            "scheduler task started"
        );
        (Some(handle), Some(shutdown_tx))
    } else {
        (None, None)
    };

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let shutdown_handle = tokio::spawn({
        let shutdown_tx = shutdown_tx.clone();
        let scheduler_shutdown = scheduler_shutdown.clone();
        async move {
            if let Err(err) = wait_for_shutdown().await {
                error!(error = %err, "shutdown signal listener failed");
                return;
            }
            info!("shutdown signal received");
            let _ = shutdown_tx.send(true);
            if let Some(tx) = scheduler_shutdown {
                let _ = tx.send(true);
            }
        }
    });

    runloop_supervisor(
        backend.clone(),
        remote_pool,
        config.instance_batch_size,
        config.instance_done_batch_size,
        config.poll_interval,
        config.persistence_interval,
        shutdown_rx,
    )
    .await;

    let _ = shutdown_handle.await;
    if let Some(handle) = scheduler_handle {
        let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
    }

    match Arc::try_unwrap(worker_pool) {
        Ok(pool) => {
            pool.shutdown().await?;
        }
        Err(_) => {
            warn!("worker pool still referenced during shutdown; skipping shutdown");
        }
    }

    worker_bridge.shutdown().await;

    if let Some(webapp) = webapp_server {
        webapp.shutdown().await;
    }

    info!("shutdown complete");
    Ok(())
}

async fn runloop_supervisor(
    backend: PostgresBackend,
    worker_pool: RemoteWorkerPool,
    instance_batch_size: usize,
    instance_done_batch_size: Option<usize>,
    poll_interval: Duration,
    persistence_interval: Duration,
    shutdown_rx: watch::Receiver<bool>,
) {
    let mut backoff = Duration::from_millis(200);
    let max_backoff = Duration::from_secs(5);

    loop {
        if *shutdown_rx.borrow() {
            break;
        }

        let mut runloop = RunLoop::new(
            worker_pool.clone(),
            backend.clone(),
            instance_batch_size,
            instance_done_batch_size,
            poll_interval.as_secs_f64(),
            persistence_interval.as_secs_f64(),
        );

        let result = runloop.run().await;

        if *shutdown_rx.borrow() {
            break;
        }

        match result {
            Ok(_) => {
                backoff = Duration::from_millis(200);
                if poll_interval > Duration::ZERO {
                    tokio::time::sleep(poll_interval).await;
                } else {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
            Err(err) => {
                error!(error = %err, "runloop exited with error; restarting");
                tokio::time::sleep(backoff).await;
                backoff = std::cmp::min(backoff * 2, max_backoff);
            }
        }
    }
}

fn build_dag_resolver(pool: sqlx::PgPool) -> DagResolver {
    Arc::new(move |workflow_name| {
        let pool = pool.clone();
        let workflow_name = workflow_name.to_string();
        tokio::task::block_in_place(|| {
            let handle = tokio::runtime::Handle::current();
            handle.block_on(async move {
                let row = sqlx::query(
                    r#"
                    SELECT program_proto
                    FROM workflow_versions
                    WHERE workflow_name = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                    "#,
                )
                .bind(&workflow_name)
                .fetch_optional(&pool)
                .await
                .ok()??;

                let payload: Vec<u8> = row.get("program_proto");
                let program = ir::Program::decode(&payload[..]).ok()?;
                convert_to_dag(&program).ok()
            })
        })
    })
}

async fn ensure_workflow_schema(pool: &sqlx::PgPool) -> Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS workflow_versions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workflow_name TEXT NOT NULL,
            dag_hash TEXT NOT NULL,
            program_proto BYTEA NOT NULL,
            concurrent BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(workflow_name, dag_hash)
        )
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_workflow_versions_name ON workflow_versions(workflow_name)",
    )
    .execute(pool)
    .await?;

    Ok(())
}

fn env_bool(var: &str) -> bool {
    env::var(var)
        .ok()
        .map(|value| {
            let lowered = value.trim().to_ascii_lowercase();
            lowered == "1" || lowered == "true" || lowered == "yes"
        })
        .unwrap_or(false)
}

fn env_u64(var: &str) -> Option<u64> {
    env::var(var)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
}

fn env_usize(var: &str) -> Option<usize> {
    env::var(var)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

fn env_i32(var: &str) -> Option<i32> {
    env::var(var)
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
}

fn default_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
}

fn parse_modules(raw: String) -> Vec<String> {
    raw.split(',')
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
        .collect()
}

async fn wait_for_shutdown() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal as unix_signal};

        let mut terminate = unix_signal(SignalKind::terminate())?;
        tokio::select! {
            _ = signal::ctrl_c() => {
                info!("Ctrl+C received");
            }
            _ = terminate.recv() => {
                info!("SIGTERM received");
            }
        }
        Ok(())
    }

    #[cfg(not(unix))]
    {
        signal::ctrl_c().await?;
        info!("Ctrl+C received");
        Ok(())
    }
}
