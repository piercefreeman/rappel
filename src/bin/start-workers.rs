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
//! - RAPPEL_MAX_CONCURRENT_INSTANCES: Max workflow instances held concurrently (default: 50)
//! - RAPPEL_INSTANCE_DONE_BATCH_SIZE: Instance completion flush batch size (default: claim size)
//! - RAPPEL_PERSIST_INTERVAL_MS: Result persistence tick (default: 500)
//! - RAPPEL_MAX_ACTION_LIFECYCLE: Max actions per worker before recycling
//! - RAPPEL_SCHEDULER_ENABLED: Enable scheduler loop
//! - RAPPEL_SCHEDULER_POLL_INTERVAL_MS: Scheduler poll interval (default: 1000)
//! - RAPPEL_SCHEDULER_BATCH_SIZE: Scheduler batch size (default: 100)
//! - RAPPEL_WEBAPP_ENABLED / RAPPEL_WEBAPP_ADDR: Web dashboard configuration
//! - RAPPEL_RUNNER_PROFILE_INTERVAL_MS: Status reporting interval (default: 5000)

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

use chrono::Utc;
use rappel::messages::ast as ir;
use rappel::pool_status::{PoolTimeSeries, TimeSeriesEntry};
use rappel::rappel_core::backends::{PostgresBackend, WorkerStatusUpdate};
use rappel::rappel_core::dag::{DAG, convert_to_dag};
use rappel::rappel_core::runloop::RunLoop;
use rappel::{
    PythonWorkerConfig, PythonWorkerPool, RemoteWorkerPool, SchedulerConfig, SchedulerDatabase,
    WebappConfig, WebappDatabase, WebappServer, WorkerBridgeServer, spawn_scheduler,
};
use uuid::Uuid;

const DEFAULT_WORKER_GRPC_ADDR: &str = "127.0.0.1:24118";
type DagResolver = Arc<dyn Fn(&str) -> Option<DAG> + Send + Sync>;

/* TODO: Move to a separate config class so we can share this potentially with multiple different bin entrypoints. Anything
 * in the rappel universe should generally be pulling from the same centralized config.
 */
#[derive(Debug, Clone)]
struct WorkerConfig {
    database_url: String,
    worker_grpc_addr: SocketAddr,
    worker_count: usize,
    concurrent_per_worker: usize,
    user_modules: Vec<String>,
    max_action_lifecycle: Option<u64>,
    poll_interval: Duration,
    max_concurrent_instances: usize,
    instance_done_batch_size: Option<usize>,
    persistence_interval: Duration,
    scheduler_enabled: bool,
    scheduler_poll_interval: Duration,
    scheduler_batch_size: i32,
    profile_interval: Option<Duration>,
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

        let max_concurrent_instances = env_usize("RAPPEL_MAX_CONCURRENT_INSTANCES").unwrap_or(50);

        let instance_done_batch_size = env_usize("RAPPEL_INSTANCE_DONE_BATCH_SIZE");

        let persistence_interval =
            Duration::from_millis(env_u64("RAPPEL_PERSIST_INTERVAL_MS").unwrap_or(500));

        let scheduler_enabled = env_bool("RAPPEL_SCHEDULER_ENABLED");
        let scheduler_poll_interval =
            Duration::from_millis(env_u64("RAPPEL_SCHEDULER_POLL_INTERVAL_MS").unwrap_or(1000));
        let scheduler_batch_size = env_i32("RAPPEL_SCHEDULER_BATCH_SIZE").unwrap_or(100);

        let profile_interval =
            env_u64("RAPPEL_RUNNER_PROFILE_INTERVAL_MS").map(Duration::from_millis);

        Ok(Self {
            database_url,
            worker_grpc_addr,
            worker_count,
            concurrent_per_worker,
            user_modules,
            max_action_lifecycle,
            poll_interval,
            max_concurrent_instances,
            instance_done_batch_size,
            persistence_interval,
            scheduler_enabled,
            scheduler_poll_interval,
            scheduler_batch_size,
            profile_interval,
        })
    }
}

/* TODO: This is a great function length that's an example of where we should add comments to indicate the sections
 * of different logic that's put together. I've tried to place it below.
 */
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

    /*
     * Configure services: db, worker pool
     */

    // TODO: We should NEVER use .map_error for these simple uses
    // instead preferring just a regular statement.await? syntax... If it
    // can't cast automatically we should modify the original error to
    // implement std::error::Error + Send + Sync + 'static instead. This
    // makes it easier to capture the error throughout the call stack
    let backend = PostgresBackend::connect(&config.database_url)
        .await
        .map_err(|err| anyhow::anyhow!(err))?;
    ensure_workflow_schema(backend.pool()).await?;

    // TODO: The python worker pool should own its own worker_bridge because it's the only thing
    // that relies on it. The rest of our services just push and pull from the worker pool.
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

    // TODO: RemoteWorkerPool should also own the worker_pool internally, the worker_pool
    // does not need to be owned locally. We can pass through any relevant init
    // configs to the constructor of RemoteWorkerPool... This constructor should be defined
    // in the base.rs because in theory it's applicable to any worker.
    let remote_pool = RemoteWorkerPool::new(Arc::clone(&worker_pool));

    // TODO: Sub-module configs should be placed in the root Config and instantiated when we instantiate
    // the main config
    let webapp_config = WebappConfig::from_env();
    let webapp_db = WebappDatabase::new(backend.pool().clone());
    let webapp_server = WebappServer::start(webapp_config, webapp_db).await?;

    // TODO: Scheduler should always be enabled, get rid of the config that
    // makes this optional
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

    // TODO: Should also be made non-optional
    // TODO: spawn_status_reporter should not require a Postgres backend. All of the usage of backend/worker/etc
    // should reference their base class instead
    // Spawn status reporter if profile interval is configured
    let status_reporter_handle = if let Some(interval) = config.profile_interval {
        let pool_id = Uuid::new_v4();
        // TODO: spawn_status_reporter should be implemented right in the workers package.
        Some(spawn_status_reporter(
            pool_id,
            backend.clone(),
            Arc::clone(&worker_pool),
            interval,
            shutdown_rx.clone(),
        ))
    } else {
        None
    };

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

    // TODO: Make the runloop export this directly like we do for some
    // of these other launch services like spawn_status_reporter
    runloop_supervisor(
        backend.clone(),
        remote_pool,
        config.max_concurrent_instances,
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
    if let Some(handle) = status_reporter_handle {
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
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
    max_concurrent_instances: usize,
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
            max_concurrent_instances,
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

// TODO: We should switch to using sqlx migration utilities to create these
// files and keeping them updated over time. All of our DB object definitions (which should
// be structs that we can easily interact with) should be specified in a `db` crate to keep
// things clean!
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

// TODO: All of these config helpers should move to the config file
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

/// Spawn a background task that reports worker status to the database.
fn spawn_status_reporter(
    pool_id: Uuid,
    backend: PostgresBackend,
    worker_pool: Arc<PythonWorkerPool>,
    interval: Duration,
    mut shutdown_rx: watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut time_series = PoolTimeSeries::new();
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        info!(
            pool_id = %pool_id,
            interval_ms = interval.as_millis(),
            "status reporter started"
        );

        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    // Gather stats from the worker pool
                    let snapshots = worker_pool.throughput_snapshots();

                    // Aggregate across all workers
                    let active_workers = snapshots.len() as i32;
                    let throughput_per_min: f64 = snapshots.iter().map(|s| s.throughput_per_min).sum();
                    let actions_per_sec = throughput_per_min / 60.0;
                    let total_completed: i64 = snapshots.iter().map(|s| s.total_completed as i64).sum();
                    let last_action_at = snapshots
                        .iter()
                        .filter_map(|s| s.last_action_at)
                        .max();

                    // Get queue stats from the pool
                    let (dispatch_queue_size, total_in_flight) = worker_pool.queue_stats();

                    // Record time series entry
                    let now = Utc::now();
                    time_series.push(TimeSeriesEntry {
                        timestamp_secs: now.timestamp(),
                        actions_per_sec: actions_per_sec as f32,
                        active_workers: active_workers as u16,
                        median_instance_duration_secs: 0.0, // Not tracked yet
                        active_instances: 0, // Not tracked yet
                        queue_depth: dispatch_queue_size as u32,
                        in_flight_actions: total_in_flight as u32,
                    });

                    let status = WorkerStatusUpdate {
                        pool_id,
                        throughput_per_min,
                        total_completed,
                        last_action_at,
                        median_dequeue_ms: None, // TODO: track median latencies
                        median_handling_ms: None,
                        dispatch_queue_size: dispatch_queue_size as i64,
                        total_in_flight: total_in_flight as i64,
                        active_workers,
                        actions_per_sec,
                        median_instance_duration_secs: None,
                        active_instance_count: 0,
                        total_instances_completed: 0,
                        instances_per_sec: 0.0,
                        instances_per_min: 0.0,
                        time_series: Some(time_series.encode()),
                    };

                    if let Err(err) = backend.upsert_worker_status(&status).await {
                        warn!(error = %err, "failed to update worker status");
                    }
                }
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        info!("status reporter shutting down");
                        break;
                    }
                }
            }
        }
    })
}
