//! Main entry point for the rappel server.
//!
//! Starts:
//! - HTTP server for workflow management
//! - gRPC server for action workers
//! - gRPC server for instance workers
//! - Worker pools for both worker types

use std::sync::Arc;

use tokio::net::TcpListener;
use tonic::transport::Server as TonicServer;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use rappel::{
    config::Config,
    db::Database,
    messages::{
        action_worker_bridge_server::ActionWorkerBridgeServer,
        instance_worker_bridge_server::InstanceWorkerBridgeServer,
    },
    server::{
        action_bridge::{ActionWorkerBridgeService, ActionWorkerBridgeState},
        http::{create_router, HttpState},
        instance_bridge::{InstanceWorkerBridgeService, InstanceWorkerBridgeState},
    },
    worker::{WorkerPool, WorkerType},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting rappel server");

    // Load configuration
    let config = Config::from_env()?;
    info!(?config, "Loaded configuration");

    // Connect to database
    let db = Arc::new(Database::connect(&config.database_url).await?);
    info!("Connected to database");

    // Run migrations
    db.migrate().await?;
    info!("Database migrations complete");

    // Create shared state
    let action_bridge_state = Arc::new(ActionWorkerBridgeState::new());
    let instance_bridge_state = Arc::new(InstanceWorkerBridgeState::new());

    // Create gRPC services
    let action_bridge_service = ActionWorkerBridgeService::new(action_bridge_state.clone(), db.clone());
    let instance_bridge_service = InstanceWorkerBridgeService::new(instance_bridge_state.clone(), db.clone());

    // Create HTTP router
    let http_state = HttpState { db: db.clone() };
    let http_router = create_router(http_state);

    // Create worker pools
    let action_worker_pool = Arc::new(WorkerPool::new(
        WorkerType::Action,
        config.action_grpc_addr.to_string(),
        config.user_modules.clone(),
        config.action_worker_count,
    ));

    let instance_worker_pool = Arc::new(WorkerPool::new(
        WorkerType::Instance,
        config.instance_grpc_addr.to_string(),
        config.user_modules.clone(),
        config.instance_worker_count,
    ));

    // Spawn HTTP server
    let http_addr = config.http_addr;
    let http_handle = tokio::spawn(async move {
        info!(%http_addr, "Starting HTTP server");
        let listener = TcpListener::bind(http_addr).await?;
        axum::serve(listener, http_router).await?;
        Ok::<_, anyhow::Error>(())
    });

    // Spawn action worker gRPC server
    let action_grpc_addr = config.action_grpc_addr;
    let action_grpc_handle = tokio::spawn(async move {
        info!(%action_grpc_addr, "Starting action worker gRPC server");
        TonicServer::builder()
            .add_service(ActionWorkerBridgeServer::new(action_bridge_service))
            .serve(action_grpc_addr)
            .await?;
        Ok::<_, anyhow::Error>(())
    });

    // Spawn instance worker gRPC server
    let instance_grpc_addr = config.instance_grpc_addr;
    let instance_grpc_handle = tokio::spawn(async move {
        info!(%instance_grpc_addr, "Starting instance worker gRPC server");
        TonicServer::builder()
            .add_service(InstanceWorkerBridgeServer::new(instance_bridge_service))
            .serve(instance_grpc_addr)
            .await?;
        Ok::<_, anyhow::Error>(())
    });

    // Start worker pools
    info!("Starting worker pools");
    action_worker_pool.start().await?;
    instance_worker_pool.start().await?;

    // Spawn worker maintenance task
    let action_pool = action_worker_pool.clone();
    let instance_pool = instance_worker_pool.clone();
    let maintenance_handle = tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            if let Err(e) = action_pool.maintain().await {
                error!(error = %e, "Failed to maintain action worker pool");
            }
            if let Err(e) = instance_pool.maintain().await {
                error!(error = %e, "Failed to maintain instance worker pool");
            }
        }
    });

    info!("Rappel server started");

    // Wait for any task to fail
    tokio::select! {
        result = http_handle => {
            error!(?result, "HTTP server exited");
        }
        result = action_grpc_handle => {
            error!(?result, "Action gRPC server exited");
        }
        result = instance_grpc_handle => {
            error!(?result, "Instance gRPC server exited");
        }
        _ = maintenance_handle => {
            error!("Maintenance task exited");
        }
    }

    // Cleanup
    action_worker_pool.stop().await;
    instance_worker_pool.stop().await;

    Ok(())
}
