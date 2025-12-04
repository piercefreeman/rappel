#![allow(clippy::collapsible_if)]
//! Boot Rappel Singleton - Ensures a single server instance is running.
//!
//! This binary:
//! 1. Probes a range of ports to find an existing Rappel server
//! 2. If found, outputs the existing server's port
//! 3. If not found, spawns a new server and outputs its port
//!
//! Usage:
//!   boot-rappel-singleton [--port-file <path>]
//!
//! The port file will contain the HTTP port number on success.

use std::{
    env, fs,
    path::PathBuf,
    process::{Command, Stdio},
    time::Duration,
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use tracing::{debug, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Default base port for probing
const DEFAULT_BASE_PORT: u16 = 24117;

/// Number of ports to probe
const PORT_PROBE_COUNT: u16 = 10;

/// Health check timeout
const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);

/// Startup wait time for new server
const STARTUP_WAIT: Duration = Duration::from_secs(5);

#[derive(Debug, Deserialize)]
struct HealthResponse {
    #[allow(dead_code)]
    status: String,
    service: String,
    http_port: u16,
    grpc_port: u16,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "boot_rappel_singleton=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse arguments
    let args: Vec<String> = env::args().collect();
    let port_file = parse_port_file_arg(&args);

    // Try to find existing server
    let base_port = env::var("RAPPEL_BASE_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BASE_PORT);

    info!(base_port, "probing for existing rappel server");

    if let Some(health) = probe_existing_server(base_port).await {
        info!(
            port = health.http_port,
            grpc_port = health.grpc_port,
            "found existing rappel server"
        );
        write_port_file(&port_file, health.http_port)?;
        return Ok(());
    }

    info!("no existing server found, spawning new instance");

    // Spawn new server
    let http_port = spawn_server(base_port).await?;

    info!(port = http_port, "rappel server started");
    write_port_file(&port_file, http_port)?;

    Ok(())
}

fn parse_port_file_arg(args: &[String]) -> Option<PathBuf> {
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        // Support both --port-file and --output-file for compatibility
        if arg == "--port-file" || arg == "--output-file" {
            return iter.next().map(PathBuf::from);
        }
    }
    None
}

async fn probe_existing_server(base_port: u16) -> Option<HealthResponse> {
    let client = reqwest::Client::builder()
        .timeout(HEALTH_TIMEOUT)
        .build()
        .ok()?;

    for offset in 0..PORT_PROBE_COUNT {
        let port = base_port + offset;
        let url = format!("http://127.0.0.1:{port}/healthz");

        debug!(port, "probing health endpoint");

        match client.get(&url).send().await {
            Ok(response) if response.status().is_success() => {
                if let Ok(health) = response.json::<HealthResponse>().await {
                    if health.service == "rappel" {
                        return Some(health);
                    }
                }
            }
            Ok(_) => {}
            Err(e) => {
                debug!(port, error = %e, "probe failed");
            }
        }
    }

    None
}

async fn spawn_server(base_port: u16) -> Result<u16> {
    let http_addr = format!("127.0.0.1:{base_port}");
    let grpc_addr = format!("127.0.0.1:{}", base_port + 1);

    // Find the rappel-server binary
    let server_bin = find_server_binary()?;

    info!(?server_bin, "spawning rappel-server");

    // Spawn the server process
    let mut cmd = Command::new(&server_bin);
    cmd.env("RAPPEL_HTTP_ADDR", &http_addr)
        .env("RAPPEL_GRPC_ADDR", &grpc_addr)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    // Detach from parent process group on Unix
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let _child = cmd.spawn().context("failed to spawn rappel-server")?;

    // Wait for server to start
    let client = reqwest::Client::builder().timeout(HEALTH_TIMEOUT).build()?;

    let deadline = tokio::time::Instant::now() + STARTUP_WAIT;
    let url = format!("http://127.0.0.1:{base_port}/healthz");

    while tokio::time::Instant::now() < deadline {
        if let Ok(response) = client.get(&url).send().await {
            if response.status().is_success() {
                if let Ok(health) = response.json::<HealthResponse>().await {
                    if health.service == "rappel" {
                        return Ok(health.http_port);
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    bail!("server failed to start within {STARTUP_WAIT:?}")
}

fn find_server_binary() -> Result<PathBuf> {
    // Try to find in same directory as this binary
    let current_exe = env::current_exe().context("failed to get current executable path")?;
    let parent = current_exe
        .parent()
        .context("failed to get parent directory")?;

    let binary_name = if cfg!(windows) {
        "rappel-server.exe"
    } else {
        "rappel-server"
    };

    let candidate = parent.join(binary_name);
    if candidate.exists() {
        return Ok(candidate);
    }

    // Try cargo target directory
    let target_dirs = ["debug", "release"];
    for dir in &target_dirs {
        let candidate = PathBuf::from(format!("target/{dir}/{binary_name}"));
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    bail!("could not find rappel-server binary")
}

fn write_port_file(port_file: &Option<PathBuf>, port: u16) -> Result<()> {
    if let Some(path) = port_file {
        fs::write(path, port.to_string())
            .with_context(|| format!("failed to write port file: {}", path.display()))?;
        info!(path = %path.display(), port, "wrote port file");
    } else {
        println!("{port}");
    }
    Ok(())
}
