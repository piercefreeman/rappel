//! Benchmark CLI for performance testing.
//!
//! Runs various benchmarks against the workflow engine to measure throughput
//! and latency characteristics.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;
use sqlx::postgres::PgPoolOptions;
use std::path::PathBuf;
use std::time::Duration;

use rappel::{
    BenchmarkHarness, BenchmarkSummary, FanoutBenchmarkConfig, FanoutBenchmarkHarness,
    HarnessConfig, PythonWorkerConfig, Store, StressBenchmarkConfig, StressBenchmarkHarness,
    WorkflowBenchmarkConfig, WorkflowBenchmarkHarness,
};

#[derive(Parser)]
#[command(name = "benchmark")]
#[command(about = "Benchmark CLI for workflow engine performance testing")]
struct Cli {
    /// Database URL for PostgreSQL
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Number of worker processes
    #[arg(long, default_value = "4")]
    workers: usize,

    /// User module for Python workers
    #[arg(long, default_value = "benchmark")]
    user_module: String,

    /// Output format (json or text)
    #[arg(long, default_value = "text")]
    format: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Clone, Copy)]
enum OutputFormat {
    Json,
    Text,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            _ => Err(format!("Unknown format: {}", s)),
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Run raw action dispatch benchmark
    Actions {
        /// Total number of messages to send
        #[arg(long, default_value = "10000")]
        messages: usize,

        /// Maximum concurrent in-flight messages
        #[arg(long, default_value = "32")]
        in_flight: usize,

        /// Payload size in bytes
        #[arg(long, default_value = "4096")]
        payload_size: usize,
    },
    /// Run workflow instances benchmark
    Instances {
        /// Number of workflow instances to run
        #[arg(long, default_value = "100")]
        instances: usize,

        /// Maximum concurrent in-flight instances
        #[arg(long, default_value = "32")]
        in_flight: usize,

        /// Batch size for each workflow
        #[arg(long, default_value = "4")]
        batch_size: usize,

        /// Payload size in bytes
        #[arg(long, default_value = "1024")]
        payload_size: usize,
    },
    /// Run fan-out benchmark
    Fanout {
        /// Number of workflow instances to run
        #[arg(long, default_value = "50")]
        instances: usize,

        /// Fan-out factor (parallel actions)
        #[arg(long, default_value = "16")]
        fan_out: usize,

        /// Work intensity per action
        #[arg(long, default_value = "1000")]
        work_intensity: usize,

        /// Payload size in bytes
        #[arg(long, default_value = "1024")]
        payload_size: usize,

        /// Maximum concurrent in-flight instances
        #[arg(long, default_value = "64")]
        in_flight: usize,
    },
    /// Run stress benchmark
    Stress {
        /// Number of workflow instances to run
        #[arg(long, default_value = "50")]
        instances: usize,

        /// Fan-out factor (parallel actions)
        #[arg(long, default_value = "16")]
        fan_out: usize,

        /// Loop iterations
        #[arg(long, default_value = "8")]
        loop_iterations: usize,

        /// Work intensity per action
        #[arg(long, default_value = "1000")]
        work_intensity: usize,

        /// Payload size in bytes
        #[arg(long, default_value = "1024")]
        payload_size: usize,

        /// Maximum concurrent in-flight instances
        #[arg(long, default_value = "64")]
        in_flight: usize,
    },
}

#[derive(Serialize)]
struct BenchmarkOutput {
    benchmark: String,
    total_messages: usize,
    elapsed_seconds: f64,
    throughput_per_sec: f64,
    avg_ack_ms: f64,
    avg_round_trip_ms: f64,
    worker_avg_ms: f64,
    p95_round_trip_ms: f64,
}

impl BenchmarkOutput {
    fn from_summary(benchmark: &str, summary: &BenchmarkSummary) -> Self {
        Self {
            benchmark: benchmark.to_string(),
            total_messages: summary.total_messages,
            elapsed_seconds: summary.elapsed.as_secs_f64(),
            throughput_per_sec: summary.throughput_per_sec,
            avg_ack_ms: summary.avg_ack_ms,
            avg_round_trip_ms: summary.avg_round_trip_ms,
            worker_avg_ms: summary.worker_avg_ms,
            p95_round_trip_ms: summary.p95_round_trip_ms,
        }
    }

    fn print(&self, format: OutputFormat) {
        match format {
            OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(self).unwrap());
            }
            OutputFormat::Text => {
                println!("\n=== {} Benchmark Results ===", self.benchmark);
                println!("  Total messages: {}", self.total_messages);
                println!("  Elapsed: {:.2}s", self.elapsed_seconds);
                println!("  Throughput: {:.0} msg/s", self.throughput_per_sec);
                println!("  Avg ack latency: {:.3} ms", self.avg_ack_ms);
                println!("  Avg round trip: {:.3} ms", self.avg_round_trip_ms);
                println!("  Avg worker time: {:.3} ms", self.worker_avg_ms);
                println!("  P95 round trip: {:.3} ms", self.p95_round_trip_ms);
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("carabiner=info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    // Connect to database
    let pool = PgPoolOptions::new()
        .max_connections(20)
        .connect(&cli.database_url)
        .await
        .context("Failed to connect to database")?;

    let store = Store::new(pool);

    // Build worker config
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let worker_script = repo_root
        .join("python")
        .join(".venv")
        .join("bin")
        .join("rappel-worker");

    let worker_config = PythonWorkerConfig {
        script_path: worker_script,
        script_args: Vec::new(),
        user_module: cli.user_module.clone(),
        extra_python_paths: Vec::new(),
    };

    match cli.command {
        Commands::Actions {
            messages,
            in_flight,
            payload_size,
        } => {
            println!("Running actions benchmark...");
            println!(
                "  messages={}, in_flight={}, payload_size={}",
                messages, in_flight, payload_size
            );

            let harness = BenchmarkHarness::new(worker_config, cli.workers, store).await?;

            let config = HarnessConfig {
                total_messages: messages,
                in_flight,
                payload_size,
                progress_interval: Some(Duration::from_secs(5)),
            };

            let summary = harness.run(&config).await?;
            harness.shutdown().await?;

            let output = BenchmarkOutput::from_summary("Actions", &summary);
            output.print(cli.format);
        }

        Commands::Instances {
            instances,
            in_flight,
            batch_size,
            payload_size,
        } => {
            println!("Running instances benchmark...");
            println!(
                "  instances={}, in_flight={}, batch_size={}, payload_size={}",
                instances, in_flight, batch_size, payload_size
            );

            let harness = WorkflowBenchmarkHarness::new(store, cli.workers, worker_config).await?;

            let config = WorkflowBenchmarkConfig {
                instance_count: instances,
                in_flight,
                batch_size,
                payload_size,
                progress_interval: Some(Duration::from_secs(5)),
            };

            let summary = harness.run(&config).await?;
            let actions_per_instance = harness.actions_per_instance();
            harness.shutdown().await?;

            let output = BenchmarkOutput::from_summary("Instances", &summary);
            output.print(cli.format);
            println!("  Actions per instance: {}", actions_per_instance);
        }

        Commands::Fanout {
            instances,
            fan_out,
            work_intensity,
            payload_size,
            in_flight,
        } => {
            println!("Running fanout benchmark...");
            println!(
                "  instances={}, fan_out={}, work_intensity={}, payload_size={}, in_flight={}",
                instances, fan_out, work_intensity, payload_size, in_flight
            );

            let harness = FanoutBenchmarkHarness::new(store, cli.workers, worker_config).await?;

            let config = FanoutBenchmarkConfig {
                instance_count: instances,
                fan_out_factor: fan_out,
                work_intensity,
                payload_size,
                in_flight,
                progress_interval: Some(Duration::from_secs(5)),
            };

            let summary = harness.run(&config).await?;
            let actions_per_instance = harness.actions_per_instance();
            harness.shutdown().await?;

            let output = BenchmarkOutput::from_summary("Fanout", &summary);
            output.print(cli.format);
            println!("  Actions per instance: {}", actions_per_instance);
        }

        Commands::Stress {
            instances,
            fan_out,
            loop_iterations,
            work_intensity,
            payload_size,
            in_flight,
        } => {
            println!("Running stress benchmark...");
            println!(
                "  instances={}, fan_out={}, loop_iterations={}, work_intensity={}, payload_size={}, in_flight={}",
                instances, fan_out, loop_iterations, work_intensity, payload_size, in_flight
            );

            let harness = StressBenchmarkHarness::new(store, cli.workers, worker_config).await?;

            let config = StressBenchmarkConfig {
                instance_count: instances,
                fan_out_factor: fan_out,
                loop_iterations,
                work_intensity,
                payload_size,
                in_flight,
                progress_interval: Some(Duration::from_secs(5)),
            };

            let summary = harness.run(&config).await?;
            let actions_per_instance = harness.actions_per_instance();
            harness.shutdown().await?;

            let output = BenchmarkOutput::from_summary("Stress", &summary);
            output.print(cli.format);
            println!("  Actions per instance: {}", actions_per_instance);
        }
    }

    Ok(())
}
