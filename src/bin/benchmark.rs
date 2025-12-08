//! Benchmark binary for measuring Rappel throughput.
//!
//! This benchmark tests the new durable execution model by running
//! the benchmark workflow with configurable parameters.

use std::env;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::process::Command;

#[derive(Parser, Debug)]
#[command(name = "benchmark", about = "Run Rappel benchmarks")]
struct Args {
    /// Output results as JSON
    #[arg(long, default_value = "false")]
    json: bool,

    /// Number of parallel hash computations
    #[arg(long, default_value = "16")]
    count: u32,

    /// Hash iterations per action (CPU intensity)
    #[arg(long, default_value = "100")]
    iterations: u32,

    /// Number of workflow runs for averaging
    #[arg(long, default_value = "3")]
    runs: u32,
}

#[derive(Serialize, Deserialize, Debug)]
struct BenchmarkOutput {
    /// Total number of actions executed
    total_actions: u64,
    /// Total elapsed time in seconds
    elapsed_s: f64,
    /// Actions per second
    throughput: f64,
    /// Number of runs averaged
    runs: u32,
}

#[derive(Deserialize, Debug)]
struct PythonBenchmarkResult {
    total_actions: u64,
    elapsed_s: f64,
    throughput: f64,
}

async fn run_python_benchmark(count: u32, iterations: u32) -> Result<PythonBenchmarkResult> {
    let python_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");

    // Create a benchmark runner script
    let script = format!(
        r#"
import asyncio
import sys
import json
import time

sys.path.insert(0, 'src')
sys.path.insert(0, 'tests')

from rappel.durable import (
    ActionCall,
    ActionResult,
    ActionStatus,
    WorkflowInstance,
    run_until_actions,
    ExecutionContext,
    _current_context,
)
from rappel.actions import get_action_registry

from benchmark_workflow import BenchmarkWorkflow

async def execute_action(action_call):
    registry = get_action_registry()
    action_func = registry.get(action_call.func_name)
    if action_func is None:
        return ActionResult(
            action_id=action_call.id,
            status=ActionStatus.FAILED,
            error=f"Unknown action: {{action_call.func_name}}",
        )
    try:
        original = getattr(action_func, "_action_func", action_func)
        result = await original(*action_call.args, **action_call.kwargs)
        return ActionResult(
            action_id=action_call.id,
            status=ActionStatus.COMPLETED,
            result=result,
        )
    except Exception as e:
        return ActionResult(
            action_id=action_call.id,
            status=ActionStatus.FAILED,
            error=str(e),
        )

async def run_workflow_to_completion(workflow_coro_fn):
    instance = WorkflowInstance(id="benchmark-instance")
    total_actions = 0
    while True:
        pending = await run_until_actions(instance, workflow_coro_fn())
        if not pending:
            return total_actions
        for action_call in pending:
            action_result = await execute_action(action_call)
            instance.action_queue.append(action_result)
            total_actions += 1

async def main():
    workflow = BenchmarkWorkflow()
    start = time.perf_counter()
    total_actions = await run_workflow_to_completion(
        lambda: workflow.run(count={count}, iterations={iterations})
    )
    elapsed = time.perf_counter() - start
    throughput = total_actions / elapsed if elapsed > 0 else 0
    print(json.dumps({{
        "total_actions": total_actions,
        "elapsed_s": elapsed,
        "throughput": throughput,
    }}))

asyncio.run(main())
"#
    );

    let output = Command::new("uv")
        .arg("run")
        .arg("python")
        .arg("-c")
        .arg(&script)
        .current_dir(&python_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .await
        .context("Failed to run Python benchmark")?;

    if !output.status.success() {
        anyhow::bail!("Python benchmark failed with status: {}", output.status);
    }

    let stdout = String::from_utf8(output.stdout)?;
    let result: PythonBenchmarkResult =
        serde_json::from_str(&stdout).context("Failed to parse benchmark result")?;

    Ok(result)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let separator = "=".repeat(70);

    if !args.json {
        println!("{}", separator);
        println!("Rappel Durable Execution Benchmark");
        println!("{}", separator);
        println!();
        println!("Configuration:");
        println!("  Count: {} parallel hashes", args.count);
        println!("  Iterations: {} per hash", args.iterations);
        println!("  Runs: {}", args.runs);
        println!();
    }

    // Warm up
    if !args.json {
        println!("Warming up...");
    }
    let _ = run_python_benchmark(4, 10).await?;

    // Run benchmarks
    let mut total_actions = 0u64;
    let mut total_elapsed = 0.0f64;

    let start = Instant::now();

    for run in 1..=args.runs {
        if !args.json {
            println!("Run {}/{}...", run, args.runs);
        }

        let result = run_python_benchmark(args.count, args.iterations).await?;
        total_actions += result.total_actions;
        total_elapsed += result.elapsed_s;

        if !args.json {
            println!(
                "  Actions: {}, Time: {:.3}s, Throughput: {:.1} actions/sec",
                result.total_actions, result.elapsed_s, result.throughput
            );
        }
    }

    let wall_time = start.elapsed().as_secs_f64();
    let avg_throughput = total_actions as f64 / total_elapsed;

    let output = BenchmarkOutput {
        total_actions,
        elapsed_s: total_elapsed,
        throughput: avg_throughput,
        runs: args.runs,
    };

    if args.json {
        println!("{}", serde_json::to_string(&output)?);
    } else {
        println!();
        println!("{}", separator);
        println!("Results (averaged over {} runs):", args.runs);
        println!("{}", separator);
        println!("  Total actions: {}", output.total_actions);
        println!("  Total workflow time: {:.3}s", output.elapsed_s);
        println!("  Wall time: {:.3}s", wall_time);
        println!("  Throughput: {:.1} actions/sec", output.throughput);
        println!();

        // Calculate expected actions
        // For count=N: N hashes + N analyses + N (special or normal) + 1 combine = 3N+1
        let expected = 3 * args.count + 1;
        println!(
            "  Expected actions per run: {} (3*{} + 1)",
            expected, args.count
        );
        println!(
            "  Actual per run: {}",
            total_actions / args.runs as u64
        );
    }

    Ok(())
}
