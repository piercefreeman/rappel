#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["click"]
# ///
"""Run benchmarks and output results as JSON."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import click


def reset_database():
    """Reset the database tables for clean benchmark runs."""
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://mountaineer:mountaineer@localhost:5432/mountaineer_daemons"
    )

    # Parse connection string
    import re

    match = re.match(r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", db_url)
    if not match:
        print(f"Warning: Could not parse DATABASE_URL: {db_url}", file=sys.stderr)
        return

    user, password, host, port, dbname = match.groups()

    env = os.environ.copy()
    env["PGPASSWORD"] = password

    tables = [
        "daemon_action_ledger",
        "workflow_instances",
        "node_ready_state",
        "node_pending_context",
        "instance_eval_context",
        "loop_iteration_state",
        "workflow_versions",
    ]

    cmd = [
        "psql",
        "-h",
        host,
        "-p",
        port,
        "-U",
        user,
        "-d",
        dbname,
        "-c",
        f"TRUNCATE {', '.join(tables)} CASCADE;",
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: Database reset failed: {result.stderr}", file=sys.stderr)


def check_benchmark_available() -> bool:
    """Check if the benchmark binary exists and has the expected subcommands."""
    binary_path = Path("./target/release/benchmark")
    if not binary_path.exists():
        return False

    # Check if it has the expected subcommands
    try:
        result = subprocess.run(
            ["./target/release/benchmark", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "actions" in result.stdout and "instances" in result.stdout
    except Exception:
        return False


def run_benchmark(benchmark_type: str, args: list[str], timeout: int = 300) -> dict:
    """Run a benchmark and parse the results."""
    cmd = ["./target/release/benchmark", benchmark_type] + args

    print(f"Running: {' '.join(cmd)}", file=sys.stderr)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        print("Benchmark binary not found", file=sys.stderr)
        return {"error": "binary_not_found"}
    except subprocess.TimeoutExpired:
        print(f"Benchmark {benchmark_type} timed out after {timeout}s", file=sys.stderr)
        return {"error": "timeout"}

    output = result.stdout + result.stderr

    # Parse the summary line
    metrics = {}

    # Extract throughput
    throughput_match = re.search(
        r"throughput[=\s\"]+(\d+(?:\.\d+)?)\s*(?:msg/s|actions/s)?", output
    )
    if throughput_match:
        metrics["throughput"] = float(throughput_match.group(1))

    # Extract P95 round-trip
    p95_match = re.search(r"p95_round_trip_ms[=\s\"]+(\d+(?:\.\d+)?)", output)
    if p95_match:
        metrics["p95_ms"] = float(p95_match.group(1))

    # Extract total messages/actions
    total_match = re.search(r"total[=\s\"]+(\d+)", output)
    if total_match:
        metrics["total"] = int(total_match.group(1))

    # Extract elapsed time
    elapsed_match = re.search(r"elapsed[=\s\"]+(\d+(?:\.\d+)?)s", output)
    if elapsed_match:
        metrics["elapsed_s"] = float(elapsed_match.group(1))

    # Extract avg round trip
    avg_rt_match = re.search(r"avg_round_trip_ms[=\s\"]+(\d+(?:\.\d+)?)", output)
    if avg_rt_match:
        metrics["avg_rt_ms"] = float(avg_rt_match.group(1))

    if not metrics:
        print(f"Warning: Could not parse output for {benchmark_type}:", file=sys.stderr)
        print(output[:500], file=sys.stderr)
        metrics["error"] = "parse_failed"
        metrics["raw_output"] = output[:1000]

    return metrics


@click.command()
@click.option("--output", "-o", type=click.Path(), required=True, help="Output JSON file path")
@click.option("--skip-actions", is_flag=True, help="Skip actions benchmark")
@click.option("--skip-instances", is_flag=True, help="Skip instances benchmark")
def main(output: str, skip_actions: bool, skip_instances: bool):
    """Run all benchmarks and output results as JSON."""
    results = {}

    # Check if benchmark binary is available
    if not check_benchmark_available():
        print("Benchmark binary not available or missing subcommands", file=sys.stderr)
        results["_meta"] = {
            "benchmark_available": False,
            "reason": "Benchmark binary not found or missing required subcommands (actions, instances)",
        }
        Path(output).write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        return

    results["_meta"] = {"benchmark_available": True}

    # Actions benchmark - raw action throughput
    if not skip_actions:
        print("=== Running Actions Benchmark ===", file=sys.stderr)
        reset_database()
        results["actions"] = run_benchmark(
            "actions",
            [
                "--messages",
                "10000",
                "--payload",
                "256",
                "--concurrency",
                "64",
                "--workers",
                "4",
                "--log-interval",
                "0",
            ],
        )

    # Instances benchmark - workflow parsing with loops
    if not skip_instances:
        print("=== Running Instances Benchmark ===", file=sys.stderr)
        reset_database()
        results["instances"] = run_benchmark(
            "instances",
            [
                "--instances",
                "30",
                "--batch-size",
                "4",
                "--concurrency",
                "32",
                "--workers",
                "2",
                "--log-interval",
                "0",
            ],
        )

    # Write results
    Path(output).write_text(json.dumps(results, indent=2))
    print(f"Results written to {output}", file=sys.stderr)

    # Also print to stdout for easy viewing
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
