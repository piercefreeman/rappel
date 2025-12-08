#!/usr/bin/env python3
"""
Benchmark CLI for measuring durable workflow execution throughput.

Usage:
    uv run python benchmarks/run_benchmark.py --count 16 --iterations 100
    uv run python benchmarks/run_benchmark.py --count 64 --iterations 50 --parallel
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, "src")

from rappel.runner import LocalRunner, ParallelLocalRunner

# Import benchmark workflow to register actions
from benchmark_workflow import (
    BenchmarkFanOutWorkflow,
    SimpleFanOutWorkflow,
)


@dataclass
class BenchmarkResult:
    """Benchmark result metrics."""

    workflow_name: str
    count: int
    iterations: int
    total_actions: int
    elapsed_s: float
    throughput: float  # actions/second
    parallel: bool

    def to_dict(self) -> dict:
        return {
            "workflow_name": self.workflow_name,
            "count": self.count,
            "iterations": self.iterations,
            "total_actions": self.total_actions,
            "elapsed_s": round(self.elapsed_s, 3),
            "throughput": round(self.throughput, 2),
            "parallel": self.parallel,
        }


async def run_benchmark(
    workflow_name: str,
    count: int,
    iterations: int,
    parallel: bool,
    max_concurrent: int,
) -> BenchmarkResult:
    """Run a single benchmark."""
    if parallel:
        runner = ParallelLocalRunner(max_concurrent=max_concurrent)
    else:
        runner = LocalRunner()

    if workflow_name == "fanout":
        workflow = BenchmarkFanOutWorkflow()
        workflow_coro = lambda: workflow.run(indices=list(range(count)), iterations=iterations)
        # Fan-out workflow does:
        # - count hash computations (parallel)
        # - count analyze_hash calls (sequential)
        # - count conditional process calls (sequential)
        # - 1 combine_results call
        expected_actions = count + count + count + 1
    elif workflow_name == "simple":
        workflow = SimpleFanOutWorkflow()
        workflow_coro = lambda: workflow.run(count=count, iterations=iterations)
        # Simple workflow does:
        # - count hash computations (parallel)
        expected_actions = count
    else:
        raise ValueError(f"Unknown workflow: {workflow_name}")

    start = time.perf_counter()
    result = await runner.run_workflow(f"benchmark-{workflow_name}", workflow_coro)
    elapsed = time.perf_counter() - start

    # Count actual actions executed
    instance = runner.instances[f"benchmark-{workflow_name}"]
    actual_actions = len(instance.action_queue)

    throughput = actual_actions / elapsed if elapsed > 0 else 0

    return BenchmarkResult(
        workflow_name=workflow_name,
        count=count,
        iterations=iterations,
        total_actions=actual_actions,
        elapsed_s=elapsed,
        throughput=throughput,
        parallel=parallel,
    )


async def main():
    parser = argparse.ArgumentParser(description="Run durable workflow benchmarks")
    parser.add_argument(
        "--workflow",
        choices=["fanout", "simple"],
        default="simple",
        help="Workflow to benchmark (default: simple)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=16,
        help="Number of parallel hash computations (default: 16)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Hash iterations per action (default: 100)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Execute actions in parallel (default: sequential)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=32,
        help="Max concurrent actions when --parallel (default: 32)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of benchmark runs (default: 1)",
    )

    args = parser.parse_args()

    results = []
    for i in range(args.runs):
        result = await run_benchmark(
            workflow_name=args.workflow,
            count=args.count,
            iterations=args.iterations,
            parallel=args.parallel,
            max_concurrent=args.max_concurrent,
        )
        results.append(result)

    if args.json:
        if len(results) == 1:
            print(json.dumps(results[0].to_dict()))
        else:
            print(json.dumps([r.to_dict() for r in results]))
    else:
        print("\n" + "=" * 60)
        print("Benchmark Results")
        print("=" * 60)

        for i, result in enumerate(results):
            if len(results) > 1:
                print(f"\nRun {i + 1}:")
            print(f"  Workflow: {result.workflow_name}")
            print(f"  Count: {result.count}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Parallel: {result.parallel}")
            print(f"  Total actions: {result.total_actions}")
            print(f"  Elapsed: {result.elapsed_s:.3f}s")
            print(f"  Throughput: {result.throughput:.2f} actions/sec")

        if len(results) > 1:
            avg_throughput = sum(r.throughput for r in results) / len(results)
            print(f"\nAverage throughput: {avg_throughput:.2f} actions/sec")


if __name__ == "__main__":
    asyncio.run(main())
