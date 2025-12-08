"""
Benchmark test for the new durable execution engine.

This tests the core durable execution model without the full Rust<->Python infrastructure.
It simulates what the Rust server would do by executing actions immediately.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

from rappel.durable import (
    ActionCall,
    ActionResult,
    ActionStatus,
    WorkflowInstance,
    run_until_actions,
)
from rappel.actions import get_action_registry

# Import to register actions
from benchmark_workflow import BenchmarkWorkflow


@dataclass
class BenchmarkResult:
    total_actions: int
    elapsed_s: float
    throughput: float  # actions/sec
    workflow_result: Any


async def execute_action(action_call: ActionCall) -> ActionResult:
    """Execute an action and return the result."""
    registry = get_action_registry()
    action_func = registry.get(action_call.func_name)

    if action_func is None:
        return ActionResult(
            action_id=action_call.id,
            status=ActionStatus.FAILED,
            error=f"Unknown action: {action_call.func_name}",
        )

    try:
        # Get the original function (not the wrapper)
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


async def run_workflow_to_completion(
    workflow_coro_fn: Callable[[], Any],
) -> tuple[int, Any]:
    """
    Run a workflow to completion, executing actions as needed.

    Returns (total_actions, final_result).
    """
    instance = WorkflowInstance(id="benchmark-instance")
    total_actions = 0

    while True:
        # Run workflow until it blocks on actions
        pending = await run_until_actions(instance, workflow_coro_fn())

        if not pending:
            # Workflow completed - get the result
            # Re-run one more time to capture the return value
            instance.reset_replay()

            # Set up context for final run
            from rappel.durable import ExecutionContext, _current_context
            ctx = ExecutionContext(instance=instance, capture_mode=True)
            token = _current_context.set(ctx)

            try:
                # This should complete without blocking since all actions are replayed
                result = await asyncio.wait_for(workflow_coro_fn(), timeout=1.0)
                return total_actions, result
            except asyncio.TimeoutError:
                # Shouldn't happen if workflow truly completed
                return total_actions, None
            finally:
                _current_context.reset(token)

        # Execute all pending actions
        for action_call in pending:
            action_result = await execute_action(action_call)
            instance.action_queue.append(action_result)
            total_actions += 1


async def run_benchmark(count: int, iterations: int) -> BenchmarkResult:
    """Run the benchmark workflow and measure performance."""
    workflow = BenchmarkWorkflow()

    start = time.perf_counter()

    total_actions, result = await run_workflow_to_completion(
        lambda: workflow.run(count=count, iterations=iterations)
    )

    elapsed = time.perf_counter() - start
    throughput = total_actions / elapsed if elapsed > 0 else 0

    return BenchmarkResult(
        total_actions=total_actions,
        elapsed_s=elapsed,
        throughput=throughput,
        workflow_result=result,
    )


async def main():
    print("=" * 70)
    print("Durable Execution Benchmark")
    print("=" * 70)

    # Warm up
    print("\nWarming up...")
    await run_benchmark(count=4, iterations=10)

    # Run benchmarks with different parameters
    configs = [
        (16, 100),   # 16 parallel hashes, 100 iterations each
        (32, 100),   # 32 parallel hashes
        (64, 100),   # 64 parallel hashes
        (128, 100),  # 128 parallel hashes
    ]

    for count, iterations in configs:
        result = await run_benchmark(count, iterations)

        print(f"\n--- count={count}, iterations={iterations} ---")
        print(f"Total actions: {result.total_actions}")
        print(f"Elapsed time: {result.elapsed_s:.3f}s")
        print(f"Throughput: {result.throughput:.1f} actions/sec")

        if result.workflow_result:
            print(f"Result: {result.workflow_result['total_processed']} processed")
            print(f"  Special: {result.workflow_result['special_count']}")
            print(f"  Normal: {result.workflow_result['normal_count']}")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
