#!/usr/bin/env python3
"""
Test durable recovery by simulating a crash mid-workflow.

This demonstrates that:
1. Actions are not re-executed after a "crash"
2. Workflow resumes from where it left off
3. Final result is correct
"""

import asyncio
import copy
import sys

sys.path.insert(0, "src")

from rappel.runner import LocalRunner
from rappel.durable import WorkflowInstance

from benchmark_workflow import SimpleFanOutWorkflow


async def main():
    print("=" * 60)
    print("Durable Recovery Test")
    print("=" * 60)

    # Run workflow partially, then simulate crash
    runner1 = LocalRunner()
    workflow = SimpleFanOutWorkflow()

    count = 16
    iterations = 100

    print(f"\n1. Starting workflow with count={count}, iterations={iterations}")

    # Run workflow to completion first to get expected result
    expected_result = await runner1.run_workflow(
        "recovery-test",
        lambda: workflow.run(count=count, iterations=iterations),
    )
    print(f"   Full execution completed: {len(runner1.instances['recovery-test'].action_queue)} actions")
    print(f"   Result: {expected_result['final_hash'][:32]}...")

    # Now simulate a crash by running partially
    print("\n2. Simulating crash after 8 actions...")

    # Create a new runner (simulating restart)
    runner2 = LocalRunner()

    # Copy instance state but truncate to 8 actions (simulating crash)
    original_instance = runner1.instances["recovery-test"]
    recovered_instance = WorkflowInstance(
        id="recovery-test",
        action_queue=copy.deepcopy(original_instance.action_queue[:8]),
    )
    runner2.instances["recovery-test"] = recovered_instance

    print(f"   Restored with {len(recovered_instance.action_queue)} completed actions")

    # Track which actions get executed
    executed_actions = []
    original_execute = runner2.execute_action

    async def tracking_execute(action_call):
        executed_actions.append(action_call.func_name)
        return await original_execute(action_call)

    runner2.execute_action = tracking_execute

    # Resume workflow
    print("\n3. Resuming workflow from saved state...")
    recovered_result = await runner2.run_workflow(
        "recovery-test",
        lambda: workflow.run(count=count, iterations=iterations),
    )

    print(f"   Execution completed!")
    print(f"   Actions executed after recovery: {len(executed_actions)}")
    print(f"   Total actions in queue: {len(runner2.instances['recovery-test'].action_queue)}")

    # Verify results match
    print("\n4. Verifying results...")
    if recovered_result == expected_result:
        print("   ✓ Results match! Recovery successful.")
    else:
        print("   ✗ Results DO NOT match!")
        print(f"   Expected: {expected_result}")
        print(f"   Got: {recovered_result}")
        return 1

    # Verify we didn't re-execute already completed actions
    expected_new_actions = count - 8  # We had 8 done, need count total
    if len(executed_actions) == expected_new_actions:
        print(f"   ✓ Correct number of actions executed: {len(executed_actions)}")
    else:
        print(f"   ✗ Wrong number of actions: expected {expected_new_actions}, got {len(executed_actions)}")
        return 1

    print("\n" + "=" * 60)
    print("All recovery tests passed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
