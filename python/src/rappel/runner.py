"""
Local durable runner for executing workflows.

This runner executes workflows locally without a distributed server,
useful for testing and benchmarking the durable execution pattern.
"""

import asyncio
from typing import Any, Callable

from rappel.actions import get_action
from rappel.durable import ActionCall, ActionResult, ActionStatus, WorkflowInstance, run_until_actions


class LocalRunner:
    """
    Local workflow runner that executes actions in-process.

    This is useful for testing and benchmarking without the full
    distributed server setup.
    """

    def __init__(self) -> None:
        self.instances: dict[str, WorkflowInstance] = {}

    async def execute_action(self, action_call: ActionCall) -> ActionResult:
        """Execute a single action and return the result."""
        action_func = get_action(action_call.func_name)

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

    async def run_workflow(
        self,
        workflow_id: str,
        workflow_coro_fn: Callable[[], Any],
    ) -> Any:
        """
        Run a workflow to completion, executing actions as needed.

        Args:
            workflow_id: Unique identifier for this workflow instance
            workflow_coro_fn: Function that returns the workflow coroutine

        Returns:
            The workflow's return value once complete
        """
        # Get or create instance
        if workflow_id not in self.instances:
            self.instances[workflow_id] = WorkflowInstance(id=workflow_id)

        instance = self.instances[workflow_id]
        iteration = 0

        while True:
            iteration += 1
            # Run workflow until it blocks on actions
            pending = await run_until_actions(instance, workflow_coro_fn())

            if not pending:
                # Workflow completed! Run one more time to get the result
                instance.reset_replay()

                # Set up context for final run
                from rappel.durable import ExecutionContext, _current_context

                ctx = ExecutionContext(instance=instance, capture_mode=True)
                token = _current_context.set(ctx)

                try:
                    result = await workflow_coro_fn()
                    return result
                finally:
                    _current_context.reset(token)

            # Execute all pending actions (could be parallelized)
            for action_call in pending:
                result = await self.execute_action(action_call)
                instance.action_queue.append(result)

            # Loop will re-run workflow, replaying completed actions


class ParallelLocalRunner:
    """
    Local runner that executes actions in parallel.

    This better simulates the distributed execution model where
    multiple actions can run concurrently.
    """

    def __init__(self, max_concurrent: int = 32) -> None:
        self.instances: dict[str, WorkflowInstance] = {}
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_action(self, action_call: ActionCall) -> ActionResult:
        """Execute a single action with concurrency limiting."""
        async with self.semaphore:
            action_func = get_action(action_call.func_name)

            if action_func is None:
                return ActionResult(
                    action_id=action_call.id,
                    status=ActionStatus.FAILED,
                    error=f"Unknown action: {action_call.func_name}",
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

    async def run_workflow(
        self,
        workflow_id: str,
        workflow_coro_fn: Callable[[], Any],
    ) -> Any:
        """Run a workflow with parallel action execution."""
        if workflow_id not in self.instances:
            self.instances[workflow_id] = WorkflowInstance(id=workflow_id)

        instance = self.instances[workflow_id]

        while True:
            pending = await run_until_actions(instance, workflow_coro_fn())

            if not pending:
                # Get final result
                instance.reset_replay()
                from rappel.durable import ExecutionContext, _current_context

                ctx = ExecutionContext(instance=instance, capture_mode=True)
                token = _current_context.set(ctx)

                try:
                    return await workflow_coro_fn()
                finally:
                    _current_context.reset(token)

            # Execute all pending actions in parallel
            results = await asyncio.gather(
                *[self.execute_action(action_call) for action_call in pending]
            )

            # Add results to queue in order
            for result in results:
                instance.action_queue.append(result)
