"""
Instance Worker - Runs workflow instances with durable execution.

This worker connects to the InstanceWorkerBridge gRPC service and:
1. Receives InstanceDispatch messages (containing completed actions to replay)
2. Runs the workflow until it blocks on new actions
3. Returns InstanceActions (pending actions) or InstanceComplete

Instance workers are the orchestrators - they run workflows but delegate
action execution to action workers via the server.
"""

import argparse
import asyncio
import importlib
import sys
from typing import Any, Callable

import grpc

from rappel.durable import ActionResult, ActionStatus, WorkflowInstance, run_until_actions
from rappel.workflow import get_workflow, get_workflow_registry

# These will be generated from proto
# from proto import messages_pb2 as pb
# from proto import messages_pb2_grpc as pb_grpc


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Rappel Instance Worker")
    parser.add_argument(
        "--bridge",
        required=True,
        help="gRPC address of the InstanceWorkerBridge (e.g., localhost:24119)",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        required=True,
        help="Unique worker ID assigned by the server",
    )
    parser.add_argument(
        "--user-module",
        action="append",
        default=[],
        help="Python modules to import (contain @workflow and @action definitions)",
    )
    return parser.parse_args(argv)


async def run_instance(
    instance_id: str,
    workflow_name: str,
    initial_args: dict[str, Any],
    completed_actions: list[ActionResult],
) -> tuple[list[Any], Any | None]:
    """
    Run a workflow instance until it blocks on actions or completes.

    Args:
        instance_id: Unique identifier for this instance
        workflow_name: Name of the workflow class to run
        initial_args: Initial arguments for the workflow
        completed_actions: Previously completed actions to replay

    Returns:
        Tuple of (pending_actions, result)
        - If pending_actions is non-empty, workflow blocked on those actions
        - If pending_actions is empty and result is not None, workflow completed
    """
    workflow_cls = get_workflow(workflow_name)
    if workflow_cls is None:
        raise ValueError(f"Unknown workflow: {workflow_name}")

    # Create workflow instance state
    instance = WorkflowInstance(
        id=instance_id,
        action_queue=completed_actions,
    )

    # Instantiate workflow and create the coroutine
    workflow_obj = workflow_cls()

    async def workflow_coro() -> Any:
        return await workflow_obj.run(**initial_args)

    # Run until blocked on actions
    pending = await run_until_actions(instance, workflow_coro())

    if not pending:
        # Workflow completed - we need to re-run to get the result
        # This time replay everything
        instance.reset_replay()
        try:
            result = await asyncio.wait_for(workflow_coro(), timeout=0.1)
            return [], result
        except asyncio.TimeoutError:
            # This shouldn't happen if workflow truly completed
            return [], None
    else:
        return pending, None


async def run_worker(bridge_addr: str, worker_id: int) -> None:
    """
    Run the instance worker main loop.

    Connects to the gRPC bridge and processes instance dispatches.
    """
    print(f"[InstanceWorker {worker_id}] Connecting to {bridge_addr}")

    # TODO: Replace with actual gRPC implementation once proto is compiled
    # For now, this is a placeholder showing the intended flow

    # channel = grpc.aio.insecure_channel(bridge_addr)
    # stub = pb_grpc.InstanceWorkerBridgeStub(channel)

    # async def outgoing_stream():
    #     # Send WorkerHello first
    #     hello = pb.WorkerHello(worker_id=worker_id, worker_type=pb.WORKER_TYPE_INSTANCE)
    #     yield pb.Envelope(
    #         kind=pb.MESSAGE_KIND_WORKER_HELLO,
    #         payload=hello.SerializeToString(),
    #     )
    #     while True:
    #         envelope = await outgoing.get()
    #         yield envelope

    # async for envelope in stub.Attach(outgoing_stream()):
    #     if envelope.kind == pb.MESSAGE_KIND_INSTANCE_DISPATCH:
    #         dispatch = pb.InstanceDispatch()
    #         dispatch.ParseFromString(envelope.payload)
    #
    #         # Convert completed actions from proto
    #         completed = [
    #             ActionResult(
    #                 action_id=ar.action_id,
    #                 status=ActionStatus.COMPLETED if ar.success else ActionStatus.FAILED,
    #                 result=deserialize_result(ar.payload) if ar.success else None,
    #                 error=ar.error_message if not ar.success else None,
    #             )
    #             for ar in dispatch.completed_actions
    #         ]
    #
    #         # Run instance
    #         pending, result = await run_instance(
    #             dispatch.instance_id,
    #             dispatch.workflow_name,
    #             deserialize_kwargs(dispatch.initial_args),
    #             completed,
    #         )
    #
    #         if pending:
    #             # Report pending actions
    #             msg = pb.InstanceActions(
    #                 instance_id=dispatch.instance_id,
    #                 replayed_count=len(completed),
    #                 pending_actions=[...],  # Convert to proto
    #             )
    #             await outgoing.put(pb.Envelope(
    #                 kind=pb.MESSAGE_KIND_INSTANCE_ACTIONS,
    #                 payload=msg.SerializeToString(),
    #             ))
    #         else:
    #             # Workflow completed
    #             msg = pb.InstanceComplete(
    #                 instance_id=dispatch.instance_id,
    #                 result=serialize_result(result),
    #                 total_actions=len(completed),
    #             )
    #             await outgoing.put(pb.Envelope(
    #                 kind=pb.MESSAGE_KIND_INSTANCE_COMPLETE,
    #                 payload=msg.SerializeToString(),
    #             ))

    print(f"[InstanceWorker {worker_id}] Worker loop placeholder - gRPC not yet implemented")
    # Keep alive for testing
    await asyncio.sleep(3600)


def main() -> None:
    """Entry point for rappel-instance-worker command."""
    args = _parse_args(sys.argv[1:])

    # Import user modules to register workflows and actions
    for module_name in args.user_module:
        print(f"[InstanceWorker] Importing module: {module_name}")
        importlib.import_module(module_name)

    print(f"[InstanceWorker] Registered workflows: {list(get_workflow_registry().keys())}")

    # Run the worker
    asyncio.run(run_worker(args.bridge, args.worker_id))


if __name__ == "__main__":
    main()
