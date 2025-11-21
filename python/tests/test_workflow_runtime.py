import asyncio
from typing import List

from proto import messages_pb2 as pb2
from rappel import registry as action_registry
from rappel.actions import action, serialize_error_payload, serialize_result_payload
from rappel.workflow_runtime import (
    _LOOP_INDEX_VAR,
    NodeExecutionResult,
    WorkflowNodeResult,
    execute_node,
)

_guard_calls: List[str] = []


@action
async def guarded_noop() -> str:
    _guard_calls.append("ran")
    return "ok"


@action
async def exception_handler() -> str:
    return "handled"


@action
async def multiply(value: int) -> int:
    return value * 2


def _build_dispatch(flag: bool) -> pb2.WorkflowNodeDispatch:
    if action_registry.get("guarded_noop") is None:
        action_registry.register("guarded_noop", guarded_noop)
    node = pb2.WorkflowDagNode(
        id="node_guard",
        action="guarded_noop",
        module=__name__,
        guard="flag",
    )
    node.produces.append("result")
    dispatch = pb2.WorkflowNodeDispatch(node=node)
    workflow_input = pb2.WorkflowArguments()
    entry = workflow_input.arguments.add()
    entry.key = "flag"
    entry.value.primitive.bool_value = flag
    dispatch.workflow_input.CopyFrom(workflow_input)
    payload = serialize_result_payload(flag)
    entry = dispatch.context.add()
    entry.variable = "flag"
    entry.payload.CopyFrom(payload)
    return dispatch


def _build_exception_dispatch(include_error: bool) -> pb2.WorkflowNodeDispatch:
    if action_registry.get("exception_handler") is None:
        action_registry.register("exception_handler", exception_handler)
    node = pb2.WorkflowDagNode(
        id="node_handler",
        action="exception_handler",
        module=__name__,
    )
    node.produces.append("value")
    node.guard = "__workflow_exceptions.get('node_source') is not None"
    edge = node.exception_edges.add()
    edge.source_node_id = "node_source"
    edge.exception_type = "RuntimeError"
    dispatch = pb2.WorkflowNodeDispatch(node=node)
    dispatch.workflow_input.CopyFrom(pb2.WorkflowArguments())
    if include_error:
        payload = serialize_error_payload("source", RuntimeError("boom"))
    else:
        payload = serialize_result_payload("noop")
    entry = dispatch.context.add()
    entry.variable = ""
    entry.workflow_node_id = "node_source"
    entry.payload.CopyFrom(payload)
    return dispatch


def test_execute_node_skips_guarded_action() -> None:
    _guard_calls.clear()
    payload = _build_dispatch(flag=False)
    result = asyncio.run(execute_node(payload))
    assert isinstance(result, NodeExecutionResult)
    assert isinstance(result.result, WorkflowNodeResult)
    assert result.result.variables == {}
    assert _guard_calls == []


def test_execute_node_runs_guarded_action_when_true() -> None:
    _guard_calls.clear()
    payload = _build_dispatch(flag=True)
    result = asyncio.run(execute_node(payload))
    assert isinstance(result, NodeExecutionResult)
    assert isinstance(result.result, WorkflowNodeResult)
    assert result.result.variables == {"result": "ok"}
    assert _guard_calls == ["ran"]


def test_execute_node_handles_exception_when_edge_matches() -> None:
    payload = _build_exception_dispatch(include_error=True)
    result = asyncio.run(execute_node(payload))
    assert isinstance(result, NodeExecutionResult)
    assert isinstance(result.result, WorkflowNodeResult)
    assert result.result.variables == {"value": "handled"}


def test_execute_node_skips_exception_handler_without_error() -> None:
    payload = _build_exception_dispatch(include_error=False)
    result = asyncio.run(execute_node(payload))
    assert isinstance(result, NodeExecutionResult)
    assert isinstance(result.result, WorkflowNodeResult)
    assert result.result.variables == {}


def _build_loop_controller_dispatch(items: list[int]) -> pb2.WorkflowNodeDispatch:
    node = pb2.WorkflowDagNode(
        id="loop_node",
        action="loop",
        module=__name__,
    )
    node.produces.append("results")
    node.loop.iterable_expr = "items"
    node.loop.loop_var = "item"
    node.loop.body_action = "multiply"
    node.loop.body_module = __name__
    node.loop.body_kwargs["value"] = "item"
    node.loop.accumulator = "results"
    workflow_input = pb2.WorkflowArguments()
    entry = workflow_input.arguments.add()
    entry.key = "items"
    for value in items:
        item = entry.value.list_value.items.add()
        item.primitive.int_value = value
    dispatch = pb2.WorkflowNodeDispatch(node=node)
    dispatch.workflow_input.CopyFrom(workflow_input)
    return dispatch


def _dispatch_with_loop_state(
    node: pb2.WorkflowDagNode,
    accumulator_payload: pb2.WorkflowArguments,
    index_payload: pb2.WorkflowArguments,
    workflow_input: pb2.WorkflowArguments,
) -> pb2.WorkflowNodeDispatch:
    dispatch = pb2.WorkflowNodeDispatch(node=node)
    for variable, payload in (("results", accumulator_payload), (_LOOP_INDEX_VAR, index_payload)):
        entry = dispatch.context.add()
        entry.variable = variable
        entry.workflow_node_id = node.id
        entry.payload.CopyFrom(payload)
    dispatch.workflow_input.CopyFrom(workflow_input)
    return dispatch


def _payload_from_value(value: pb2.WorkflowArgumentValue) -> pb2.WorkflowArguments:
    arguments = pb2.WorkflowArguments()
    entry = arguments.arguments.add()
    entry.key = "result"
    entry.value.CopyFrom(value)
    return arguments


def test_loop_controller_accumulates_results_and_signals_continuation() -> None:
    if action_registry.get("multiply") is None:
        action_registry.register("multiply", multiply)
    dispatch = _build_loop_controller_dispatch([1, 2, 3])
    first = asyncio.run(execute_node(dispatch))
    assert isinstance(first, NodeExecutionResult)
    assert first.control is not None
    assert first.control.loop.has_next is True
    assert first.control.loop.next_index == 1
    assert isinstance(first.result, WorkflowNodeResult)
    assert first.result.variables.get("results") == [2]
    loop_payload = _payload_from_value(first.control.loop.accumulator_value)
    index_value = pb2.WorkflowArgumentValue()
    index_value.primitive.int_value = first.control.loop.next_index
    index_payload = _payload_from_value(index_value)
    second = asyncio.run(
        execute_node(
            _dispatch_with_loop_state(
                dispatch.node, loop_payload, index_payload, dispatch.workflow_input
            )
        )
    )
    assert isinstance(second, NodeExecutionResult)
    assert second.control is not None
    assert second.control.loop.has_next is True
    assert second.control.loop.next_index == 2
    assert isinstance(second.result, WorkflowNodeResult)
    assert second.result.variables.get("results") == [2, 4]
    second_payload = _payload_from_value(second.control.loop.accumulator_value)
    index_value = pb2.WorkflowArgumentValue()
    index_value.primitive.int_value = second.control.loop.next_index
    index_payload = _payload_from_value(index_value)
    final = asyncio.run(
        execute_node(
            _dispatch_with_loop_state(
                dispatch.node, second_payload, index_payload, dispatch.workflow_input
            )
        )
    )
    assert isinstance(final, NodeExecutionResult)
    assert final.control is not None
    assert final.control.loop.has_next is False
    assert isinstance(final.result, WorkflowNodeResult)
    assert final.result.variables.get("results") == [2, 4, 6]
