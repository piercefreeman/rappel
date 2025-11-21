"""Runtime helpers for executing workflow DAG nodes inside the worker."""

import asyncio
import importlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from pydantic import BaseModel

from proto import messages_pb2 as pb2

from .actions import deserialize_result_payload
from .registry import registry
from .serialization import arguments_to_kwargs, dumps

_LOOP_INDEX_VAR = "__loop_index"


class WorkflowNodeResult(BaseModel):
    variables: Dict[str, Any]


@dataclass
class NodeExecutionResult:
    result: Any
    control: pb2.WorkflowNodeControl | None = None


def _decode_workflow_input(payload: pb2.WorkflowArguments | None) -> dict[str, Any]:
    return arguments_to_kwargs(payload)


def _build_context(
    dispatch: pb2.WorkflowNodeDispatch,
) -> Tuple[dict[str, Any], dict[str, Dict[str, Any]]]:
    context: dict[str, Any] = {}
    exceptions: dict[str, Dict[str, Any]] = {}
    inputs = _decode_workflow_input(dispatch.workflow_input)
    context.update(inputs)
    for entry in dispatch.context:
        variable = entry.variable
        decoded = deserialize_result_payload(entry.payload)
        if decoded.error is not None:
            source_id = getattr(entry, "workflow_node_id", "")
            if source_id:
                error_data = dict(decoded.error)
                exceptions[source_id] = error_data
            continue
        result = decoded.result
        if not variable:
            continue
        if isinstance(result, WorkflowNodeResult):
            if variable in result.variables:
                context[variable] = result.variables[variable]
        else:
            context[variable] = result
    context["__workflow_exceptions"] = exceptions
    return context, exceptions


def _ensure_action_module(node: pb2.WorkflowDagNode) -> None:
    module_name = getattr(node, "module", "")
    if module_name:
        importlib.import_module(module_name)


def _evaluate_kwargs(node: pb2.WorkflowDagNode, context: dict[str, Any]) -> Dict[str, Any]:
    namespace = {**context}
    evaluated: Dict[str, Any] = {}
    for key, expr in node.kwargs.items():
        evaluated[key] = eval(expr, {}, namespace)  # noqa: S307 - controlled input
    return evaluated


def _evaluate_loop_kwargs(
    raw_kwargs: Mapping[str, str], namespace: dict[str, Any]
) -> Dict[str, Any]:
    evaluated: Dict[str, Any] = {}
    for key, expr in raw_kwargs.items():
        evaluated[key] = eval(expr, {}, namespace)  # noqa: S307 - controlled input
    return evaluated


def _guard_allows_execution(node: pb2.WorkflowDagNode, context: dict[str, Any]) -> bool:
    guard_expr = getattr(node, "guard", "")
    if not guard_expr:
        return True
    namespace = {**context}
    return bool(eval(guard_expr, {}, namespace))  # noqa: S307 - controlled input


def _matching_exception_sources(
    node: pb2.WorkflowDagNode, exceptions: dict[str, Dict[str, Any]]
) -> list[str]:
    matches: list[str] = []
    for edge in getattr(node, "exception_edges", []):
        source_id = edge.source_node_id
        if not source_id:
            continue
        error = exceptions.get(source_id)
        if error is None:
            continue
        type_name = edge.exception_type or ""
        module_name = edge.exception_module or ""
        if type_name and error.get("type") != type_name:
            continue
        if module_name and error.get("module") != module_name:
            continue
        matches.append(source_id)
    return matches


def _validate_exception_context(
    node: pb2.WorkflowDagNode,
    exceptions: dict[str, Dict[str, Any]],
    matched_sources: list[str],
) -> None:
    if not exceptions:
        return
    allowed = set(matched_sources)
    unmatched = [source for source in exceptions if source not in allowed]
    if unmatched:
        source = unmatched[0]
        raise RuntimeError(f"dependency {source} failed")


def _import_support_blocks(node: pb2.WorkflowDagNode, namespace: dict[str, Any]) -> None:
    imports = node.kwargs.get("imports")
    definitions = node.kwargs.get("definitions")
    if imports:
        if isinstance(imports, str):
            blocks = [imports]
        else:
            blocks = list(imports)
        for block in blocks:
            exec(block, namespace)  # noqa: S102 - controlled by workflow author
    if definitions:
        if isinstance(definitions, str):
            defs = [definitions]
        else:
            defs = list(definitions)
        for definition in defs:
            exec(definition, namespace)  # noqa: S102 - controlled by workflow author


def _execute_python_block(node: pb2.WorkflowDagNode, context: dict[str, Any]) -> dict[str, Any]:
    namespace = dict(context)
    _import_support_blocks(node, namespace)
    code = node.kwargs.get("code")
    if not code:
        return {}
    exec(code, namespace)  # noqa: S102 - controlled by workflow author
    result: dict[str, Any] = {}
    for name in node.produces:
        if name in namespace:
            result[name] = namespace[name]
    return result


async def _execute_loop_controller(
    node: pb2.WorkflowDagNode, context: dict[str, Any]
) -> NodeExecutionResult:
    if not node.HasField("loop"):
        raise RuntimeError("loop node missing specification")
    loop_spec = node.loop
    iterable_expr = loop_spec.iterable_expr
    loop_var = loop_spec.loop_var
    accumulator_name = loop_spec.accumulator
    body_action = loop_spec.body_action
    body_kwargs_map = dict(loop_spec.body_kwargs)
    preamble = loop_spec.preamble or ""
    body_module = loop_spec.body_module or ""
    if not iterable_expr or not loop_var or not accumulator_name or not body_action:
        raise RuntimeError("loop node missing required metadata")

    namespace = dict(context)
    items = list(eval(iterable_expr, {}, namespace))  # noqa: S307 - controlled input
    raw_accumulator = context.get(accumulator_name, [])
    accumulator: list[Any] = []
    if isinstance(raw_accumulator, list):
        accumulator = list(raw_accumulator)
    elif raw_accumulator:
        try:
            accumulator = list(raw_accumulator)
        except TypeError:
            accumulator = [raw_accumulator]
    raw_index = context.get(_LOOP_INDEX_VAR, 0)
    try:
        loop_index = int(raw_index) if raw_index is not None else 0
    except (TypeError, ValueError):
        loop_index = 0
    if loop_index >= len(items):
        return NodeExecutionResult(
            result=WorkflowNodeResult(
                variables={
                    accumulator_name: accumulator,
                    _LOOP_INDEX_VAR: loop_index,
                }
            )
        )

    namespace[loop_var] = items[loop_index]
    if preamble:
        exec(preamble, namespace)  # noqa: S102 - controlled by workflow author
    evaluated_kwargs = _evaluate_loop_kwargs(body_kwargs_map, namespace)
    if body_module:
        importlib.import_module(body_module)
    handler = registry.get(body_action)
    if handler is None:
        raise RuntimeError(f"action '{body_action}' not registered")
    value = handler(**evaluated_kwargs)
    if asyncio.iscoroutine(value):
        value = await value
    accumulator.append(value)
    next_index = loop_index + 1
    continue_loop = next_index < len(items)
    control = pb2.WorkflowNodeControl(
        loop=pb2.WorkflowLoopControl(
            node_id=node.id,
            next_index=next_index,
            has_next=continue_loop,
            accumulator=accumulator_name,
            accumulator_value=dumps(accumulator),
        )
    )
    return NodeExecutionResult(
        result=WorkflowNodeResult(
            variables={
                accumulator_name: accumulator,
                _LOOP_INDEX_VAR: next_index,
            }
        ),
        control=control,
    )


async def execute_node(dispatch: pb2.WorkflowNodeDispatch) -> NodeExecutionResult:
    context, exceptions = _build_context(dispatch)
    node = dispatch.node
    if node is None:
        raise RuntimeError("workflow dispatch missing node definition")
    if not _guard_allows_execution(node, context):
        return NodeExecutionResult(result=WorkflowNodeResult(variables={}))
    matched_sources = _matching_exception_sources(node, exceptions)
    _validate_exception_context(node, exceptions, matched_sources)
    if node.action == "python_block":
        result_map = _execute_python_block(node, context)
        return NodeExecutionResult(result=WorkflowNodeResult(variables=result_map))
    elif node.action == "loop":
        return await _execute_loop_controller(node, context)
    else:
        _ensure_action_module(node)
        handler = registry.get(node.action)
        if handler is None:
            raise RuntimeError(f"action '{node.action}' not registered")
        kwargs = _evaluate_kwargs(node, context)
        value = handler(**kwargs)
        if asyncio.iscoroutine(value):
            value = await value
        if node.produces:
            result_map = {node.produces[0]: value}
        else:
            result_map = {}
    return NodeExecutionResult(result=WorkflowNodeResult(variables=result_map))
