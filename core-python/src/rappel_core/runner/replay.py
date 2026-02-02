"""Replay variable values from a runner state snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
from uuid import UUID

from proto import ast_pb2 as ir

from ..dag import EdgeType, assert_never
from .state import (
    ActionResultValue,
    BinaryOpValue,
    DictValue,
    DotValue,
    FunctionCallValue,
    IndexValue,
    ListValue,
    LiteralValue,
    RunnerState,
    SpreadValue,
    UnaryOpValue,
    ValueExpr,
    VariableValue,
)


class ReplayError(Exception):
    """Raised when replay cannot reconstruct variable values."""


@dataclass(frozen=True)
class ReplayResult:
    variables: dict[str, Any]


@dataclass(frozen=True)
class EvalContext:
    state: RunnerState
    action_results: Mapping[UUID, Any]
    cache: dict[tuple[UUID, str], Any]
    index: Mapping[UUID, int]
    timeline: Sequence[UUID]
    incoming_data: Mapping[UUID, Sequence[UUID]]


def replay_variables(
    state: RunnerState,
    action_results: Mapping[UUID, Any],
) -> ReplayResult:
    variables: dict[str, Any] = {}
    cache: dict[tuple[UUID, str], Any] = {}
    timeline: Sequence[UUID] = state.timeline or list(state.nodes.keys())
    index = {node_id: idx for idx, node_id in enumerate(timeline)}
    incoming_data = _build_incoming_data_map(state, index)
    ctx = EvalContext(
        state=state,
        action_results=action_results,
        cache=cache,
        index=index,
        timeline=timeline,
        incoming_data=incoming_data,
    )

    for node_id in timeline:
        node = state.nodes.get(node_id)
        if node is None or not node.assignments:
            continue
        for target in node.assignments:
            variables[target] = _evaluate_assignment(
                ctx,
                node_id,
                target,
                set(),
            )

    return ReplayResult(variables=variables)


def _evaluate_assignment(
    ctx: EvalContext,
    node_id: UUID,
    target: str,
    stack: set[tuple[UUID, str]],
) -> Any:
    key = (node_id, target)
    if key in ctx.cache:
        return ctx.cache[key]
    if key in stack:
        raise ReplayError(f"recursive assignment detected for {target} in {node_id}")

    node = ctx.state.nodes.get(node_id)
    if node is None or target not in node.assignments:
        raise ReplayError(f"missing assignment for {target} in {node_id}")

    stack.add(key)
    value = _evaluate_value(
        node.assignments[target],
        ctx,
        node_id,
        stack,
    )
    stack.remove(key)
    ctx.cache[key] = value
    return value


def _evaluate_value(
    expr: ValueExpr,
    ctx: EvalContext,
    current_node_id: UUID,
    stack: set[tuple[UUID, str]],
) -> Any:
    if isinstance(expr, LiteralValue):
        return expr.value
    if isinstance(expr, VariableValue):
        return _resolve_variable(
            ctx,
            current_node_id,
            expr.name,
            stack,
        )
    if isinstance(expr, ActionResultValue):
        return _resolve_action_result(expr, ctx)
    if isinstance(expr, BinaryOpValue):
        left = _evaluate_value(
            expr.left, ctx, current_node_id, stack
        )
        right = _evaluate_value(
            expr.right, ctx, current_node_id, stack
        )
        return _apply_binary(expr.op, left, right)
    if isinstance(expr, UnaryOpValue):
        operand = _evaluate_value(
            expr.operand, ctx, current_node_id, stack
        )
        return _apply_unary(expr.op, operand)
    if isinstance(expr, ListValue):
        return [
            _evaluate_value(item, ctx, current_node_id, stack)
            for item in expr.elements
        ]
    if isinstance(expr, DictValue):
        return {
            _evaluate_value(entry.key, ctx, current_node_id, stack): _evaluate_value(
                entry.value, ctx, current_node_id, stack
            )
            for entry in expr.entries
        }
    if isinstance(expr, IndexValue):
        obj = _evaluate_value(expr.object, ctx, current_node_id, stack)
        idx = _evaluate_value(expr.index, ctx, current_node_id, stack)
        return obj[idx]
    if isinstance(expr, DotValue):
        obj = _evaluate_value(expr.object, ctx, current_node_id, stack)
        if isinstance(obj, dict):
            if expr.attribute in obj:
                return obj[expr.attribute]
            raise ReplayError(f"dict has no key '{expr.attribute}'")
        try:
            return object.__getattribute__(obj, expr.attribute)
        except AttributeError as exc:
            raise ReplayError(f"attribute '{expr.attribute}' not found") from exc
    if isinstance(expr, FunctionCallValue):
        return _evaluate_function_call(
            expr,
            ctx,
            current_node_id,
            stack,
        )
    if isinstance(expr, SpreadValue):
        raise ReplayError("cannot replay unresolved spread expression")
    assert_never(expr)


def _resolve_variable(
    ctx: EvalContext,
    current_node_id: UUID,
    name: str,
    stack: set[tuple[UUID, str]],
) -> Any:
    source_node_id = _find_variable_source_node(
        ctx, current_node_id, name
    )
    if source_node_id is None:
        raise ReplayError(f"variable not found via data-flow edges: {name}")
    return _evaluate_assignment(
        ctx,
        source_node_id,
        name,
        stack,
    )


def _find_variable_source_node(
    ctx: EvalContext,
    current_node_id: UUID,
    name: str,
) -> Optional[UUID]:
    sources = ctx.incoming_data.get(current_node_id, ())
    current_idx = ctx.index.get(current_node_id, len(ctx.index))
    for source_id in sources:
        if ctx.index.get(source_id, -1) > current_idx:
            continue
        node = ctx.state.nodes.get(source_id)
        if node is not None and name in node.assignments:
            return source_id
    return None


def _resolve_action_result(expr: ActionResultValue, ctx: EvalContext) -> Any:
    if expr.node_id not in ctx.action_results:
        raise ReplayError(f"missing action result for {expr.node_id}")
    value = ctx.action_results[expr.node_id]
    if expr.result_index is None:
        return value
    try:
        return value[expr.result_index]
    except Exception as exc:  # noqa: BLE001
        raise ReplayError(
            f"action result for {expr.node_id} has no index {expr.result_index}"
        ) from exc


def _evaluate_function_call(
    expr: FunctionCallValue,
    ctx: EvalContext,
    current_node_id: UUID,
    stack: set[tuple[UUID, str]],
) -> Any:
    args = [
        _evaluate_value(arg, ctx, current_node_id, stack)
        for arg in expr.args
    ]
    kwargs = {
        name: _evaluate_value(value, ctx, current_node_id, stack)
        for name, value in expr.kwargs.items()
    }

    if (
        expr.global_function
        and expr.global_function != ir.GlobalFunction.GLOBAL_FUNCTION_UNSPECIFIED
    ):
        return _evaluate_global_function(expr.global_function, args, kwargs)

    raise ReplayError(f"cannot replay non-global function call: {expr.name}")


def _evaluate_global_function(
    global_function: ir.GlobalFunction,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Any:
    match global_function:
        case ir.GlobalFunction.GLOBAL_FUNCTION_RANGE:
            return list(range(*args))
        case ir.GlobalFunction.GLOBAL_FUNCTION_LEN:
            if args:
                return len(args[0])
            if "items" in kwargs:
                return len(kwargs["items"])
            raise ReplayError("len() missing argument")
        case ir.GlobalFunction.GLOBAL_FUNCTION_ENUMERATE:
            if args:
                return list(enumerate(args[0]))
            if "items" in kwargs:
                return list(enumerate(kwargs["items"]))
            raise ReplayError("enumerate() missing argument")
        case ir.GlobalFunction.GLOBAL_FUNCTION_ISEXCEPTION:
            if args:
                return _is_exception_value(args[0])
            if "value" in kwargs:
                return _is_exception_value(kwargs["value"])
            raise ReplayError("isexception() missing argument")
        case ir.GlobalFunction.GLOBAL_FUNCTION_UNSPECIFIED:
            raise ReplayError("global function unspecified")
        case _:
            assert_never(global_function)


def _apply_binary(op: ir.BinaryOperator, left: Any, right: Any) -> Any:
    match op:
        case ir.BinaryOperator.BINARY_OP_OR:
            return left or right
        case ir.BinaryOperator.BINARY_OP_AND:
            return left and right
        case ir.BinaryOperator.BINARY_OP_EQ:
            return left == right
        case ir.BinaryOperator.BINARY_OP_NE:
            return left != right
        case ir.BinaryOperator.BINARY_OP_LT:
            return left < right
        case ir.BinaryOperator.BINARY_OP_LE:
            return left <= right
        case ir.BinaryOperator.BINARY_OP_GT:
            return left > right
        case ir.BinaryOperator.BINARY_OP_GE:
            return left >= right
        case ir.BinaryOperator.BINARY_OP_IN:
            return left in right
        case ir.BinaryOperator.BINARY_OP_NOT_IN:
            return left not in right
        case ir.BinaryOperator.BINARY_OP_ADD:
            return left + right
        case ir.BinaryOperator.BINARY_OP_SUB:
            return left - right
        case ir.BinaryOperator.BINARY_OP_MUL:
            return left * right
        case ir.BinaryOperator.BINARY_OP_DIV:
            return left / right
        case ir.BinaryOperator.BINARY_OP_FLOOR_DIV:
            return left // right
        case ir.BinaryOperator.BINARY_OP_MOD:
            return left % right
        case ir.BinaryOperator.BINARY_OP_UNSPECIFIED:
            raise ReplayError("binary operator unspecified")
        case _:
            assert_never(op)


def _apply_unary(op: ir.UnaryOperator, operand: Any) -> Any:
    match op:
        case ir.UnaryOperator.UNARY_OP_NEG:
            return -operand
        case ir.UnaryOperator.UNARY_OP_NOT:
            return not operand
        case ir.UnaryOperator.UNARY_OP_UNSPECIFIED:
            raise ReplayError("unary operator unspecified")
        case _:
            assert_never(op)


def _is_exception_value(value: Any) -> bool:
    if isinstance(value, BaseException):
        return True
    if isinstance(value, dict) and "type" in value and "message" in value:
        return True
    return False


def _build_incoming_data_map(
    state: RunnerState,
    index: Mapping[UUID, int],
) -> dict[UUID, list[UUID]]:
    incoming: dict[UUID, list[UUID]] = {}
    for edge in state.edges:
        if edge.edge_type != EdgeType.DATA_FLOW:
            continue
        incoming.setdefault(edge.target, []).append(edge.source)
    for target, sources in incoming.items():
        sources.sort(key=lambda node_id: (index.get(node_id, -1), str(node_id)), reverse=True)
    return incoming
