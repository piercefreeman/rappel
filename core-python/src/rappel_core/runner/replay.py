"""Replay variable values from a runner state snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

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


def replay_variables(
    state: RunnerState,
    action_results: Mapping[str, Any],
) -> ReplayResult:
    variables: dict[str, Any] = {}
    cache: dict[tuple[str, str], Any] = {}
    timeline = state.timeline or list(state.nodes.keys())
    index = {node_id: idx for idx, node_id in enumerate(timeline)}

    for node_id in timeline:
        node = state.nodes.get(node_id)
        if node is None or not node.assignments:
            continue
        for target in node.assignments:
            variables[target] = _evaluate_assignment(
                state, node_id, target, action_results, cache, index, timeline, set()
            )

    return ReplayResult(variables=variables)


def _evaluate_assignment(
    state: RunnerState,
    node_id: str,
    target: str,
    action_results: Mapping[str, Any],
    cache: dict[tuple[str, str], Any],
    index: Mapping[str, int],
    timeline: Sequence[str],
    stack: set[tuple[str, str]],
) -> Any:
    key = (node_id, target)
    if key in cache:
        return cache[key]
    if key in stack:
        raise ReplayError(f"recursive assignment detected for {target} in {node_id}")

    node = state.nodes.get(node_id)
    if node is None or target not in node.assignments:
        raise ReplayError(f"missing assignment for {target} in {node_id}")

    stack.add(key)
    value = _evaluate_value(
        node.assignments[target],
        action_results,
        state,
        node_id,
        cache,
        index,
        timeline,
        stack,
    )
    stack.remove(key)
    cache[key] = value
    return value


def _evaluate_value(
    expr: ValueExpr,
    action_results: Mapping[str, Any],
    state: RunnerState,
    current_node_id: str,
    cache: dict[tuple[str, str], Any],
    index: Mapping[str, int],
    timeline: Sequence[str],
    stack: set[tuple[str, str]],
) -> Any:
    if isinstance(expr, LiteralValue):
        return expr.value
    if isinstance(expr, VariableValue):
        return _resolve_variable(
            state,
            current_node_id,
            expr.name,
            action_results,
            cache,
            index,
            timeline,
            stack,
        )
    if isinstance(expr, ActionResultValue):
        return _resolve_action_result(expr, action_results)
    if isinstance(expr, BinaryOpValue):
        left = _evaluate_value(
            expr.left, action_results, state, current_node_id, cache, index, timeline, stack
        )
        right = _evaluate_value(
            expr.right, action_results, state, current_node_id, cache, index, timeline, stack
        )
        return _apply_binary(expr.op, left, right)
    if isinstance(expr, UnaryOpValue):
        operand = _evaluate_value(
            expr.operand,
            action_results,
            state,
            current_node_id,
            cache,
            index,
            timeline,
            stack,
        )
        return _apply_unary(expr.op, operand)
    if isinstance(expr, ListValue):
        return [
            _evaluate_value(
                item, action_results, state, current_node_id, cache, index, timeline, stack
            )
            for item in expr.elements
        ]
    if isinstance(expr, DictValue):
        return {
            _evaluate_value(
                entry.key,
                action_results,
                state,
                current_node_id,
                cache,
                index,
                timeline,
                stack,
            ): _evaluate_value(
                entry.value,
                action_results,
                state,
                current_node_id,
                cache,
                index,
                timeline,
                stack,
            )
            for entry in expr.entries
        }
    if isinstance(expr, IndexValue):
        obj = _evaluate_value(
            expr.object, action_results, state, current_node_id, cache, index, timeline, stack
        )
        idx = _evaluate_value(
            expr.index, action_results, state, current_node_id, cache, index, timeline, stack
        )
        return obj[idx]
    if isinstance(expr, DotValue):
        obj = _evaluate_value(
            expr.object, action_results, state, current_node_id, cache, index, timeline, stack
        )
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
            action_results,
            state,
            current_node_id,
            cache,
            index,
            timeline,
            stack,
        )
    if isinstance(expr, SpreadValue):
        raise ReplayError("cannot replay unresolved spread expression")
    assert_never(expr)


def _resolve_variable(
    state: RunnerState,
    current_node_id: str,
    name: str,
    action_results: Mapping[str, Any],
    cache: dict[tuple[str, str], Any],
    index: Mapping[str, int],
    timeline: Sequence[str],
    stack: set[tuple[str, str]],
) -> Any:
    source_node_id = _find_variable_source_node(state, current_node_id, name, index, timeline)
    if source_node_id is None:
        raise ReplayError(f"variable not found: {name}")
    return _evaluate_assignment(
        state,
        source_node_id,
        name,
        action_results,
        cache,
        index,
        timeline,
        stack,
    )


def _find_variable_source_node(
    state: RunnerState,
    current_node_id: str,
    name: str,
    index: Mapping[str, int],
    timeline: Sequence[str],
) -> Optional[str]:
    candidates: list[str] = []
    for edge in state.edges:
        if edge.edge_type != EdgeType.DATA_FLOW or edge.target != current_node_id:
            continue
        node = state.nodes.get(edge.source)
        if node is None:
            continue
        if name in node.assignments:
            candidates.append(edge.source)

    if candidates:
        current_idx = index.get(current_node_id, len(timeline))
        eligible = [node_id for node_id in candidates if index.get(node_id, -1) <= current_idx]
        if eligible:
            return max(eligible, key=lambda node_id: index.get(node_id, -1))
        return max(candidates, key=lambda node_id: index.get(node_id, -1))

    current_idx = index.get(current_node_id, len(timeline))
    for node_id in reversed(timeline[: current_idx + 1]):
        node = state.nodes.get(node_id)
        if node is not None and name in node.assignments:
            return node_id
    return None


def _resolve_action_result(expr: ActionResultValue, action_results: Mapping[str, Any]) -> Any:
    if expr.node_id not in action_results:
        raise ReplayError(f"missing action result for {expr.node_id}")
    value = action_results[expr.node_id]
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
    action_results: Mapping[str, Any],
    state: RunnerState,
    current_node_id: str,
    cache: dict[tuple[str, str], Any],
    index: Mapping[str, int],
    timeline: Sequence[str],
    stack: set[tuple[str, str]],
) -> Any:
    args = [
        _evaluate_value(
            arg, action_results, state, current_node_id, cache, index, timeline, stack
        )
        for arg in expr.args
    ]
    kwargs = {
        name: _evaluate_value(
            value, action_results, state, current_node_id, cache, index, timeline, stack
        )
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
