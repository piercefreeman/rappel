from __future__ import annotations

import textwrap
from typing import Iterable
from unittest.mock import ANY

from proto import ast_pb2 as ir
from rappel_core.dag import DAGConverter, assert_never
from rappel_core.dag.models import EXCEPTION_SCOPE_VAR, DAGEdge, EdgeType
from rappel_core.dag.nodes import (
    ActionCallNode,
    AggregatorNode,
    AssignmentNode,
    BranchNode,
    BreakNode,
    ContinueNode,
    ExpressionNode,
    FnCallNode,
    InputNode,
    JoinNode,
    OutputNode,
    ParallelNode,
    ReturnNode,
)
from rappel_core.ir_parser import IRParser, parse_program


def build_dag(source: str):
    program = parse_program(textwrap.dedent(source).strip())
    return DAGConverter().convert_with_pointers(program)


def parse_expr(source: str) -> ir.Expr:
    return IRParser().parse_expr(source)


def normalize_len_calls(expr: ir.Expr) -> None:
    kind = expr.WhichOneof("kind")
    match kind:
        case "binary_op":
            normalize_len_calls(expr.binary_op.left)
            normalize_len_calls(expr.binary_op.right)
        case "unary_op":
            normalize_len_calls(expr.unary_op.operand)
        case "function_call":
            call = expr.function_call
            if call.global_function == ir.GlobalFunction.GLOBAL_FUNCTION_LEN and not call.name:
                call.name = "len"
            for arg in call.args:
                normalize_len_calls(arg)
            for kwarg in call.kwargs:
                if kwarg.HasField("value"):
                    normalize_len_calls(kwarg.value)
        case "list":
            for item in expr.list.elements:
                normalize_len_calls(item)
        case "dict":
            for entry in expr.dict.entries:
                if entry.HasField("key"):
                    normalize_len_calls(entry.key)
                if entry.HasField("value"):
                    normalize_len_calls(entry.value)
        case "index":
            normalize_len_calls(expr.index.object)
            normalize_len_calls(expr.index.index)
        case "dot":
            normalize_len_calls(expr.dot.object)
        case "action_call":
            for kwarg in expr.action_call.kwargs:
                if kwarg.HasField("value"):
                    normalize_len_calls(kwarg.value)
        case "parallel_expr":
            for call in expr.parallel_expr.calls:
                call_kind = call.WhichOneof("kind")
                match call_kind:
                    case "action":
                        for kwarg in call.action.kwargs:
                            if kwarg.HasField("value"):
                                normalize_len_calls(kwarg.value)
                    case "function":
                        for arg in call.function.args:
                            normalize_len_calls(arg)
                        for kwarg in call.function.kwargs:
                            if kwarg.HasField("value"):
                                normalize_len_calls(kwarg.value)
                    case None:
                        continue
                    case _:
                        assert_never(call_kind)
        case "spread_expr":
            normalize_len_calls(expr.spread_expr.collection)
            for kwarg in expr.spread_expr.action.kwargs:
                if kwarg.HasField("value"):
                    normalize_len_calls(kwarg.value)
        case "literal" | "variable" | None:
            return
        case _:
            assert_never(kind)


def parse_guard(source: str) -> ir.Expr:
    expr = parse_expr(source)
    normalize_len_calls(expr)
    return expr


def input_node(node_id: str, inputs: list[str]) -> InputNode:
    return InputNode(id=node_id, io_vars=inputs, function_name="main", node_uuid=ANY)


def output_node(node_id: str, outputs: list[str]) -> OutputNode:
    return OutputNode(id=node_id, io_vars=outputs, function_name="main", node_uuid=ANY)


def action_node(
    node_id: str,
    action_name: str,
    *,
    targets: list[str] | None = None,
    target: str | None = None,
    parallel_index: int | None = None,
    aggregates_to: str | None = None,
    spread_loop_var: str | None = None,
    spread_collection_expr: ir.Expr | None = None,
) -> ActionCallNode:
    return ActionCallNode(
        id=node_id,
        action_name=action_name,
        module_name=None,
        kwargs={},
        kwarg_exprs={},
        policies=[],
        targets=targets,
        target=target,
        parallel_index=parallel_index,
        aggregates_to=aggregates_to,
        spread_loop_var=spread_loop_var,
        spread_collection_expr=spread_collection_expr,
        function_name="main",
        node_uuid=ANY,
    )


def fn_call_node(
    node_id: str,
    called_function: str,
    *,
    targets: list[str] | None = None,
    assign_expr: ir.Expr | None = None,
) -> FnCallNode:
    return FnCallNode(
        id=node_id,
        called_function=called_function,
        kwargs={},
        kwarg_exprs={},
        targets=targets,
        assign_expr=assign_expr,
        function_name="main",
        node_uuid=ANY,
    )


def assignment_node(
    node_id: str,
    targets: list[str],
    *,
    assign_expr: ir.Expr | None = None,
    label_hint: str | None = None,
) -> AssignmentNode:
    return AssignmentNode(
        id=node_id,
        targets=targets,
        assign_expr=assign_expr,
        label_hint=label_hint,
        function_name="main",
        node_uuid=ANY,
    )


def aggregator_node(
    node_id: str,
    aggregates_from: str,
    *,
    targets: list[str] | None = None,
    aggregator_kind: str = "aggregate",
) -> AggregatorNode:
    return AggregatorNode(
        id=node_id,
        aggregates_from=aggregates_from,
        targets=targets,
        aggregator_kind=aggregator_kind,
        function_name="main",
        node_uuid=ANY,
    )


def branch_node(node_id: str, description: str) -> BranchNode:
    return BranchNode(
        id=node_id,
        description=description,
        function_name="main",
        node_uuid=ANY,
    )


def join_node(node_id: str, description: str) -> JoinNode:
    return JoinNode(
        id=node_id,
        description=description,
        function_name="main",
        node_uuid=ANY,
    )


def return_node(
    node_id: str,
    *,
    assign_expr: ir.Expr | None = None,
    target: str | None = None,
) -> ReturnNode:
    return ReturnNode(
        id=node_id,
        assign_expr=assign_expr,
        target=target,
        function_name="main",
        node_uuid=ANY,
    )


def expression_node(node_id: str) -> ExpressionNode:
    return ExpressionNode(id=node_id, function_name="main", node_uuid=ANY)


def break_node(node_id: str) -> BreakNode:
    return BreakNode(id=node_id, function_name="main", node_uuid=ANY)


def continue_node(node_id: str) -> ContinueNode:
    return ContinueNode(id=node_id, function_name="main", node_uuid=ANY)


def assert_nodes(dag, expected_nodes: Iterable):
    expected_by_id = {node.id: node for node in expected_nodes}
    assert expected_by_id == dag.nodes


def assert_edge_pairs(
    dag,
    edge_type: EdgeType,
    expected_edges: Iterable[tuple],
) -> None:
    expected = {(source.id, target.id) for source, target in expected_edges}
    actual = {(edge.source, edge.target) for edge in dag.edges if edge.edge_type == edge_type}
    assert actual == expected


def get_edge(dag, source, target, edge_type: EdgeType) -> DAGEdge:
    matches = [
        edge
        for edge in dag.edges
        if edge.edge_type == edge_type and edge.source == source.id and edge.target == target.id
    ]
    assert len(matches) == 1
    return matches[0]


def test_assignment_literal_builds_assignment_node() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            value = 42
        """
    )

    input_def = input_node("main_input_1", [])
    assign_def = assignment_node("assign_2", ["value"], assign_expr=parse_expr("42"))
    output_def = output_node("main_output_3", [])

    assert_nodes(dag, [input_def, assign_def, output_def])

    state_edges = [
        # input -> assignment
        (input_def, assign_def),
        # assignment -> output
        (assign_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])


def test_assignment_action_call_builds_action_node() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: [result]):
            result = @noop()
        """
    )

    input_def = input_node("main_input_1", [])
    action_def = action_node("action_2", "noop", targets=["result"])
    output_def = output_node("main_output_3", ["result"])

    assert_nodes(dag, [input_def, action_def, output_def])

    state_edges = [
        # input -> action call assignment
        (input_def, action_def),
        # action call -> output
        (action_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])


def test_assignment_function_call_builds_fn_call_node() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: [result]):
            result = helper()
        """
    )

    input_def = input_node("main_input_1", [])
    fn_call_def = fn_call_node(
        "fn_call_2",
        "helper",
        targets=["result"],
        assign_expr=parse_expr("helper()"),
    )
    output_def = output_node("main_output_3", ["result"])

    assert_nodes(dag, [input_def, fn_call_def, output_def])

    state_edges = [
        # input -> function call assignment
        (input_def, fn_call_def),
        # function call -> output
        (fn_call_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])


def test_spread_expression_builds_action_and_aggregator() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: [results]):
            results = spread [1, 2]: item -> @noop()
        """
    )

    collection_expr = parse_expr("[1, 2]")

    input_def = input_node("main_input_1", [])
    action_def = action_node(
        "spread_action_2",
        "noop",
        target="_spread_result",
        aggregates_to="aggregator_3",
        spread_loop_var="item",
        spread_collection_expr=collection_expr,
    )
    aggregator_def = aggregator_node(
        "aggregator_3",
        "spread_action_2",
        targets=["results"],
    )
    output_def = output_node("main_output_4", ["results"])

    assert_nodes(dag, [input_def, action_def, aggregator_def, output_def])

    state_edges = [
        # input -> spread action
        (input_def, action_def),
        # spread action -> aggregator
        (action_def, aggregator_def),
        # aggregator -> output
        (aggregator_def, output_def),
    ]
    data_edges = [
        # spread result -> aggregator collects results
        (action_def, aggregator_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, data_edges)
    spread_edge = get_edge(dag, action_def, aggregator_def, EdgeType.DATA_FLOW)
    assert spread_edge.variable == "_spread_result"


def test_spread_action_statement_builds_action_and_aggregator() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            spread [1, 2]: item -> @noop()
        """
    )

    collection_expr = parse_expr("[1, 2]")

    input_def = input_node("main_input_1", [])
    action_def = action_node(
        "spread_action_2",
        "noop",
        target="_spread_result",
        aggregates_to="aggregator_3",
        spread_loop_var="item",
        spread_collection_expr=collection_expr,
    )
    aggregator_def = aggregator_node("aggregator_3", "spread_action_2")
    output_def = output_node("main_output_4", [])

    assert_nodes(dag, [input_def, action_def, aggregator_def, output_def])

    state_edges = [
        # input -> spread action
        (input_def, action_def),
        # spread action -> aggregator
        (action_def, aggregator_def),
        # aggregator -> output
        (aggregator_def, output_def),
    ]
    data_edges = [
        # spread result -> aggregator collects results
        (action_def, aggregator_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, data_edges)
    spread_edge = get_edge(dag, action_def, aggregator_def, EdgeType.DATA_FLOW)
    assert spread_edge.variable == "_spread_result"


def test_parallel_block_builds_parallel_graph() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            parallel:
                @a()
                @b()
        """
    )

    input_def = input_node("main_input_1", [])
    parallel_def = ParallelNode(id="parallel_2", function_name="main", node_uuid=ANY)
    action_one = action_node("parallel_action_4", "a", parallel_index=0)
    action_two = action_node("parallel_action_5", "b", parallel_index=1)
    aggregator_def = aggregator_node(
        "parallel_aggregator_3",
        "parallel_2",
        aggregator_kind="parallel",
    )
    output_def = output_node("main_output_6", [])

    assert_nodes(dag, [input_def, parallel_def, action_one, action_two, aggregator_def, output_def])

    state_edges = [
        # input -> parallel fan-out
        (input_def, parallel_def),
        # parallel -> action[0]
        (parallel_def, action_one),
        # parallel -> action[1]
        (parallel_def, action_two),
        # action[0] -> aggregator
        (action_one, aggregator_def),
        # action[1] -> aggregator
        (action_two, aggregator_def),
        # aggregator -> output
        (aggregator_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])

    edge_zero = get_edge(dag, parallel_def, action_one, EdgeType.STATE_MACHINE)
    edge_one = get_edge(dag, parallel_def, action_two, EdgeType.STATE_MACHINE)
    assert edge_zero.condition == "parallel:0"
    assert edge_one.condition == "parallel:1"


def test_parallel_expression_builds_parallel_graph_with_targets() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: [a, b]):
            a, b = parallel:
                @x()
                @y()
        """
    )

    input_def = input_node("main_input_1", [])
    parallel_def = ParallelNode(id="parallel_2", function_name="main", node_uuid=ANY)
    action_one = action_node("parallel_action_4", "x", target="a", parallel_index=0)
    action_two = action_node("parallel_action_5", "y", target="b", parallel_index=1)
    aggregator_def = aggregator_node(
        "parallel_aggregator_3",
        "parallel_2",
        targets=["a", "b"],
        aggregator_kind="parallel",
    )
    output_def = output_node("main_output_6", ["a", "b"])

    assert_nodes(dag, [input_def, parallel_def, action_one, action_two, aggregator_def, output_def])

    state_edges = [
        # input -> parallel fan-out
        (input_def, parallel_def),
        # parallel -> action[0]
        (parallel_def, action_one),
        # parallel -> action[1]
        (parallel_def, action_two),
        # action[0] -> aggregator
        (action_one, aggregator_def),
        # action[1] -> aggregator
        (action_two, aggregator_def),
        # aggregator -> output
        (aggregator_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])

    edge_zero = get_edge(dag, parallel_def, action_one, EdgeType.STATE_MACHINE)
    edge_one = get_edge(dag, parallel_def, action_two, EdgeType.STATE_MACHINE)
    assert edge_zero.condition == "parallel:0"
    assert edge_one.condition == "parallel:1"


def test_action_call_statement_builds_action_node() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            @noop()
        """
    )

    input_def = input_node("main_input_1", [])
    action_def = action_node("action_2", "noop")
    output_def = output_node("main_output_3", [])

    assert_nodes(dag, [input_def, action_def, output_def])

    state_edges = [
        # input -> action call statement
        (input_def, action_def),
        # action call -> output
        (action_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])


def test_expr_statement_function_call_builds_fn_call_node() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            helper()
        """
    )

    input_def = input_node("main_input_1", [])
    fn_call_def = fn_call_node("fn_call_2", "helper")
    output_def = output_node("main_output_3", [])

    assert_nodes(dag, [input_def, fn_call_def, output_def])

    state_edges = [
        # input -> function call statement
        (input_def, fn_call_def),
        # function call -> output
        (fn_call_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])


def test_expr_statement_builds_expression_node() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            1 + 2
        """
    )

    input_def = input_node("main_input_1", [])
    expr_def = expression_node("expr_2")
    output_def = output_node("main_output_3", [])

    assert_nodes(dag, [input_def, expr_def, output_def])

    state_edges = [
        # input -> expression statement
        (input_def, expr_def),
        # expression -> output
        (expr_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])


def test_for_loop_builds_loop_nodes() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            for item in [1, 2]:
                @work()
        """
    )

    loop_var = "__loop_loop_2_i"
    continue_guard = parse_guard(f"{loop_var} < len(items=[1, 2])")
    break_guard = parse_guard(f"not ({loop_var} < len(items=[1, 2]))")

    input_def = input_node("main_input_1", [])
    init_def = assignment_node(
        "loop_init_3",
        [loop_var],
        assign_expr=parse_expr("0"),
        label_hint=f"{loop_var} = 0",
    )
    cond_def = branch_node("loop_cond_4", "for item in [1, 2]")
    extract_def = assignment_node(
        "loop_extract_5",
        ["item"],
        assign_expr=parse_expr(f"[1, 2][{loop_var}]"),
        label_hint=f"item = [1, 2][{loop_var}]",
    )
    action_def = action_node("action_8", "work")
    incr_def = assignment_node(
        "loop_incr_7",
        [loop_var],
        assign_expr=parse_expr(f"{loop_var} + 1"),
        label_hint=f"{loop_var} = {loop_var} + 1",
    )
    exit_def = join_node("loop_exit_6", "end for item")
    output_def = output_node("main_output_9", [])

    assert_nodes(
        dag,
        [
            input_def,
            init_def,
            cond_def,
            extract_def,
            action_def,
            incr_def,
            exit_def,
            output_def,
        ],
    )

    state_edges = [
        # input -> loop init
        (input_def, init_def),
        # loop init -> loop condition
        (init_def, cond_def),
        # loop condition -> loop extract
        (cond_def, extract_def),
        # loop condition -> loop exit
        (cond_def, exit_def),
        # loop extract -> loop body action
        (extract_def, action_def),
        # loop body action -> loop increment
        (action_def, incr_def),
        # loop increment -> loop condition
        (incr_def, cond_def),
        # loop exit -> output
        (exit_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])

    guarded_edge = get_edge(dag, cond_def, extract_def, EdgeType.STATE_MACHINE)
    assert guarded_edge.condition == "guarded"
    assert guarded_edge.guard_expr == continue_guard

    break_edge = get_edge(dag, cond_def, exit_def, EdgeType.STATE_MACHINE)
    assert break_edge.condition == "guarded"
    assert break_edge.guard_expr == break_guard

    loop_back = get_edge(dag, incr_def, cond_def, EdgeType.STATE_MACHINE)
    assert loop_back.is_loop_back is True


def test_while_loop_builds_loop_nodes() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            while True:
                @work()
        """
    )

    continue_guard = parse_expr("True")
    break_guard = parse_expr("not True")

    input_def = input_node("main_input_1", [])
    cond_def = branch_node("loop_cond_2", "while true")
    action_def = action_node("action_5", "work")
    continue_def = assignment_node(
        "loop_continue_4",
        [],
        label_hint="loop_continue",
    )
    exit_def = join_node("loop_exit_3", "end while true")
    output_def = output_node("main_output_6", [])

    assert_nodes(
        dag,
        [
            input_def,
            cond_def,
            action_def,
            continue_def,
            exit_def,
            output_def,
        ],
    )

    state_edges = [
        # input -> loop condition
        (input_def, cond_def),
        # loop condition -> body action
        (cond_def, action_def),
        # body action -> loop continue
        (action_def, continue_def),
        # loop continue -> loop condition
        (continue_def, cond_def),
        # loop condition -> loop exit
        (cond_def, exit_def),
        # loop exit -> output
        (exit_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])

    guarded_edge = get_edge(dag, cond_def, action_def, EdgeType.STATE_MACHINE)
    assert guarded_edge.condition == "guarded"
    assert guarded_edge.guard_expr == continue_guard

    break_edge = get_edge(dag, cond_def, exit_def, EdgeType.STATE_MACHINE)
    assert break_edge.condition == "guarded"
    assert break_edge.guard_expr == break_guard

    loop_back = get_edge(dag, continue_def, cond_def, EdgeType.STATE_MACHINE)
    assert loop_back.is_loop_back is True


def test_conditional_builds_branch_join_graph() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            if True:
                @first()
            elif False:
                @second()
            else:
                @third()
        """
    )

    if_guard = parse_expr("True")
    elif_guard = parse_expr("not True and False")

    input_def = input_node("main_input_1", [])
    branch_def = branch_node("branch_2", "branch")
    first_def = action_node("action_3", "first")
    second_def = action_node("action_4", "second")
    third_def = action_node("action_5", "third")
    join_def = join_node("join_6", "join")
    output_def = output_node("main_output_7", [])

    assert_nodes(
        dag,
        [input_def, branch_def, first_def, second_def, third_def, join_def, output_def],
    )

    state_edges = [
        # input -> branch
        (input_def, branch_def),
        # branch -> if body
        (branch_def, first_def),
        # branch -> elif body
        (branch_def, second_def),
        # branch -> else body
        (branch_def, third_def),
        # if body -> join
        (first_def, join_def),
        # elif body -> join
        (second_def, join_def),
        # else body -> join
        (third_def, join_def),
        # join -> output
        (join_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])

    if_edge = get_edge(dag, branch_def, first_def, EdgeType.STATE_MACHINE)
    assert if_edge.condition == "guarded"
    assert if_edge.guard_expr == if_guard

    elif_edge = get_edge(dag, branch_def, second_def, EdgeType.STATE_MACHINE)
    assert elif_edge.condition == "guarded"
    assert elif_edge.guard_expr == elif_guard

    else_edge = get_edge(dag, branch_def, third_def, EdgeType.STATE_MACHINE)
    assert else_edge.condition == "else"
    assert else_edge.is_else is True


def test_try_except_builds_exception_edges() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            try:
                @work()
            except Exception as err:
                @handle()
        """
    )

    input_def = input_node("main_input_1", [])
    try_def = action_node("action_2", "work")
    handler_def = action_node("action_3", "handle")
    bind_def = assignment_node(
        "exc_bind_4",
        ["err"],
        assign_expr=parse_expr(EXCEPTION_SCOPE_VAR),
        label_hint=f"err = {EXCEPTION_SCOPE_VAR}",
    )
    join_def = join_node("join_5", "join")
    output_def = output_node("main_output_6", [])

    assert_nodes(dag, [input_def, try_def, handler_def, bind_def, join_def, output_def])

    state_edges = [
        # input -> try action
        (input_def, try_def),
        # try action -> join on success
        (try_def, join_def),
        # try action -> exception binding
        (try_def, bind_def),
        # exception binding -> handler action
        (bind_def, handler_def),
        # handler action -> join
        (handler_def, join_def),
        # join -> output
        (join_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])

    success_edge = get_edge(dag, try_def, join_def, EdgeType.STATE_MACHINE)
    assert success_edge.condition == "success"

    exc_edge = get_edge(dag, try_def, bind_def, EdgeType.STATE_MACHINE)
    assert exc_edge.condition == "except:*"
    assert exc_edge.exception_types == []
    assert exc_edge.exception_depth == 1


def test_return_statement_builds_return_node() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: [result]):
            return 1
        """
    )

    input_def = input_node("main_input_1", [])
    return_def = return_node("return_2", assign_expr=parse_expr("1"), target="result")
    output_def = output_node("main_output_3", ["result"])

    assert_nodes(dag, [input_def, return_def, output_def])

    state_edges = [
        # input -> return
        (input_def, return_def),
        # return -> output
        (return_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])


def test_break_statement_wires_loop_exit() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            for item in [1]:
                break
        """
    )

    loop_var = "__loop_loop_2_i"
    continue_guard = parse_guard(f"{loop_var} < len(items=[1])")
    break_guard = parse_guard(f"not ({loop_var} < len(items=[1]))")

    input_def = input_node("main_input_1", [])
    init_def = assignment_node(
        "loop_init_3",
        [loop_var],
        assign_expr=parse_expr("0"),
        label_hint=f"{loop_var} = 0",
    )
    cond_def = branch_node("loop_cond_4", "for item in [1]")
    extract_def = assignment_node(
        "loop_extract_5",
        ["item"],
        assign_expr=parse_expr(f"[1][{loop_var}]"),
        label_hint=f"item = [1][{loop_var}]",
    )
    break_def = break_node("break_8")
    incr_def = assignment_node(
        "loop_incr_7",
        [loop_var],
        assign_expr=parse_expr(f"{loop_var} + 1"),
        label_hint=f"{loop_var} = {loop_var} + 1",
    )
    exit_def = join_node("loop_exit_6", "end for item")
    output_def = output_node("main_output_9", [])

    assert_nodes(
        dag,
        [
            input_def,
            init_def,
            cond_def,
            extract_def,
            break_def,
            incr_def,
            exit_def,
            output_def,
        ],
    )

    state_edges = [
        # input -> loop init
        (input_def, init_def),
        # loop init -> loop condition
        (init_def, cond_def),
        # loop condition -> loop extract
        (cond_def, extract_def),
        # loop condition -> loop exit
        (cond_def, exit_def),
        # loop extract -> break
        (extract_def, break_def),
        # break -> loop exit
        (break_def, exit_def),
        # loop increment -> loop condition
        (incr_def, cond_def),
        # loop exit -> output
        (exit_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])

    continue_edge = get_edge(dag, cond_def, extract_def, EdgeType.STATE_MACHINE)
    assert continue_edge.condition == "guarded"
    assert continue_edge.guard_expr == continue_guard

    break_edge = get_edge(dag, cond_def, exit_def, EdgeType.STATE_MACHINE)
    assert break_edge.condition == "guarded"
    assert break_edge.guard_expr == break_guard

    loop_back = get_edge(dag, incr_def, cond_def, EdgeType.STATE_MACHINE)
    assert loop_back.is_loop_back is True


def test_continue_statement_wires_loop_continue() -> None:
    dag = build_dag(
        """
        fn main(input: [], output: []):
            while True:
                continue
        """
    )

    continue_guard = parse_expr("True")
    break_guard = parse_expr("not True")

    input_def = input_node("main_input_1", [])
    cond_def = branch_node("loop_cond_2", "while true")
    continue_stmt_def = continue_node("continue_5")
    loop_continue_def = assignment_node(
        "loop_continue_4",
        [],
        label_hint="loop_continue",
    )
    exit_def = join_node("loop_exit_3", "end while true")
    output_def = output_node("main_output_6", [])

    assert_nodes(
        dag,
        [
            input_def,
            cond_def,
            continue_stmt_def,
            loop_continue_def,
            exit_def,
            output_def,
        ],
    )

    state_edges = [
        # input -> loop condition
        (input_def, cond_def),
        # loop condition -> continue
        (cond_def, continue_stmt_def),
        # continue -> loop continue node
        (continue_stmt_def, loop_continue_def),
        # loop continue -> loop condition
        (loop_continue_def, cond_def),
        # loop condition -> loop exit
        (cond_def, exit_def),
        # loop exit -> output
        (exit_def, output_def),
    ]
    assert_edge_pairs(dag, EdgeType.STATE_MACHINE, state_edges)
    assert_edge_pairs(dag, EdgeType.DATA_FLOW, [])

    continue_edge = get_edge(dag, cond_def, continue_stmt_def, EdgeType.STATE_MACHINE)
    assert continue_edge.condition == "guarded"
    assert continue_edge.guard_expr == continue_guard

    break_edge = get_edge(dag, cond_def, exit_def, EdgeType.STATE_MACHINE)
    assert break_edge.condition == "guarded"
    assert break_edge.guard_expr == break_guard

    loop_back = get_edge(dag, loop_continue_def, cond_def, EdgeType.STATE_MACHINE)
    assert loop_back.is_loop_back is True
