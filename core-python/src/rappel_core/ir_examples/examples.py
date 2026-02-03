"""Collection of complex IR example programs."""

from __future__ import annotations

from typing import Callable

from proto import ast_pb2 as ir

from .helpers import (
    action_call,
    action_expr,
    assignment,
    binary,
    break_stmt,
    call_action,
    conditional_stmt,
    continue_stmt,
    dict_expr,
    dot_expr,
    except_handler,
    for_loop_stmt,
    function_def,
    function_expr,
    global_expr,
    index_expr,
    list_expr,
    literal,
    parallel_expr,
    program,
    return_stmt,
    spread_expr,
    try_except_stmt,
    variable,
    while_loop_stmt,
)


def build_control_flow_program() -> ir.Program:
    """Program with for/if/break/continue, action calls, and dict/index usage."""
    payload_expr = dict_expr(
        [
            ("items", list_expr([literal(1), literal(2), literal(3), literal(4)])),
            ("limit", variable("base")),
        ]
    )
    results_append = assignment(
        ["results"],
        binary(
            variable("results"),
            ir.BinaryOperator.BINARY_OP_ADD,
            list_expr([variable("item")]),
        ),
    )
    doubled_append = assignment(
        ["results"],
        binary(
            variable("results"),
            ir.BinaryOperator.BINARY_OP_ADD,
            list_expr([variable("doubled")]),
        ),
    )

    loop_body = [
        conditional_stmt(
            if_condition=binary(
                binary(
                    variable("item"),
                    ir.BinaryOperator.BINARY_OP_MOD,
                    literal(2),
                ),
                ir.BinaryOperator.BINARY_OP_EQ,
                literal(0),
            ),
            if_body=[
                assignment(
                    ["doubled"],
                    action_expr("double", {"value": variable("item")}),
                ),
                doubled_append,
                continue_stmt(),
            ],
            elifs=[
                (
                    binary(
                        variable("item"),
                        ir.BinaryOperator.BINARY_OP_GT,
                        variable("limit"),
                    ),
                    [break_stmt()],
                )
            ],
            else_body=[results_append],
        )
    ]

    statements = [
        assignment(["payload"], payload_expr),
        assignment(["items"], dot_expr(variable("payload"), "items")),
        assignment(["first_item"], index_expr(variable("items"), literal(0))),
        assignment(["limit"], dot_expr(variable("payload"), "limit")),
        assignment(["results"], list_expr([])),
        for_loop_stmt(
            ["idx", "item"],
            global_expr(
                ir.GlobalFunction.GLOBAL_FUNCTION_ENUMERATE,
                args=[variable("items")],
            ),
            loop_body,
        ),
        assignment(
            ["count"],
            global_expr(ir.GlobalFunction.GLOBAL_FUNCTION_LEN, args=[variable("results")]),
        ),
        assignment(
            ["summary"],
            dict_expr(
                [
                    ("count", variable("count")),
                    ("first", variable("first_item")),
                    ("results", variable("results")),
                ]
            ),
        ),
        return_stmt(variable("summary")),
    ]

    return program([
        function_def(
            "main",
            inputs=["base"],
            outputs=["summary"],
            body_statements=statements,
        )
    ])


def build_parallel_spread_program() -> ir.Program:
    """Program with spread and parallel expressions feeding action calls."""
    values_expr = global_expr(
        ir.GlobalFunction.GLOBAL_FUNCTION_RANGE,
        args=[
            literal(1),
            binary(variable("base"), ir.BinaryOperator.BINARY_OP_ADD, literal(1)),
        ],
    )
    doubles_expr = spread_expr(
        collection=variable("values"),
        loop_var="item",
        action=action_call("double", {"value": variable("item")}),
    )
    parallel_calls = [
        call_action(action_call("double", {"value": variable("base")})),
        call_action(
            action_call(
                "double",
                {
                    "value": binary(
                        variable("base"),
                        ir.BinaryOperator.BINARY_OP_ADD,
                        literal(1),
                    )
                },
            )
        ),
    ]
    parallel_value = parallel_expr(parallel_calls)

    statements = [
        assignment(["values"], values_expr),
        assignment(["doubles"], doubles_expr),
        assignment(["a", "b"], parallel_value),
        assignment(
            ["pair_sum"],
            binary(variable("a"), ir.BinaryOperator.BINARY_OP_ADD, variable("b")),
        ),
        assignment(
            ["total"],
            action_expr("sum", {"values": variable("doubles")}),
        ),
        assignment(
            ["final"],
            binary(variable("pair_sum"), ir.BinaryOperator.BINARY_OP_ADD, variable("total")),
        ),
        return_stmt(variable("final")),
    ]

    return program([
        function_def(
            "main",
            inputs=["base"],
            outputs=["final"],
            body_statements=statements,
        )
    ])


def build_try_except_program() -> ir.Program:
    """Program with try/except and a user-defined function call."""
    risky_body = [
        try_except_stmt(
            try_body=[
                assignment(
                    ["result"],
                    binary(
                        variable("numerator"),
                        ir.BinaryOperator.BINARY_OP_DIV,
                        variable("denominator"),
                    ),
                )
            ],
            handlers=[
                except_handler(
                    ["ZeroDivisionError"],
                    [assignment(["result"], literal(0))],
                    exception_var="err",
                )
            ],
        ),
        return_stmt(variable("result")),
    ]

    risky_fn = function_def(
        "risky",
        inputs=["numerator", "denominator"],
        outputs=["result"],
        body_statements=risky_body,
    )

    main_body = [
        assignment(["total"], literal(0)),
        for_loop_stmt(
            ["item"],
            variable("values"),
            [
                assignment(
                    ["denom"],
                    binary(
                        variable("item"),
                        ir.BinaryOperator.BINARY_OP_SUB,
                        literal(2),
                    ),
                ),
                assignment(
                    ["part"],
                    function_expr(
                        "risky",
                        kwargs={
                            "numerator": variable("item"),
                            "denominator": variable("denom"),
                        },
                    ),
                ),
                assignment(
                    ["total"],
                    binary(
                        variable("total"),
                        ir.BinaryOperator.BINARY_OP_ADD,
                        variable("part"),
                    ),
                ),
            ],
        ),
        return_stmt(variable("total")),
    ]

    main_fn = function_def(
        "main",
        inputs=["values"],
        outputs=["total"],
        body_statements=main_body,
    )

    return program([risky_fn, main_fn])


def build_while_loop_program() -> ir.Program:
    """Program with while-loop control flow and incremental updates."""
    loop_body = [
        assignment(
            ["accum"],
            binary(
                variable("accum"),
                ir.BinaryOperator.BINARY_OP_ADD,
                list_expr([variable("index")]),
            ),
        ),
        conditional_stmt(
            if_condition=binary(
                variable("index"),
                ir.BinaryOperator.BINARY_OP_EQ,
                literal(2),
            ),
            if_body=[
                assignment(
                    ["index"],
                    binary(
                        variable("index"),
                        ir.BinaryOperator.BINARY_OP_ADD,
                        literal(1),
                    ),
                ),
                continue_stmt(),
            ],
        ),
        conditional_stmt(
            if_condition=binary(
                variable("index"),
                ir.BinaryOperator.BINARY_OP_EQ,
                literal(4),
            ),
            if_body=[break_stmt()],
        ),
        assignment(
            ["index"],
            binary(
                variable("index"),
                ir.BinaryOperator.BINARY_OP_ADD,
                literal(1),
            ),
        ),
    ]

    statements = [
        assignment(["index"], literal(0)),
        assignment(["accum"], list_expr([])),
        while_loop_stmt(
            binary(
                variable("index"),
                ir.BinaryOperator.BINARY_OP_LT,
                variable("limit"),
            ),
            loop_body,
        ),
        return_stmt(variable("accum")),
    ]

    return program([
        function_def(
            "main",
            inputs=["limit"],
            outputs=["accum"],
            body_statements=statements,
        )
    ])


EXAMPLES: dict[str, Callable[[], ir.Program]] = {
    "control_flow": build_control_flow_program,
    "parallel_spread": build_parallel_spread_program,
    "try_except": build_try_except_program,
    "while_loop": build_while_loop_program,
}


def list_examples() -> list[str]:
    """Return the available example program names."""
    return sorted(EXAMPLES.keys())


def get_example(name: str) -> ir.Program:
    """Fetch an example IR program by name."""
    builder = EXAMPLES.get(name)
    if builder is None:
        raise KeyError(f"unknown example: {name}")
    return builder()
