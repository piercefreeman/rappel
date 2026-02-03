"""Helper constructors for building IR example programs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from proto import ast_pb2 as ir


def literal(value: Any) -> ir.Expr:
    if isinstance(value, bool):
        return ir.Expr(literal=ir.Literal(bool_value=value))
    if isinstance(value, int):
        return ir.Expr(literal=ir.Literal(int_value=value))
    if isinstance(value, float):
        return ir.Expr(literal=ir.Literal(float_value=value))
    if isinstance(value, str):
        return ir.Expr(literal=ir.Literal(string_value=value))
    if value is None:
        return ir.Expr(literal=ir.Literal(is_none=True))
    raise ValueError(f"unsupported literal: {value!r}")


def variable(name: str) -> ir.Expr:
    return ir.Expr(variable=ir.Variable(name=name))


def binary(left: ir.Expr, op: ir.BinaryOperator, right: ir.Expr) -> ir.Expr:
    return ir.Expr(binary_op=ir.BinaryOp(left=left, op=op, right=right))


def unary(op: ir.UnaryOperator, operand: ir.Expr) -> ir.Expr:
    return ir.Expr(unary_op=ir.UnaryOp(op=op, operand=operand))


def list_expr(elements: Sequence[ir.Expr]) -> ir.Expr:
    return ir.Expr(list=ir.ListExpr(elements=list(elements)))


def dict_entry(key: str | ir.Expr, value: ir.Expr) -> ir.DictEntry:
    key_expr = key if isinstance(key, ir.Expr) else literal(key)
    return ir.DictEntry(key=key_expr, value=value)


def dict_expr(entries: Iterable[tuple[str | ir.Expr, ir.Expr]]) -> ir.Expr:
    return ir.Expr(dict=ir.DictExpr(entries=[dict_entry(key, value) for key, value in entries]))


def index_expr(object_expr: ir.Expr, index_value: ir.Expr) -> ir.Expr:
    return ir.Expr(index=ir.IndexAccess(object=object_expr, index=index_value))


def dot_expr(object_expr: ir.Expr, attribute: str) -> ir.Expr:
    return ir.Expr(dot=ir.DotAccess(object=object_expr, attribute=attribute))


def kwarg(name: str, value: ir.Expr) -> ir.Kwarg:
    return ir.Kwarg(name=name, value=value)


def action_call(
    name: str,
    kwargs: Mapping[str, ir.Expr] | None = None,
    module_name: str | None = None,
) -> ir.ActionCall:
    call_kwargs = [kwarg(key, value) for key, value in (kwargs or {}).items()]
    if module_name is None:
        return ir.ActionCall(action_name=name, kwargs=call_kwargs)
    return ir.ActionCall(action_name=name, kwargs=call_kwargs, module_name=module_name)


def action_expr(
    name: str,
    kwargs: Mapping[str, ir.Expr] | None = None,
    module_name: str | None = None,
) -> ir.Expr:
    return ir.Expr(action_call=action_call(name, kwargs=kwargs, module_name=module_name))


def function_call(
    name: str,
    args: Sequence[ir.Expr] | None = None,
    kwargs: Mapping[str, ir.Expr] | None = None,
) -> ir.FunctionCall:
    call_kwargs = [kwarg(key, value) for key, value in (kwargs or {}).items()]
    return ir.FunctionCall(name=name, args=list(args or []), kwargs=call_kwargs)


def function_expr(
    name: str,
    args: Sequence[ir.Expr] | None = None,
    kwargs: Mapping[str, ir.Expr] | None = None,
) -> ir.Expr:
    return ir.Expr(function_call=function_call(name, args=args, kwargs=kwargs))


def global_call(
    global_function: ir.GlobalFunction,
    args: Sequence[ir.Expr] | None = None,
    kwargs: Mapping[str, ir.Expr] | None = None,
) -> ir.FunctionCall:
    call_kwargs = [kwarg(key, value) for key, value in (kwargs or {}).items()]
    return ir.FunctionCall(
        global_function=global_function,
        args=list(args or []),
        kwargs=call_kwargs,
    )


def global_expr(
    global_function: ir.GlobalFunction,
    args: Sequence[ir.Expr] | None = None,
    kwargs: Mapping[str, ir.Expr] | None = None,
) -> ir.Expr:
    return ir.Expr(function_call=global_call(global_function, args=args, kwargs=kwargs))


def call_action(action: ir.ActionCall) -> ir.Call:
    return ir.Call(action=action)


def call_function(call: ir.FunctionCall) -> ir.Call:
    return ir.Call(function=call)


def parallel_expr(calls: Sequence[ir.Call]) -> ir.Expr:
    return ir.Expr(parallel_expr=ir.ParallelExpr(calls=list(calls)))


def spread_expr(collection: ir.Expr, loop_var: str, action: ir.ActionCall) -> ir.Expr:
    return ir.Expr(
        spread_expr=ir.SpreadExpr(collection=collection, loop_var=loop_var, action=action)
    )


def assignment(targets: Sequence[str], value: ir.Expr) -> ir.Statement:
    return ir.Statement(assignment=ir.Assignment(targets=list(targets), value=value))


def return_stmt(value: ir.Expr) -> ir.Statement:
    return ir.Statement(return_stmt=ir.ReturnStmt(value=value))


def break_stmt() -> ir.Statement:
    return ir.Statement(break_stmt=ir.BreakStmt())


def continue_stmt() -> ir.Statement:
    return ir.Statement(continue_stmt=ir.ContinueStmt())


def expr_stmt(value: ir.Expr) -> ir.Statement:
    return ir.Statement(expr_stmt=ir.ExprStmt(expr=value))


def block(statements: Sequence[ir.Statement]) -> ir.Block:
    return ir.Block(statements=list(statements))


def for_loop_stmt(
    loop_vars: Sequence[str],
    iterable: ir.Expr,
    body_statements: Sequence[ir.Statement],
) -> ir.Statement:
    return ir.Statement(
        for_loop=ir.ForLoop(
            loop_vars=list(loop_vars),
            iterable=iterable,
            block_body=block(body_statements),
        )
    )


def while_loop_stmt(condition: ir.Expr, body_statements: Sequence[ir.Statement]) -> ir.Statement:
    return ir.Statement(
        while_loop=ir.WhileLoop(condition=condition, block_body=block(body_statements))
    )


def if_branch(condition: ir.Expr, body_statements: Sequence[ir.Statement]) -> ir.IfBranch:
    return ir.IfBranch(condition=condition, block_body=block(body_statements))


def elif_branch(condition: ir.Expr, body_statements: Sequence[ir.Statement]) -> ir.ElifBranch:
    return ir.ElifBranch(condition=condition, block_body=block(body_statements))


def else_branch(body_statements: Sequence[ir.Statement]) -> ir.ElseBranch:
    return ir.ElseBranch(block_body=block(body_statements))


def conditional_stmt(
    if_condition: ir.Expr,
    if_body: Sequence[ir.Statement],
    elifs: Sequence[tuple[ir.Expr, Sequence[ir.Statement]]] | None = None,
    else_body: Sequence[ir.Statement] | None = None,
) -> ir.Statement:
    elif_branches = [
        elif_branch(condition, body) for condition, body in (elifs or [])
    ]
    else_value = else_branch(else_body) if else_body is not None else None
    return ir.Statement(
        conditional=ir.Conditional(
            if_branch=if_branch(if_condition, if_body),
            elif_branches=elif_branches,
            else_branch=else_value,
        )
    )


def except_handler(
    exception_types: Sequence[str],
    body_statements: Sequence[ir.Statement],
    exception_var: str | None = None,
) -> ir.ExceptHandler:
    handler_kwargs: dict[str, Any] = {
        "exception_types": list(exception_types),
        "block_body": block(body_statements),
    }
    if exception_var is not None:
        handler_kwargs["exception_var"] = exception_var
    return ir.ExceptHandler(**handler_kwargs)


def try_except_stmt(
    try_body: Sequence[ir.Statement],
    handlers: Sequence[ir.ExceptHandler],
) -> ir.Statement:
    return ir.Statement(
        try_except=ir.TryExcept(try_block=block(try_body), handlers=list(handlers))
    )


def function_def(
    name: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    body_statements: Sequence[ir.Statement],
) -> ir.FunctionDef:
    return ir.FunctionDef(
        name=name,
        io=ir.IoDecl(inputs=list(inputs), outputs=list(outputs)),
        body=block(body_statements),
    )


def program(functions: Sequence[ir.FunctionDef]) -> ir.Program:
    return ir.Program(functions=list(functions))
