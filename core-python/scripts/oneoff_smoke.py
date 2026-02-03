"""One-off smoke script to exercise DAG execution with the runloop."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

from proto import ast_pb2 as ir

from rappel_core import convert_to_dag, format_program, render_dag_image
from rappel_core.backends import MemoryBackend, QueuedInstance
from rappel_core.ir_executor import ExecutionError, StatementExecutor
from rappel_core.runloop import RunLoop
from rappel_core.runner import RunnerState, replay_variables
from rappel_core.runner.state import LiteralValue
from rappel_core.workers import InlineWorkerPool


def _literal_int(value: int) -> ir.Expr:
    return ir.Expr(literal=ir.Literal(int_value=value))


def _variable(name: str) -> ir.Expr:
    return ir.Expr(variable=ir.Variable(name=name))


def _binary(left: ir.Expr, op: ir.BinaryOperator, right: ir.Expr) -> ir.Expr:
    return ir.Expr(binary_op=ir.BinaryOp(left=left, op=op, right=right))


def _list(items: list[ir.Expr]) -> ir.Expr:
    return ir.Expr(list=ir.ListExpr(elements=items))


def _literal_from_value(value: Any) -> ir.Expr:
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
    raise ValueError(f"unsupported input literal: {value!r}")


def _build_program() -> ir.Program:
    values_expr = ir.Expr(
        list=ir.ListExpr(elements=[_literal_int(1), _literal_int(2), _literal_int(3)])
    )
    doubles_expr = ir.Expr(
        spread_expr=ir.SpreadExpr(
            collection=_variable("values"),
            loop_var="item",
            action=ir.ActionCall(
                action_name="double",
                kwargs=[ir.Kwarg(name="value", value=_variable("item"))],
            ),
        )
    )
    parallel_expr = ir.Expr(
        parallel_expr=ir.ParallelExpr(
            calls=[
                ir.Call(
                    action=ir.ActionCall(
                        action_name="double",
                        kwargs=[ir.Kwarg(name="value", value=_variable("base"))],
                    )
                ),
                ir.Call(
                    action=ir.ActionCall(
                        action_name="double",
                        kwargs=[
                            ir.Kwarg(
                                name="value",
                                value=_binary(
                                    _variable("base"),
                                    ir.BinaryOperator.BINARY_OP_ADD,
                                    _literal_int(1),
                                ),
                            )
                        ],
                    )
                ),
            ]
        )
    )

    statements = [
        ir.Statement(assignment=ir.Assignment(targets=["values"], value=values_expr)),
        ir.Statement(assignment=ir.Assignment(targets=["doubles"], value=doubles_expr)),
        ir.Statement(assignment=ir.Assignment(targets=["a", "b"], value=parallel_expr)),
        ir.Statement(
            assignment=ir.Assignment(
                targets=["pair_sum"],
                value=_binary(
                    _variable("a"),
                    ir.BinaryOperator.BINARY_OP_ADD,
                    _variable("b"),
                ),
            )
        ),
        ir.Statement(
            assignment=ir.Assignment(
                targets=["total"],
                value=ir.Expr(
                    action_call=ir.ActionCall(
                        action_name="sum",
                        kwargs=[ir.Kwarg(name="values", value=_variable("doubles"))],
                    )
                ),
            )
        ),
        ir.Statement(
            assignment=ir.Assignment(
                targets=["final"],
                value=_binary(
                    _variable("pair_sum"),
                    ir.BinaryOperator.BINARY_OP_ADD,
                    _variable("total"),
                ),
            )
        ),
        ir.Statement(return_stmt=ir.ReturnStmt(value=_variable("final"))),
    ]

    main_block = ir.Block(statements=statements)
    main_fn = ir.FunctionDef(
        name="main",
        io=ir.IoDecl(inputs=["base"], outputs=["final"]),
        body=main_block,
    )
    return ir.Program(functions=[main_fn])


async def _action_double(value: int) -> int:
    return value * 2


async def _action_sum(values: list[int]) -> int:
    return sum(values)


ACTION_REGISTRY = {"double": _action_double, "sum": _action_sum}


async def _action_handler(action: ir.ActionCall, kwargs: dict[str, Any]) -> Any:
    handler = ACTION_REGISTRY.get(action.action_name)
    if handler is None:
        raise ExecutionError(f"unknown action: {action.action_name}")
    return await handler(**kwargs)


def _build_runner_demo_state() -> tuple[RunnerState, dict[UUID, int]]:
    state = RunnerState()
    state.record_assignment(targets=["results"], expr=_list([]), label="results = []")

    action_results: dict[UUID, int] = {}
    for idx, item in enumerate([1, 2]):
        action_ref = state.queue_action(
            "action",
            targets=["action_result"],
            iteration_index=idx,
            kwargs={"item": LiteralValue(item)},
        )
        action_results[action_ref.node_id] = item
        action_plus = _binary(
            _variable("action_result"),
            ir.BinaryOperator.BINARY_OP_ADD,
            _literal_int(2),
        )
        concat_expr = _binary(
            _variable("results"), ir.BinaryOperator.BINARY_OP_ADD, _list([action_plus])
        )
        state.record_assignment(targets=["results"], expr=concat_expr)

    return state, action_results


async def main() -> int:
    program = _build_program()
    inputs = {"base": 5}

    print("IR program")
    print(format_program(program))
    print(f"IR inputs: {inputs}")

    dag = convert_to_dag(program)
    output_path = render_dag_image(dag, Path.cwd() / "dag_smoke.png")
    print(f"DAG image written to {output_path}")

    executor = StatementExecutor(program, _action_handler)
    result = await executor.execute_program(inputs=inputs)
    print(f"Execution result: {result}")

    demo_state, action_results = _build_runner_demo_state()
    replayed = replay_variables(demo_state, action_results)
    print(f"Runner replay variables: {replayed.variables}")

    state = RunnerState(dag=dag, link_queued_nodes=False)
    for name, value in inputs.items():
        state.record_assignment(
            targets=[name],
            expr=_literal_from_value(value),
            label=f"input {name} = {value!r}",
        )

    if dag.entry_node is None:
        raise RuntimeError("DAG entry node not found")

    entry_exec = state.queue_template_node(dag.entry_node)
    instance_queue: asyncio.Queue[QueuedInstance] = asyncio.Queue()
    backend = MemoryBackend(instance_queue)
    worker_pool = InlineWorkerPool(ACTION_REGISTRY)
    runloop = RunLoop(worker_pool, backend, poll_interval=0)

    await instance_queue.put(
        QueuedInstance(
            dag=dag,
            entry_node=entry_exec.node_id,
            state=state,
            action_results=action_results,
        )
    )
    runloop_result = await runloop.run()
    executed = next(iter(runloop_result.completed_actions.values()), [])
    print(f"Runner executor actions: {[node.label for node in executed]}")
    print(f"Runloop action results: {action_results}")
    final_replay = replay_variables(state, action_results)
    print(f"Runloop replay variables: {final_replay.variables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
