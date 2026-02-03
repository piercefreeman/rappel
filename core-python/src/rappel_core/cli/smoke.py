"""CLI smoke check for core-python components."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from proto import ast_pb2 as ir

from ..backends import MemoryBackend, QueuedInstance
from ..dag import DAG, convert_to_dag
from ..dag_viz import render_dag_image
from ..ir_examples import EXAMPLES
from ..ir_executor import ExecutionError, StatementExecutor
from ..ir_format import format_program
from ..runloop import RunLoop
from ..runner import RunnerState, replay_variables
from ..runner.state import LiteralValue
from ..workers import InlineWorkerPool


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
        list=ir.ListExpr(
            elements=[
                _literal_int(1),
                _literal_int(2),
                _literal_int(3),
            ]
        )
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


ACTION_REGISTRY = {
    "double": _action_double,
    "sum": _action_sum,
}


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


async def _run_executor_demo(dag: DAG, inputs: dict[str, Any]) -> None:
    state = RunnerState(dag=dag, link_queued_nodes=False)
    instance_queue: asyncio.Queue[QueuedInstance] = asyncio.Queue()
    backend = MemoryBackend(instance_queue)
    for name, value in inputs.items():
        state.record_assignment(
            targets=[name],
            expr=_literal_from_value(value),
            label=f"input {name} = {value!r}",
        )

    if dag.entry_node is None:
        raise RuntimeError("DAG entry node not found")

    entry_exec = state.queue_template_node(dag.entry_node)
    action_results: dict[UUID, Any] = {}
    worker_pool = InlineWorkerPool(ACTION_REGISTRY)
    runloop = RunLoop(worker_pool, backend)
    await instance_queue.put(
        QueuedInstance(
            dag=dag,
            entry_node=entry_exec.node_id,
            state=state,
            action_results=action_results,
        )
    )
    result = await runloop.run()
    executed = next(iter(result.completed_actions.values()), [])
    print("Runner executor actions: %s" % [node.label for node in executed])


@dataclass(frozen=True)
class SmokeCase:
    name: str
    program: ir.Program
    inputs: dict[str, Any]
    run_runner_demo: bool = False


def _slugify(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


async def _run_program_smoke(case: SmokeCase) -> None:
    print("\nIR program (%s)" % case.name)
    print(format_program(case.program))
    print("IR inputs (%s): %s" % (case.name, case.inputs))
    dag = convert_to_dag(case.program)
    slug = _slugify(case.name)
    output_path = render_dag_image(dag, Path.cwd() / f"dag_smoke_{slug}.png")
    print("DAG image (%s) written to %s" % (case.name, output_path))

    executor = StatementExecutor(case.program, _action_handler)
    result = await executor.execute_program(inputs=case.inputs)
    print("Execution result (%s): %s" % (case.name, result))

    if case.run_runner_demo:
        demo_state, action_results = _build_runner_demo_state()
        replayed = replay_variables(demo_state, action_results)
        print("Runner replay variables: %s" % replayed.variables)
        await _run_executor_demo(dag, case.inputs)


async def _run_smoke(base: int) -> int:
    cases = [
        SmokeCase(
            name="smoke",
            program=_build_program(),
            inputs={"base": base},
            run_runner_demo=True,
        ),
        SmokeCase(
            name="control_flow",
            program=EXAMPLES["control_flow"](),
            inputs={"base": 2},
        ),
        SmokeCase(
            name="parallel_spread",
            program=EXAMPLES["parallel_spread"](),
            inputs={"base": 3},
        ),
        SmokeCase(
            name="try_except",
            program=EXAMPLES["try_except"](),
            inputs={"values": [1, 2, 3]},
        ),
        SmokeCase(
            name="while_loop",
            program=EXAMPLES["while_loop"](),
            inputs={"limit": 6},
        ),
    ]

    failures = 0
    for case in cases:
        try:
            await _run_program_smoke(case)
        except Exception as exc:
            failures += 1
            print("Smoke case '%s' failed: %s" % (case.name, exc))

    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke check core-python components.")
    parser.add_argument("--base", type=int, default=5, help="Base input for the demo program.")
    args = parser.parse_args()

    raise SystemExit(asyncio.run(_run_smoke(args.base)))
