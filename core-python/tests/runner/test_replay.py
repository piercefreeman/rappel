from proto import ast_pb2 as ir
from rappel_core.runner.replay import replay_variables
from rappel_core.runner.state import (
    ActionResultValue,
    RunnerState,
)


def _action_plus_two_expr() -> ir.Expr:
    return ir.Expr(
        binary_op=ir.BinaryOp(
            left=ir.Expr(variable=ir.Variable(name="action_result")),
            op=ir.BinaryOperator.BINARY_OP_ADD,
            right=ir.Expr(literal=ir.Literal(int_value=2)),
        )
    )


def test_replay_variables_resolves_action_results() -> None:
    state = RunnerState()

    action0 = state.queue_action("action", targets=["action_result"], iteration_index=0)
    state.record_assignment(
        targets=["results"],
        expr=ir.Expr(list=ir.ListExpr(elements=[_action_plus_two_expr()])),
    )

    action1 = state.queue_action("action", targets=["action_result"], iteration_index=1)
    concat_expr = ir.Expr(
        binary_op=ir.BinaryOp(
            left=ir.Expr(variable=ir.Variable(name="results")),
            op=ir.BinaryOperator.BINARY_OP_ADD,
            right=ir.Expr(list=ir.ListExpr(elements=[_action_plus_two_expr()])),
        )
    )
    state.record_assignment(targets=["results"], expr=concat_expr)

    assert isinstance(action0, ActionResultValue)
    assert isinstance(action1, ActionResultValue)

    replayed = replay_variables(
        state,
        {
            action0.node_id: 1,
            action1.node_id: 2,
        },
    )

    assert replayed.variables["results"] == [3, 4]
