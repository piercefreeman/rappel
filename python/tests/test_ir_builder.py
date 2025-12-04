"""Tests for IR builder functionality.

This module contains comprehensive tests for the Python AST to IR conversion.
The tests are organized by category:
- TestAsyncioSleepDetection: asyncio.sleep() -> @sleep action
- TestAsyncioGatherDetection: asyncio.gather() -> ParallelBlock
- TestPolicyParsing: RetryPolicy and TimeoutPolicy extraction
- TestForLoopConversion: for loop IR generation
- TestConditionalConversion: if/elif/else IR generation
- TestTryExceptConversion: try/except IR generation
- TestActionCallExtraction: action call detection and kwargs
- TestImplicitFunctionWrapping: multi-action body wrapping
"""

from typing import List, Optional

from proto import ast_pb2 as ir


class TestAsyncioSleepDetection:
    """Test that asyncio.sleep is detected and converted to @sleep action."""

    def _find_sleep_action(self, program: ir.Program) -> ir.ActionCall | None:
        """Find a @sleep action call in the program."""
        for fn in program.functions:
            for stmt in fn.body.statements:
                if stmt.HasField("action_call"):
                    if stmt.action_call.action_name == "sleep":
                        return stmt.action_call
        return None

    def _get_duration_kwarg(self, action_call: ir.ActionCall) -> ir.Kwarg | None:
        """Get the duration kwarg from a sleep action."""
        for kw in action_call.kwargs:
            if kw.name == "duration":
                return kw
        return None

    def test_asyncio_dot_sleep_pattern(self) -> None:
        """Test: import asyncio; asyncio.sleep(1)"""
        from tests.fixtures_sleep.sleep_import_asyncio import SleepImportAsyncioWorkflow

        program = SleepImportAsyncioWorkflow.workflow_ir()

        sleep_action = self._find_sleep_action(program)
        assert sleep_action is not None, "Expected @sleep action in IR"

        duration = self._get_duration_kwarg(sleep_action)
        assert duration is not None, "Expected duration kwarg"
        assert duration.value.HasField("literal"), "Expected literal value"
        assert duration.value.literal.int_value == 1

    def test_from_asyncio_import_sleep_pattern(self) -> None:
        """Test: from asyncio import sleep; sleep(2)"""
        from tests.fixtures_sleep.sleep_from_import import SleepFromImportWorkflow

        program = SleepFromImportWorkflow.workflow_ir()

        sleep_action = self._find_sleep_action(program)
        assert sleep_action is not None, "Expected @sleep action in IR"

        duration = self._get_duration_kwarg(sleep_action)
        assert duration is not None, "Expected duration kwarg"
        assert duration.value.HasField("literal"), "Expected literal value"
        assert duration.value.literal.int_value == 2

    def test_from_asyncio_import_sleep_as_alias_pattern(self) -> None:
        """Test: from asyncio import sleep as async_sleep; async_sleep(3)"""
        from tests.fixtures_sleep.sleep_aliased_import import SleepAliasedImportWorkflow

        program = SleepAliasedImportWorkflow.workflow_ir()

        sleep_action = self._find_sleep_action(program)
        assert sleep_action is not None, "Expected @sleep action in IR"

        duration = self._get_duration_kwarg(sleep_action)
        assert duration is not None, "Expected duration kwarg"
        assert duration.value.HasField("literal"), "Expected literal value"
        assert duration.value.literal.int_value == 3


class TestPolicyParsing:
    """Test that retry and timeout policies are parsed from run_action calls."""

    def _find_action_with_policies(
        self, program: ir.Program, action_name: str
    ) -> ir.ActionCall | None:
        """Find an action call by name, searching in all contexts."""
        for fn in program.functions:
            for stmt in fn.body.statements:
                # Direct action call
                if stmt.HasField("action_call"):
                    if stmt.action_call.action_name == action_name:
                        return stmt.action_call
                # Action in try body
                elif stmt.HasField("try_except"):
                    te = stmt.try_except
                    if te.try_body.HasField("call") and te.try_body.call.HasField("action"):
                        if te.try_body.call.action.action_name == action_name:
                            return te.try_body.call.action
        return None

    def test_timeout_policy_with_timedelta(self) -> None:
        """Test: self.run_action(action(), timeout=timedelta(seconds=2))"""
        from tests.fixtures_policy.integration_crash_recovery import CrashRecoveryWorkflow

        program = CrashRecoveryWorkflow.workflow_ir()

        # Find step_one which has timeout=timedelta(seconds=2)
        action = self._find_action_with_policies(program, "step_one")
        assert action is not None, "Expected @step_one action"
        assert len(action.policies) == 1, "Expected 1 policy"

        policy = action.policies[0]
        assert policy.HasField("timeout"), "Expected timeout policy"
        assert policy.timeout.timeout.seconds == 2

    def test_retry_policy_with_attempts(self) -> None:
        """Test: self.run_action(action(), retry=RetryPolicy(attempts=1))"""
        from tests.fixtures_policy.integration_exception_custom import ExceptionCustomWorkflow

        program = ExceptionCustomWorkflow.workflow_ir()

        # Find explode_custom which has retry=RetryPolicy(attempts=1)
        action = self._find_action_with_policies(program, "explode_custom")
        assert action is not None, "Expected @explode_custom action"
        assert len(action.policies) == 1, "Expected 1 policy"

        policy = action.policies[0]
        assert policy.HasField("retry"), "Expected retry policy"
        assert policy.retry.max_retries == 1

    def test_direct_action_call_no_policies(self) -> None:
        """Test: await action() - direct call without run_action wrapper."""

        # CrashRecoveryWorkflow uses run_action, so all have policies
        # Let's check a different workflow
        from tests.fixtures_policy.integration_exception_custom import ExceptionCustomWorkflow

        program = ExceptionCustomWorkflow.workflow_ir()

        # provide_value is called directly (not via run_action)
        action = self._find_action_with_policies(program, "provide_value")
        assert action is not None, "Expected @provide_value action"
        assert len(action.policies) == 0, "Direct action call should have no policies"


class TestAsyncioGatherDetection:
    """Test that asyncio.gather is detected and converted to parallel blocks."""

    def _find_parallel_block(self, program: ir.Program) -> ir.ParallelBlock | None:
        """Find a parallel block in the program."""
        for fn in program.functions:
            for stmt in fn.body.statements:
                if stmt.HasField("parallel_block"):
                    return stmt.parallel_block
        return None

    def _find_all_parallel_blocks(self, program: ir.Program) -> List[ir.ParallelBlock]:
        """Find all parallel blocks in the program."""
        blocks = []
        for fn in program.functions:
            for stmt in fn.body.statements:
                if stmt.HasField("parallel_block"):
                    blocks.append(stmt.parallel_block)
        return blocks

    def _get_action_names_from_parallel(self, block: ir.ParallelBlock) -> List[str]:
        """Extract action names from a parallel block."""
        names = []
        for call in block.calls:
            if call.HasField("action"):
                names.append(call.action.action_name)
        return names

    def test_gather_simple_two_actions(self) -> None:
        """Test: a, b = await asyncio.gather(action_a(), action_b())"""
        from tests.fixtures_gather.gather_simple import GatherSimpleWorkflow

        program = GatherSimpleWorkflow.workflow_ir()

        parallel = self._find_parallel_block(program)
        assert parallel is not None, "Expected parallel block from asyncio.gather"

        action_names = self._get_action_names_from_parallel(parallel)
        assert len(action_names) == 2, f"Expected 2 actions in parallel, got {len(action_names)}"
        assert "action_a" in action_names, "Expected action_a in parallel block"
        assert "action_b" in action_names, "Expected action_b in parallel block"

    def test_gather_with_args(self) -> None:
        """Test: asyncio.gather with actions that have arguments."""
        from tests.fixtures_gather.gather_with_args import GatherWithArgsWorkflow

        program = GatherWithArgsWorkflow.workflow_ir()

        parallel = self._find_parallel_block(program)
        assert parallel is not None, "Expected parallel block from asyncio.gather"

        action_names = self._get_action_names_from_parallel(parallel)
        assert "compute_square" in action_names, "Expected compute_square in parallel"
        assert "compute_cube" in action_names, "Expected compute_cube in parallel"

        # Check that kwargs are preserved
        for call in parallel.calls:
            if call.HasField("action"):
                action = call.action
                if action.action_name in ("compute_square", "compute_cube"):
                    assert len(action.kwargs) == 1, f"Expected 1 kwarg for {action.action_name}"
                    assert action.kwargs[0].name == "n", "Expected 'n' kwarg"

    def test_gather_to_single_variable(self) -> None:
        """Test: results = await asyncio.gather(a(), b(), c())"""
        from tests.fixtures_gather.gather_to_variable import GatherToVariableWorkflow

        program = GatherToVariableWorkflow.workflow_ir()

        parallel = self._find_parallel_block(program)
        assert parallel is not None, "Expected parallel block from asyncio.gather"

        # Check target variable is set
        assert parallel.target == "results", f"Expected target 'results', got '{parallel.target}'"

        action_names = self._get_action_names_from_parallel(parallel)
        assert len(action_names) == 3, f"Expected 3 actions, got {len(action_names)}"

    def test_gather_nested_fan_in(self) -> None:
        """Test: Fan-out with gather, then fan-in with another action."""
        from tests.fixtures_gather.gather_nested import GatherNestedWorkflow

        program = GatherNestedWorkflow.workflow_ir()

        # Should have a parallel block for the gather
        parallel = self._find_parallel_block(program)
        assert parallel is not None, "Expected parallel block from asyncio.gather"

        action_names = self._get_action_names_from_parallel(parallel)
        assert "fetch_a" in action_names, "Expected fetch_a in parallel"
        assert "fetch_b" in action_names, "Expected fetch_b in parallel"

        # Should also have the combine action after the parallel block
        combine_found = False
        for fn in program.functions:
            for stmt in fn.body.statements:
                if stmt.HasField("action_call"):
                    if stmt.action_call.action_name == "combine":
                        combine_found = True
        assert combine_found, "Expected combine action after parallel block"

    def test_gather_starred_list_comprehension(self) -> None:
        """Test: await asyncio.gather(*[action(x) for x in items])

        This common pattern should produce a SpreadAction IR node that represents
        parallel execution over a collection.
        """
        from tests.fixtures_gather.gather_listcomp import GatherListCompWorkflow

        program = GatherListCompWorkflow.workflow_ir()

        # Should produce a SpreadAction node for parallel iteration
        spread_found = False
        for fn in program.functions:
            for stmt in fn.body.statements:
                if stmt.HasField("spread_action"):
                    spread_found = True
                    spread = stmt.spread_action
                    # Check the loop variable
                    assert spread.loop_var == "item", f"Expected loop_var 'item', got '{spread.loop_var}'"
                    # Check action is process_item
                    assert spread.action.action_name == "process_item", (
                        f"Expected action 'process_item', got '{spread.action.action_name}'"
                    )
                    # Check target is set (for collecting results)
                    assert spread.target == "results", f"Expected target 'results', got '{spread.target}'"

        assert spread_found, "Expected SpreadAction node from gather(*[listcomp])"


class TestForLoopConversion:
    """Test for loop conversion to IR."""

    def _find_for_loop(self, program: ir.Program) -> ir.ForLoop | None:
        """Find a for loop in the program."""
        for fn in program.functions:
            for stmt in fn.body.statements:
                if stmt.HasField("for_loop"):
                    return stmt.for_loop
        return None

    def _find_implicit_function(self, program: ir.Program, prefix: str) -> ir.FunctionDef | None:
        """Find an implicit function by name prefix."""
        for fn in program.functions:
            if fn.name.startswith(prefix):
                return fn
        return None

    def test_simple_for_loop_structure(self) -> None:
        """Test: Simple for loop has correct structure."""
        from tests.fixtures_control_flow.for_simple import ForSimpleWorkflow

        program = ForSimpleWorkflow.workflow_ir()

        for_loop = self._find_for_loop(program)
        assert for_loop is not None, "Expected for_loop in IR"

        # Check loop variable
        assert "item" in for_loop.loop_vars, "Expected 'item' as loop variable"

        # Check iterable is present
        assert for_loop.HasField("iterable"), "Expected iterable expression"

    def test_for_loop_body_has_call(self) -> None:
        """Test: For loop body contains the action call."""
        from tests.fixtures_control_flow.for_simple import ForSimpleWorkflow

        program = ForSimpleWorkflow.workflow_ir()

        for_loop = self._find_for_loop(program)
        assert for_loop is not None, "Expected for_loop in IR"

        # Body should have a call
        assert for_loop.body.HasField("call"), "Expected call in for loop body"

    def test_multi_action_for_creates_implicit_function(self) -> None:
        """Test: Multi-action for loop body is wrapped in implicit function."""
        from tests.fixtures_control_flow.for_multi_action import ForMultiActionWorkflow

        program = ForMultiActionWorkflow.workflow_ir()

        # Should have an implicit function for the multi-action body
        implicit_fn = self._find_implicit_function(program, "__for_body")
        assert implicit_fn is not None, "Expected implicit function for multi-action for body"

        # The implicit function should have multiple statements
        assert len(implicit_fn.body.statements) >= 2, "Implicit function should have multiple statements"


class TestConditionalConversion:
    """Test if/elif/else conversion to IR."""

    def _find_conditional(self, program: ir.Program) -> ir.Conditional | None:
        """Find a conditional in the program."""
        for fn in program.functions:
            for stmt in fn.body.statements:
                if stmt.HasField("conditional"):
                    return stmt.conditional
        return None

    def _find_implicit_function(self, program: ir.Program, prefix: str) -> ir.FunctionDef | None:
        """Find an implicit function by name prefix."""
        for fn in program.functions:
            if fn.name.startswith(prefix):
                return fn
        return None

    def test_simple_if_else_structure(self) -> None:
        """Test: Simple if/else has correct structure."""
        from tests.fixtures_control_flow.if_simple import IfSimpleWorkflow

        program = IfSimpleWorkflow.workflow_ir()

        conditional = self._find_conditional(program)
        assert conditional is not None, "Expected conditional in IR"

        # Should have if_branch with condition
        assert conditional.HasField("if_branch"), "Expected if_branch"
        assert conditional.if_branch.HasField("condition"), "Expected condition expression"
        assert conditional.if_branch.HasField("body"), "Expected if_branch body"

        # Should have else_branch
        assert conditional.HasField("else_branch"), "Expected else_branch"

    def test_elif_chain_creates_branches(self) -> None:
        """Test: if/elif/elif/else creates proper branch structure."""
        from tests.fixtures_control_flow.if_elif_else import IfElifElseWorkflow

        program = IfElifElseWorkflow.workflow_ir()

        conditional = self._find_conditional(program)
        assert conditional is not None, "Expected conditional in IR"

        # Should have elif branches
        assert len(conditional.elif_branches) >= 2, "Expected at least 2 elif branches"

    def test_multi_action_branches_create_implicit_functions(self) -> None:
        """Test: Multi-action if/else branches are wrapped in implicit functions."""
        from tests.fixtures_control_flow.if_multi_action import IfMultiActionWorkflow

        program = IfMultiActionWorkflow.workflow_ir()

        # Should have implicit functions for the branches
        if_fn = self._find_implicit_function(program, "__if_then")
        else_fn = self._find_implicit_function(program, "__if_else")

        # At least one should exist (both branches have multi-action)
        assert if_fn is not None or else_fn is not None, (
            "Expected implicit function for multi-action branches"
        )


class TestTryExceptConversion:
    """Test try/except conversion to IR."""

    def _find_try_except(self, program: ir.Program) -> ir.TryExcept | None:
        """Find a try/except in the program."""
        for fn in program.functions:
            for stmt in fn.body.statements:
                if stmt.HasField("try_except"):
                    return stmt.try_except
        return None

    def _find_implicit_function(self, program: ir.Program, prefix: str) -> ir.FunctionDef | None:
        """Find an implicit function by name prefix."""
        for fn in program.functions:
            if fn.name.startswith(prefix):
                return fn
        return None

    def test_simple_try_except_structure(self) -> None:
        """Test: Simple try/except has correct structure."""
        from tests.fixtures_control_flow.try_simple import TrySimpleWorkflow

        program = TrySimpleWorkflow.workflow_ir()

        try_except = self._find_try_except(program)
        assert try_except is not None, "Expected try_except in IR"

        # Should have try body
        assert try_except.HasField("try_body"), "Expected try_body"

        # Should have at least one handler
        assert len(try_except.handlers) >= 1, "Expected at least one exception handler"

    def test_multi_action_try_creates_implicit_function(self) -> None:
        """Test: Multi-action try body is wrapped in implicit function."""
        from tests.fixtures_control_flow.try_multi_action import TryMultiActionWorkflow

        program = TryMultiActionWorkflow.workflow_ir()

        # Should have an implicit function for the multi-action try body
        implicit_fn = self._find_implicit_function(program, "__try_body")
        assert implicit_fn is not None, "Expected implicit function for multi-action try body"

    def test_multiple_exception_handlers(self) -> None:
        """Test: Multiple except clauses create multiple handlers."""
        from tests.fixtures_control_flow.try_multi_except import TryMultiExceptWorkflow

        program = TryMultiExceptWorkflow.workflow_ir()

        try_except = self._find_try_except(program)
        assert try_except is not None, "Expected try_except in IR"

        # Should have multiple handlers
        assert len(try_except.handlers) >= 3, (
            f"Expected at least 3 exception handlers, got {len(try_except.handlers)}"
        )

        # Check exception types are captured (exception_types is a repeated field)
        all_exception_types: List[str] = []
        for h in try_except.handlers:
            all_exception_types.extend(h.exception_types)
        assert "ValueError" in all_exception_types, "Expected ValueError handler"
        assert "TypeError" in all_exception_types, "Expected TypeError handler"


class TestActionCallExtraction:
    """Test action call detection and argument handling."""

    def _find_action_call(self, program: ir.Program, name: str) -> ir.ActionCall | None:
        """Find an action call by name."""
        for fn in program.functions:
            for stmt in fn.body.statements:
                if stmt.HasField("action_call"):
                    if stmt.action_call.action_name == name:
                        return stmt.action_call
        return None

    def _find_all_action_calls(self, program: ir.Program) -> List[ir.ActionCall]:
        """Find all action calls in the program."""
        calls = []
        for fn in program.functions:
            for stmt in fn.body.statements:
                if stmt.HasField("action_call"):
                    calls.append(stmt.action_call)
        return calls

    def test_action_with_kwargs(self) -> None:
        """Test: Action called with keyword arguments preserves kwargs."""
        from tests.fixtures_actions.action_kwargs import ActionKwargsWorkflow

        program = ActionKwargsWorkflow.workflow_ir()

        action = self._find_action_call(program, "greet_person")
        assert action is not None, "Expected greet_person action"

        # Check kwargs
        kwarg_names = [kw.name for kw in action.kwargs]
        assert "name" in kwarg_names, "Expected 'name' kwarg"
        assert "greeting" in kwarg_names, "Expected 'greeting' kwarg"

    def test_action_with_positional_args_converted_to_kwargs(self) -> None:
        """Test: Positional arguments are converted to kwargs using signature.

        The IR builder converts positional args to kwargs using the
        action's signature. This requires the action to be properly decorated.
        """
        from tests.fixtures_actions.action_positional_args import ActionPositionalArgsWorkflow

        program = ActionPositionalArgsWorkflow.workflow_ir()

        action = self._find_action_call(program, "add_numbers")
        assert action is not None, "Expected add_numbers action"

        # The IR builder should have 2 kwargs (from positional args)
        # They get converted using the action's signature
        assert len(action.kwargs) == 2, (
            f"Expected 2 kwargs from positional args, got {len(action.kwargs)}"
        )

        # Verify parameter names match the action signature (a, b)
        kwarg_names = [kw.name for kw in action.kwargs]
        assert "a" in kwarg_names, "Expected 'a' kwarg from signature"
        assert "b" in kwarg_names, "Expected 'b' kwarg from signature"

        # Verify the values are literals (10 and 20) with correct mapping
        for kw in action.kwargs:
            if kw.name == "a":
                assert kw.value.HasField("literal"), "Expected literal for 'a'"
                assert kw.value.literal.int_value == 10, "Expected a=10"
            elif kw.name == "b":
                assert kw.value.HasField("literal"), "Expected literal for 'b'"
                assert kw.value.literal.int_value == 20, "Expected b=20"

    def test_action_with_variable_references(self) -> None:
        """Test: Action arguments that are variable references."""
        from tests.fixtures_actions.action_variable_args import ActionVariableArgsWorkflow

        program = ActionVariableArgsWorkflow.workflow_ir()

        action = self._find_action_call(program, "multiply_by")
        assert action is not None, "Expected multiply_by action"

        # Check that kwargs reference variables
        for kw in action.kwargs:
            if kw.name == "value":
                assert kw.value.HasField("variable"), "Expected variable reference for 'value'"
                assert kw.value.variable.name == "base", "Expected reference to 'base'"
            elif kw.name == "factor":
                assert kw.value.HasField("variable"), "Expected variable reference for 'factor'"
                assert kw.value.variable.name == "factor", "Expected reference to 'factor'"

    def test_action_without_assignment(self) -> None:
        """Test: Action called without capturing return value."""
        from tests.fixtures_actions.action_no_assignment import ActionNoAssignmentWorkflow

        program = ActionNoAssignmentWorkflow.workflow_ir()

        # Find log_event calls - should exist without target
        calls = self._find_all_action_calls(program)
        log_calls = [c for c in calls if c.action_name == "log_event"]

        assert len(log_calls) >= 1, "Expected at least one log_event call"

        # At least one should have no target (side effect only)
        has_no_target = any(not c.target for c in log_calls)
        assert has_no_target, "Expected log_event call without target assignment"

    def test_action_target_variable_captured(self) -> None:
        """Test: Action result is assigned to correct target variable."""
        from tests.fixtures_actions.action_kwargs import ActionKwargsWorkflow

        program = ActionKwargsWorkflow.workflow_ir()

        action = self._find_action_call(program, "greet_person")
        assert action is not None, "Expected greet_person action"

        # Should have target
        assert action.target == "result", f"Expected target 'result', got '{action.target}'"

    def test_action_module_name_set(self) -> None:
        """Test: Action has module_name set for worker dispatch."""
        from tests.fixtures_actions.action_kwargs import ActionKwargsWorkflow

        program = ActionKwargsWorkflow.workflow_ir()

        action = self._find_action_call(program, "greet_person")
        assert action is not None, "Expected greet_person action"

        # Should have module name
        assert action.module_name, "Expected module_name to be set"
        assert "action_kwargs" in action.module_name, (
            f"Expected module name to contain 'action_kwargs', got '{action.module_name}'"
        )

    def test_action_with_mixed_positional_and_keyword_args(self) -> None:
        """Test: Mix of positional and keyword args are all converted to kwargs."""
        from tests.fixtures_actions.action_mixed_args import ActionMixedArgsWorkflow

        program = ActionMixedArgsWorkflow.workflow_ir()

        action = self._find_action_call(program, "compute_value")
        assert action is not None, "Expected compute_value action"

        # Should have 3 kwargs (2 from positional, 1 explicit kwarg)
        assert len(action.kwargs) == 3, (
            f"Expected 3 kwargs, got {len(action.kwargs)}"
        )

        # Verify all parameter names are present
        kwarg_names = [kw.name for kw in action.kwargs]
        assert "x" in kwarg_names, "Expected 'x' kwarg from positional arg"
        assert "y" in kwarg_names, "Expected 'y' kwarg from positional arg"
        assert "multiplier" in kwarg_names, "Expected 'multiplier' kwarg"

        # Verify values are correct
        for kw in action.kwargs:
            if kw.name == "x":
                assert kw.value.literal.int_value == 5, "Expected x=5"
            elif kw.name == "y":
                assert kw.value.literal.int_value == 10, "Expected y=10"
            elif kw.name == "multiplier":
                assert kw.value.literal.int_value == 2, "Expected multiplier=2"
