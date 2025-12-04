"""Tests for IR builder functionality."""

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
