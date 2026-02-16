import os

import pytest

from example_app.workflows import (
    EarlyReturnLoopWorkflow,
    ParallelMathWorkflow,
    RetryCounterWorkflow,
    TimeoutProbeWorkflow,
    WhileLoopWorkflow,
)


def _enable_real_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("WAYMARK_RUN_REAL_CLUSTER") == "1":
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)


def _require_real_cluster() -> None:
    if os.environ.get("WAYMARK_RUN_REAL_CLUSTER") != "1":
        pytest.skip("requires WAYMARK_RUN_REAL_CLUSTER=1")


@pytest.mark.asyncio
async def test_run_task_endpoint_executes_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_real_cluster(monkeypatch)

    workflow = ParallelMathWorkflow()
    result = await workflow.run(number=5)

    assert result.factorial == 120
    assert result.fibonacci == 5
    assert result.summary == "5! is larger, but Fibonacci is 5"


@pytest.mark.asyncio
async def test_early_return_loop_workflow_with_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the early return + loop workflow when session exists (should execute loop)."""
    _enable_real_cluster(monkeypatch)

    workflow = EarlyReturnLoopWorkflow()
    result = await workflow.run(input_text="apple, banana, cherry")

    assert result.had_session is True
    assert result.processed_count == 3
    assert result.all_items == ["apple", "banana", "cherry"]


@pytest.mark.asyncio
async def test_early_return_loop_workflow_early_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the early return + loop workflow when no session (should return early)."""
    _enable_real_cluster(monkeypatch)

    workflow = EarlyReturnLoopWorkflow()
    result = await workflow.run(input_text="no_session:test")

    assert result.had_session is False
    assert result.processed_count == 0
    assert result.all_items == []


@pytest.mark.asyncio
async def test_while_loop_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the while loop workflow executes until the limit."""
    _enable_real_cluster(monkeypatch)

    workflow = WhileLoopWorkflow()
    result = await workflow.run(limit=4)

    assert result.limit == 4
    assert result.final == 4
    assert result.iterations == 4


@pytest.mark.asyncio
async def test_retry_counter_workflow_eventual_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry workflow should succeed when threshold is within max attempts."""
    _require_real_cluster()
    _enable_real_cluster(monkeypatch)

    workflow = RetryCounterWorkflow()
    result = await workflow.run(
        succeed_on_attempt=3,
        max_attempts=4,
        counter_slot=901,
    )

    assert result.succeeded is True
    assert result.final_attempt == 3
    assert result.succeed_on_attempt == 3
    assert result.max_attempts == 4


@pytest.mark.asyncio
async def test_retry_counter_workflow_eventual_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry workflow should fail when threshold exceeds max attempts."""
    _require_real_cluster()
    _enable_real_cluster(monkeypatch)

    workflow = RetryCounterWorkflow()
    result = await workflow.run(
        succeed_on_attempt=5,
        max_attempts=3,
        counter_slot=902,
    )

    assert result.succeeded is False
    assert result.final_attempt == 3
    assert result.succeed_on_attempt == 5
    assert result.max_attempts == 3


@pytest.mark.skip(reason="TODO: timeout exceptions not retrying as expected")
@pytest.mark.asyncio
async def test_timeout_probe_workflow_eventual_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout probe should always fail with timeout after configured attempts."""
    _require_real_cluster()
    _enable_real_cluster(monkeypatch)

    workflow = TimeoutProbeWorkflow()
    result = await workflow.run(
        max_attempts=3,
        counter_slot=903,
    )

    assert result.timed_out is True
    assert result.final_attempt == 3
    assert result.timeout_seconds == 1
    assert result.max_attempts == 3


@pytest.mark.skip(reason="TODO: timeout exceptions not retrying as expected")
@pytest.mark.asyncio
async def test_timeout_probe_workflow_eventual_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout probe should honor lower max attempts."""
    _require_real_cluster()
    _enable_real_cluster(monkeypatch)

    workflow = TimeoutProbeWorkflow()
    result = await workflow.run(
        max_attempts=2,
        counter_slot=904,
    )

    assert result.timed_out is True
    assert result.final_attempt == 2
    assert result.timeout_seconds == 1
    assert result.max_attempts == 2
