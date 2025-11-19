from __future__ import annotations

import pytest

from carabiner_worker.actions import action


def test_action_decorator_stores_timeout_and_retry_metadata() -> None:
    @action(name="custom", timeout_seconds=10, max_retries=5)
    async def sample() -> None:  # pragma: no cover - metadata only
        raise NotImplementedError

    assert sample.__carabiner_action_name__ == "custom"
    assert sample.__carabiner_timeout_seconds__ == 10
    assert sample.__carabiner_max_retries__ == 5


def test_action_decorator_rejects_negative_values() -> None:
    with pytest.raises(ValueError):

        @action(timeout_seconds=-1)
        async def _bad_timeout() -> None:  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(ValueError):

        @action(max_retries=-2)
        async def _bad_retry() -> None:  # pragma: no cover
            raise NotImplementedError
