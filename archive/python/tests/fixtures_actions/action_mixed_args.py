"""Test fixture: Action with mixed positional and keyword arguments."""

from rappel import action, workflow
from rappel.workflow import Workflow


@action
async def compute_value(x: int, y: int, multiplier: int = 1) -> int:
    return (x + y) * multiplier


@workflow
class ActionMixedArgsWorkflow(Workflow):
    """Action with mix of positional and keyword args."""

    async def run(self) -> int:
        # First two positional, third as kwarg
        result = await compute_value(5, 10, multiplier=2)
        return result
