"""Test fixture: Simple asyncio.gather with two actions."""

import asyncio

from rappel import action, workflow
from rappel.workflow import Workflow


@action
async def action_a() -> int:
    return 1


@action
async def action_b() -> int:
    return 2


@workflow
class GatherSimpleWorkflow(Workflow):
    """Simple gather of two actions."""

    async def run(self) -> int:
        a, b = await asyncio.gather(
            action_a(),
            action_b(),
        )
        return a + b
