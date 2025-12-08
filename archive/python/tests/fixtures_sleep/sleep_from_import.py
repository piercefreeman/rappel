"""Test fixture: from asyncio import sleep pattern."""

from asyncio import sleep

from rappel import action, workflow
from rappel.workflow import Workflow


@action
async def get_value_from_import() -> int:
    return 42


@workflow
class SleepFromImportWorkflow(Workflow):
    async def run(self) -> str:
        val = await get_value_from_import()
        await sleep(2)
        return "done:" + str(val)
