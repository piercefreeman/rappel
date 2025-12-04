"""Test fixture: import asyncio; asyncio.sleep() pattern."""

import asyncio

from rappel import action, workflow
from rappel.workflow import Workflow


@action
async def get_value_asyncio_import() -> int:
    return 42


@workflow
class SleepImportAsyncioWorkflow(Workflow):
    async def run(self) -> str:
        val = await get_value_asyncio_import()
        await asyncio.sleep(1)
        return "done:" + str(val)
