"""Integration test: parallel block and side-effect-only spread."""

import asyncio

from rappel import action, workflow
from rappel.workflow import Workflow

EVENTS: list[str] = []


@action
async def record_event(name: str) -> None:
    EVENTS.append(name)


@action
async def snapshot_events() -> list[str]:
    return sorted(EVENTS)


@action
async def format_events(events: list[str]) -> str:
    return ",".join(events)


@workflow
class ParallelBlockWorkflow(Workflow):
    async def run(self) -> str:
        await record_event(name="start")
        await asyncio.gather(
            record_event(name="alpha"),
            record_event(name="beta"),
        )
        await asyncio.gather(*[record_event(name=item) for item in ["gamma", "delta"]])
        events = await snapshot_events()
        return await format_events(events=events)
