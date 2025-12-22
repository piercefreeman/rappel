"""Repro fixture: kw-only workflow args resolve to null in action request."""

from __future__ import annotations

from pydantic import BaseModel

from rappel import action, workflow
from rappel.workflow import Workflow


class LocationRequest(BaseModel):
    latitude: float | None = None
    longitude: float | None = None


@action
async def resolve_location(request: LocationRequest) -> float:
    return request.latitude or 0.0


@workflow
class KwOnlyActionRequestWorkflow(Workflow):
    async def run(
        self,
        *,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> None:
        await self.run_action(
            resolve_location(
                LocationRequest(
                    latitude=latitude,
                    longitude=longitude,
                )
            )
        )
