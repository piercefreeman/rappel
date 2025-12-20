"""Repro fixture for action request constructor evaluation."""

from rappel import action, workflow
from rappel.workflow import Workflow


class CheckRequest:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id


@action
async def check_enabled(request: CheckRequest) -> str:
    return request.user_id


class PredictCommon(Workflow):
    async def run_internal(self, user_id: str) -> None:
        await self.run_action(check_enabled(CheckRequest(user_id=user_id)))


@workflow
class PredictWorkflow(PredictCommon):
    async def run(self, user_id: str) -> None:
        await self.run_internal(user_id)
