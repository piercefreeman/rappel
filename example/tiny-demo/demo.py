# /// script
# dependencies = [
#   "pytest",
#   "pytest-asyncio",
#   "waymark",
# ]
# ///

import asyncio
import sys
from dataclasses import dataclass
from typing import Annotated

import pytest
from waymark import Depend, Workflow, action, workflow
from waymark.workflow import workflow_registry

workflow_registry._workflows.clear()


@dataclass
class User:
    id: str
    email: str
    active: bool


@dataclass
class EmailResult:
    to: str
    subject: str
    success: bool


async def get_mock_db():
    return {
        "user1": User(id="user1", email="alice@example.com", active=True),
        "user2": User(id="user2", email="bob@example.com", active=False),
        "user3": User(id="user3", email="carol@example.com", active=True),
    }


async def get_mock_email_client():
    return "email_client"


@action
async def fetch_users(
    user_ids: list[str],
    db: Annotated[dict, Depend(get_mock_db)],
) -> list[User]:
    return [db[uid] for uid in user_ids if uid in db]


@action
async def send_email(
    to: str,
    subject: str,
    emailer: Annotated[str, Depend(get_mock_email_client)],
) -> EmailResult:
    return EmailResult(to=to, subject=subject, success=True)


@workflow
class WelcomeEmailWorkflow(Workflow):
    async def run(self, user_ids: list[str]) -> dict:
        """Send welcome emails to active users"""

        users = await fetch_users(user_ids)
        active_users = [user for user in users if user.active]

        results = await asyncio.gather(
            *[send_email(to=user.email, subject="Welcome") for user in active_users],
            return_exceptions=True,
        )

        return {
            "total_users": len(users),
            "active_users": len(active_users),
            "emails_sent": len(results),
        }


@pytest.mark.asyncio
async def test_welcome_email_workflow():
    result = await WelcomeEmailWorkflow().run(user_ids=["user1", "user2", "user3"])
    assert result["total_users"] == 3
    assert result["active_users"] == 2
    assert result["emails_sent"] == 2


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
