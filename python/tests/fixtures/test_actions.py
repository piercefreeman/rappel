"""Test actions for integration tests."""

from rappel.actions import action


@action
async def greet(name: str) -> str:
    """Simple greeting action for testing."""
    return f"Hello, {name}!"


@action
async def add(a: int, b: int) -> int:
    """Simple addition action for testing."""
    return a + b


@action
async def echo(message: str) -> str:
    """Echo action that returns the input."""
    return message
