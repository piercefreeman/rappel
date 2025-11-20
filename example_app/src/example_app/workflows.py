"""Workflow definitions used by the sample FastAPI app."""

from __future__ import annotations

import asyncio
from pydantic import BaseModel, Field
from rappel import Workflow, action, workflow


class ComputationResult(BaseModel):
    """Aggregate result that is shown on the frontend template."""

    input_number: int
    factorial: int
    fibonacci: int
    summary: str


class ComputationRequest(BaseModel):
    number: int = Field(ge=1, le=10, description="Number to feed into the workflow")


@action
async def compute_factorial(n: int) -> int:
    total = 1
    for value in range(2, n + 1):
        total *= value
        await asyncio.sleep(0)
    return total


@action
async def compute_fibonacci(n: int) -> int:
    previous, current = 0, 1
    for _ in range(n):
        previous, current = current, previous + current
        await asyncio.sleep(0)
    return previous


@action
async def summarize_outcome(
    *,
    input_number: int,
    factorial_value: int,
    fibonacci_value: int,
) -> ComputationResult:
    if factorial_value > 5_000:
        summary = f"{input_number}! is massive compared to Fib({input_number})={fibonacci_value}"
    elif factorial_value > 100:
        summary = f"{input_number}! is larger, but Fibonacci is {fibonacci_value}"
    else:
        summary = f"{input_number}! ({factorial_value}) stays tame next to Fibonacci={fibonacci_value}"
    await asyncio.sleep(0)
    return ComputationResult(
        input_number=input_number,
        factorial=factorial_value,
        fibonacci=fibonacci_value,
        summary=summary,
    )


@workflow
class ExampleMathWorkflow(Workflow):
    """Workflow that fans out across simple math actions."""

    def __init__(self, number: int) -> None:
        self.number = number

    async def run(self) -> ComputationResult:
        factorial_value, fib_value = await asyncio.gather(
            compute_factorial(self.number),
            compute_fibonacci(self.number),
        )
        result = await summarize_outcome(
            input_number=self.number,
            factorial_value=factorial_value,
            fibonacci_value=fib_value,
        )
        return result
