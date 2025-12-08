import asyncio
from typing import Annotated, Any, AsyncIterator

from rappel.dependencies import Depend, provide_dependencies


def test_provide_dependencies_resolves_regular_values() -> None:
    def dependency() -> str:
        return "dependent"

    async def target(value: Annotated[str, Depend(dependency)]) -> str:
        return value

    async def run() -> str:
        async with provide_dependencies(target) as kwargs:
            return await target(**kwargs)

    result = asyncio.run(run())
    assert result == "dependent"


def test_provide_dependencies_passes_kwargs_to_dependencies() -> None:
    calls: list[str] = []

    def dependency(prefix: str) -> str:
        calls.append(prefix)
        return f"{prefix}-dep"

    dependency_marker = Depend(dependency)

    async def target(prefix: str, value: Any = dependency_marker) -> str:
        return str(value)

    async def run() -> str:
        async with provide_dependencies(target, {"prefix": "root"}) as kwargs:
            return await target(**kwargs)

    result = asyncio.run(run())
    assert result == "root-dep"
    assert calls == ["root"]


def test_provide_dependencies_handles_async_generator_dependency() -> None:
    events: list[str] = []

    async def dependency() -> AsyncIterator[str]:
        events.append("enter")
        try:
            yield "resource"
        finally:
            events.append("exit")

    async def target(resource: Annotated[str, Depend(dependency)]) -> str:
        return resource

    async def run() -> str:
        async with provide_dependencies(target) as kwargs:
            return await target(**kwargs)

    result = asyncio.run(run())
    assert result == "resource"
    assert events == ["enter", "exit"]


def test_provide_dependencies_supports_recursive_dependencies() -> None:
    def base() -> str:
        return "base"

    def layer_one(base_value: Annotated[str, Depend(base)]) -> str:
        return f"one-{base_value}"

    async def target(final: Annotated[str, Depend(layer_one)]) -> str:
        return final

    async def run() -> str:
        async with provide_dependencies(target) as kwargs:
            return await target(**kwargs)

    result = asyncio.run(run())
    assert result == "one-base"
