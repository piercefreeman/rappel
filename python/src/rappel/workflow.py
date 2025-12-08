"""
Workflow base class and decorator.

Workflows define the high-level logic of a durable computation.
They are composed of action calls which are executed by action workers.
"""

from typing import Any, Callable, TypeVar

T = TypeVar("T")

# Global registry of workflow classes
_workflow_registry: dict[str, type["Workflow"]] = {}


class Workflow:
    """
    Base class for durable workflows.

    Subclass this and implement the `run` method with your workflow logic.
    Use @action decorated functions for any I/O or side effects.

    Example:
        @workflow
        class MyWorkflow(Workflow):
            async def run(self, user_id: str) -> dict:
                user = await fetch_user(user_id)
                result = await process_user(user)
                return result
    """

    @classmethod
    def workflow_name(cls) -> str:
        """Get the registered name of this workflow."""
        return cls.__name__

    @classmethod
    def module_name(cls) -> str:
        """Get the module containing this workflow."""
        return cls.__module__

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the workflow logic.

        Override this method in your workflow subclass.
        Use @action decorated functions for durable operations.
        """
        raise NotImplementedError("Workflow subclasses must implement run()")


def workflow(cls: type[T]) -> type[T]:
    """
    Decorator to register a workflow class.

    Usage:
        @workflow
        class MyWorkflow(Workflow):
            async def run(self, data: dict) -> dict:
                ...
    """
    if not issubclass(cls, Workflow):
        raise TypeError(f"{cls.__name__} must be a subclass of Workflow")

    # Register the workflow
    _workflow_registry[cls.__name__] = cls
    return cls


def get_workflow_registry() -> dict[str, type[Workflow]]:
    """Get the global workflow registry."""
    return _workflow_registry


def get_workflow(name: str) -> type[Workflow] | None:
    """Get a workflow class by name."""
    return _workflow_registry.get(name)
