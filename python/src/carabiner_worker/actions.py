from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, overload

from proto import messages_pb2 as pb2

from .registry import AsyncAction, registry
from .serialization import (
    arguments_to_kwargs,
    build_arguments_from_kwargs,
    dumps,
    loads,
)

TAsync = TypeVar("TAsync", bound=AsyncAction)


@dataclass
class ActionCall:
    module: str
    action: str
    kwargs: dict[str, Any]


@dataclass
class ActionResultPayload:
    result: Any | None
    error: dict[str, str] | None


def serialize_action_call(module: str, action: str, /, **kwargs: Any) -> bytes:
    """Serialize an action name and keyword arguments into bytes."""
    if not isinstance(module, str) or not module:
        raise ValueError("action module must be a non-empty string")
    if not isinstance(action, str) or not action:
        raise ValueError("action name must be a non-empty string")
    invocation = pb2.WorkflowInvocation(module=module, function_name=action)
    invocation.kwargs.CopyFrom(build_arguments_from_kwargs(kwargs))
    return invocation.SerializeToString()


def deserialize_action_call(payload: bytes) -> ActionCall:
    """Deserialize a payload into an action invocation."""
    invocation = pb2.WorkflowInvocation()
    invocation.ParseFromString(payload)
    if not invocation.module:
        raise ValueError("payload missing module name")
    if not invocation.function_name:
        raise ValueError("payload missing function name")
    kwargs = arguments_to_kwargs(invocation.kwargs)
    return ActionCall(module=invocation.module, action=invocation.function_name, kwargs=kwargs)


def serialize_result_payload(value: Any) -> pb2.WorkflowArguments:
    """Serialize a successful action result."""
    arguments = pb2.WorkflowArguments()
    entry = arguments.arguments.add()
    entry.key = "result"
    entry.value.CopyFrom(dumps(value))
    return arguments


def serialize_error_payload(_action: str, exc: BaseException) -> pb2.WorkflowArguments:
    """Serialize an error raised during action execution."""
    arguments = pb2.WorkflowArguments()
    entry = arguments.arguments.add()
    entry.key = "error"
    entry.value.CopyFrom(dumps(exc))
    return arguments


def deserialize_result_payload(payload: pb2.WorkflowArguments | None) -> ActionResultPayload:
    """Deserialize WorkflowArguments produced by serialize_result_payload/error."""
    if payload is None:
        return ActionResultPayload(result=None, error=None)
    values = {entry.key: entry.value for entry in payload.arguments}
    if "error" in values:
        error_value = values["error"]
        data = loads(error_value)
        if not isinstance(data, dict):
            raise ValueError("error payload must deserialize to a mapping")
        return ActionResultPayload(result=None, error=data)
    result_value = values.get("result")
    if result_value is None:
        raise ValueError("result payload missing 'result' field")
    return ActionResultPayload(result=loads(result_value), error=None)


@overload
def action(func: TAsync, /) -> TAsync: ...


@overload
def action(*, name: Optional[str] = None) -> Callable[[TAsync], TAsync]: ...


def action(
    func: Optional[TAsync] = None, *, name: Optional[str] = None
) -> Callable[[TAsync], TAsync] | TAsync:
    """Decorator for registering async actions."""

    def decorator(target: TAsync) -> TAsync:
        if not inspect.iscoroutinefunction(target):
            raise TypeError(f"action '{target.__name__}' must be defined with 'async def'")
        action_name = name or target.__name__
        registry.register(action_name, target)
        target.__carabiner_action_name__ = action_name
        target.__carabiner_action_module__ = target.__module__
        return target

    if func is not None:
        return decorator(func)
    return decorator


class ActionRunner:
    """Executes registered actions for incoming invocations."""

    def __init__(self) -> None:
        self._loaded_modules: set[str] = set()

    def _ensure_module_loaded(self, module_name: str) -> None:
        if module_name in self._loaded_modules:
            return
        logging.info("Importing user module %s", module_name)
        importlib.import_module(module_name)
        self._loaded_modules.add(module_name)
        names = registry.names()
        summary = ", ".join(names) if names else "<none>"
        logging.info("Registered %s actions: %s", len(names), summary)

    async def run_serialized(self, payload: bytes) -> tuple[ActionCall, Any]:
        """Deserialize a payload and execute the referenced action."""
        invocation_message = pb2.WorkflowInvocation()
        invocation_message.ParseFromString(payload)
        return await self.run_invocation(invocation_message)

    async def run_invocation(self, message: pb2.WorkflowInvocation) -> tuple[ActionCall, Any]:
        invocation = ActionCall(
            module=message.module,
            action=message.function_name,
            kwargs=arguments_to_kwargs(message.kwargs),
        )
        self._ensure_module_loaded(invocation.module)
        handler = registry.get(invocation.action)
        if handler is None:
            raise RuntimeError(f"action '{invocation.action}' is not registered")
        result = handler(**invocation.kwargs)
        if not asyncio.iscoroutine(result):
            raise RuntimeError(
                f"action '{invocation.action}' did not return a coroutine; "
                "ensure it is defined with 'async def'"
            )
        return invocation, await result
