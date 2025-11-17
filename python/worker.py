"""Python worker that executes user-defined carabiner actions."""

from __future__ import annotations

import asyncio
import importlib
import logging
import struct
import sys
import time
from typing import Any, BinaryIO

from carabiner_worker import registry as action_registry
from carabiner_worker.serialization import (
    ActionCall,
    deserialize_action_call,
    serialize_error_payload,
    serialize_result_payload,
)
from proto.messages_pb2 import Ack, ActionDispatch, ActionResult, Envelope, MessageKind

logging.basicConfig(
    level=logging.INFO, format="[worker] %(message)s", stream=sys.stderr
)

_HEADER = struct.Struct("<I")


def _read_frame(stream: BinaryIO) -> Envelope | None:
    header = stream.read(_HEADER.size)
    if not header:
        return None
    (length,) = _HEADER.unpack(header)
    payload = stream.read(length)
    message = Envelope()
    message.ParseFromString(payload)
    return message


def _write_frame(stream: BinaryIO, envelope: Envelope) -> None:
    payload = envelope.SerializeToString()
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _send_ack(stream: BinaryIO, delivery_id: int, partition_id: int) -> None:
    ack = Ack(acked_delivery_id=delivery_id)
    envelope = Envelope(
        delivery_id=delivery_id,
        partition_id=partition_id,
        kind=MessageKind.MESSAGE_KIND_ACK,
        payload=ack.SerializeToString(),
    )
    _write_frame(stream, envelope)


class UserActionExecutor:
    """Ensures user action modules are imported before invocation."""

    def __init__(self) -> None:
        self._loaded_modules: set[str] = set()

    def _ensure_module_loaded(self, module_name: str) -> None:
        if module_name in self._loaded_modules:
            return
        logging.info("Importing user module %s", module_name)
        importlib.import_module(module_name)
        self._loaded_modules.add(module_name)
        names = action_registry.names()
        summary = ", ".join(names) if names else "<none>"
        logging.info("Registered %s actions: %s", len(names), summary)

    def invoke(self, invocation: ActionCall) -> Any:
        self._ensure_module_loaded(invocation.module)
        handler = action_registry.get(invocation.action)
        if handler is None:
            raise RuntimeError(f"action '{invocation.action}' is not registered")
        result = handler(**invocation.kwargs)
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        raise RuntimeError(
            f"action '{invocation.action}' did not return a coroutine; "
            "ensure it is defined with 'async def'"
        )


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        logging.warning("Ignoring unexpected CLI arguments: %s", " ".join(argv))
    logging.info("Python worker booted")
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    executor = UserActionExecutor()

    while True:
        envelope = _read_frame(stdin)
        if envelope is None:
            logging.info("stdin closed - exiting")
            break

        kind = envelope.kind
        partition = envelope.partition_id

        if kind == MessageKind.MESSAGE_KIND_ACTION_DISPATCH:
            _send_ack(stdout, envelope.delivery_id, partition)

            dispatch = ActionDispatch()
            dispatch.ParseFromString(envelope.payload)

            worker_start = time.perf_counter_ns()
            payload = dispatch.payload
            action_name = "unknown"
            success = True
            try:
                invocation = deserialize_action_call(payload)
                action_name = invocation.action
                result = executor.invoke(invocation)
                response_payload = serialize_result_payload(result)
            except Exception as exc:  # noqa: BLE001 - propagate errors back to rust
                success = False
                response_payload = serialize_error_payload(action_name, exc)
                logging.exception(
                    "Action %s failed for action_id=%s sequence=%s",
                    action_name,
                    dispatch.action_id,
                    dispatch.sequence,
                )
            worker_end = time.perf_counter_ns()

            response = ActionResult(
                action_id=dispatch.action_id,
                success=success,
                payload=response_payload,
                worker_start_ns=worker_start,
                worker_end_ns=worker_end,
            )
            response_envelope = Envelope(
                delivery_id=envelope.delivery_id,
                partition_id=partition,
                kind=MessageKind.MESSAGE_KIND_ACTION_RESULT,
                payload=response.SerializeToString(),
            )
            _write_frame(stdout, response_envelope)
            logging.debug(
                "Handled action=%s seq=%s success=%s",
                action_name,
                dispatch.sequence,
                success,
            )
        elif kind == MessageKind.MESSAGE_KIND_HEARTBEAT:
            logging.debug("Received heartbeat delivery=%s", envelope.delivery_id)
            _send_ack(stdout, envelope.delivery_id, partition)
        else:
            logging.warning("Unhandled message kind: %s", kind)
            _send_ack(stdout, envelope.delivery_id, partition)


if __name__ == "__main__":
    main()
