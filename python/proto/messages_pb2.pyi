from typing import Protocol

class _ProtoMessage(Protocol):
    def SerializeToString(self) -> bytes: ...
    def ParseFromString(self, data: bytes) -> None: ...

class Ack(_ProtoMessage):
    def __init__(self, acked_delivery_id: int = ...) -> None: ...
    acked_delivery_id: int

class ActionDispatch(_ProtoMessage):
    def __init__(
        self,
        action_id: int = ...,
        instance_id: int = ...,
        sequence: int = ...,
        payload: bytes = ...,
    ) -> None: ...
    action_id: int
    instance_id: int
    sequence: int
    payload: bytes

class ActionResult(_ProtoMessage):
    def __init__(
        self,
        action_id: int = ...,
        success: bool = ...,
        payload: bytes = ...,
        worker_start_ns: int = ...,
        worker_end_ns: int = ...,
    ) -> None: ...
    action_id: int
    success: bool
    payload: bytes
    worker_start_ns: int
    worker_end_ns: int

class Envelope(_ProtoMessage):
    def __init__(
        self,
        delivery_id: int = ...,
        partition_id: int = ...,
        kind: int = ...,
        payload: bytes = ...,
    ) -> None: ...
    delivery_id: int
    partition_id: int
    kind: int
    payload: bytes

class MessageKind:
    MESSAGE_KIND_UNSPECIFIED: int
    MESSAGE_KIND_ACTION_DISPATCH: int
    MESSAGE_KIND_ACTION_RESULT: int
    MESSAGE_KIND_ACK: int
    MESSAGE_KIND_HEARTBEAT: int
