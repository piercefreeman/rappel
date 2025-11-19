from typing import Any, Iterator, Protocol

Struct = Any
Value = Any

class _ProtoMessage(Protocol):
    def SerializeToString(self) -> bytes: ...
    def ParseFromString(self, data: bytes) -> None: ...
    def CopyFrom(self, other: _ProtoMessage) -> None: ...

class Ack(_ProtoMessage):
    def __init__(self, acked_delivery_id: int = ...) -> None: ...
    acked_delivery_id: int

class ActionDispatch(_ProtoMessage):
    def __init__(
        self,
        action_id: str = ...,
        instance_id: str = ...,
        sequence: int = ...,
        dispatch: WorkflowNodeDispatch | None = ...,
    ) -> None: ...
    action_id: str
    instance_id: str
    sequence: int
    dispatch: WorkflowNodeDispatch

class ActionResult(_ProtoMessage):
    def __init__(
        self,
        action_id: str = ...,
        success: bool = ...,
        payload: WorkflowArguments | None = ...,
        worker_start_ns: int = ...,
        worker_end_ns: int = ...,
    ) -> None: ...
    action_id: str
    success: bool
    payload: WorkflowArguments
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
    MESSAGE_KIND_WORKER_HELLO: int

class WorkerHello(_ProtoMessage):
    def __init__(self, worker_id: int = ...) -> None: ...
    worker_id: int

class WorkflowExceptionEdge(_ProtoMessage):
    def __init__(
        self,
        source_node_id: str = ...,
        exception_type: str = ...,
        exception_module: str = ...,
    ) -> None: ...
    source_node_id: str
    exception_type: str
    exception_module: str

class _ExceptionEdgeContainer(Protocol):
    def add(self) -> WorkflowExceptionEdge: ...
    def __iter__(self) -> Iterator[WorkflowExceptionEdge]: ...

class _WorkflowArgumentContainer(Protocol):
    def add(self) -> "WorkflowArgument": ...
    def __iter__(self) -> Iterator["WorkflowArgument"]: ...

class _WorkflowArgumentValueContainer(Protocol):
    def add(self) -> "WorkflowArgumentValue": ...
    def __iter__(self) -> Iterator["WorkflowArgumentValue"]: ...

class _WorkflowNodeContextContainer(Protocol):
    def add(self) -> "WorkflowNodeContext": ...
    def __iter__(self) -> Iterator["WorkflowNodeContext"]: ...

class WorkflowDagNode(_ProtoMessage):
    def __init__(
        self,
        id: str = ...,
        action: str = ...,
        kwargs: dict[str, str] | None = ...,
        depends_on: list[str] | None = ...,
        wait_for_sync: list[str] | None = ...,
        produces: list[str] | None = ...,
        module: str = ...,
        guard: str = ...,
        exception_edges: list[WorkflowExceptionEdge] | None = ...,
    ) -> None: ...
    id: str
    action: str
    kwargs: dict[str, str]
    depends_on: list[str]
    wait_for_sync: list[str]
    produces: list[str]
    module: str
    guard: str
    exception_edges: _ExceptionEdgeContainer

class WorkflowDagDefinition(_ProtoMessage):
    def __init__(
        self,
        concurrent: bool = ...,
        nodes: list[WorkflowDagNode] | None = ...,
        return_variable: str = ...,
    ) -> None: ...
    concurrent: bool
    nodes: list[WorkflowDagNode]
    return_variable: str

class WorkflowRegistration(_ProtoMessage):
    def __init__(
        self,
        workflow_name: str = ...,
        dag: WorkflowDagDefinition | None = ...,
        dag_hash: str = ...,
    ) -> None: ...
    workflow_name: str
    dag: WorkflowDagDefinition
    dag_hash: str

class WorkflowNodeContext(_ProtoMessage):
    def __init__(
        self,
        variable: str = ...,
        payload: WorkflowArguments | None = ...,
        workflow_node_id: str = ...,
    ) -> None: ...
    variable: str
    payload: WorkflowArguments
    workflow_node_id: str

class WorkflowArgumentValue(_ProtoMessage):
    def __init__(
        self,
        primitive: PrimitiveWorkflowArgument | None = ...,
        basemodel: BaseModelWorkflowArgument | None = ...,
        exception: WorkflowErrorValue | None = ...,
        list_value: WorkflowListArgument | None = ...,
        tuple_value: WorkflowTupleArgument | None = ...,
        dict_value: WorkflowDictArgument | None = ...,
    ) -> None: ...
    primitive: PrimitiveWorkflowArgument
    basemodel: BaseModelWorkflowArgument
    exception: WorkflowErrorValue
    list_value: WorkflowListArgument
    tuple_value: WorkflowTupleArgument
    dict_value: WorkflowDictArgument

class WorkflowArgument(_ProtoMessage):
    def __init__(
        self,
        key: str = ...,
        value: WorkflowArgumentValue | None = ...,
    ) -> None: ...
    key: str
    value: WorkflowArgumentValue

class WorkflowArguments(_ProtoMessage):
    def __init__(
        self,
        arguments: list[WorkflowArgument] | None = ...,
    ) -> None: ...
    arguments: _WorkflowArgumentContainer

class PrimitiveWorkflowArgument(_ProtoMessage):
    def __init__(
        self,
        string_value: str = ...,
        double_value: float = ...,
        int_value: int = ...,
        bool_value: bool = ...,
        null_value: int = ...,
    ) -> None: ...
    string_value: str
    double_value: float
    int_value: int
    bool_value: bool
    null_value: int

class BaseModelWorkflowArgument(_ProtoMessage):
    def __init__(self, module: str = ..., name: str = ..., data: Struct | None = ...) -> None: ...
    module: str
    name: str
    data: Struct

class WorkflowErrorValue(_ProtoMessage):
    def __init__(
        self,
        type: str = ...,
        module: str = ...,
        message: str = ...,
        traceback: str = ...,
    ) -> None: ...
    type: str
    module: str
    message: str
    traceback: str

class WorkflowListArgument(_ProtoMessage):
    def __init__(self, items: list[WorkflowArgumentValue] | None = ...) -> None: ...
    items: _WorkflowArgumentValueContainer

class WorkflowTupleArgument(_ProtoMessage):
    def __init__(self, items: list[WorkflowArgumentValue] | None = ...) -> None: ...
    items: _WorkflowArgumentValueContainer

class WorkflowDictArgument(_ProtoMessage):
    def __init__(self, entries: list[WorkflowArgument] | None = ...) -> None: ...
    entries: _WorkflowArgumentContainer

class WorkflowNodeDispatch(_ProtoMessage):
    def __init__(
        self,
        node: WorkflowDagNode | None = ...,
        workflow_input: WorkflowArguments | None = ...,
        context: list[WorkflowNodeContext] | None = ...,
    ) -> None: ...
    node: WorkflowDagNode
    workflow_input: WorkflowArguments
    context: _WorkflowNodeContextContainer

class RegisterWorkflowRequest(_ProtoMessage):
    def __init__(
        self,
        registration: WorkflowRegistration | None = ...,
    ) -> None: ...
    registration: WorkflowRegistration

class RegisterWorkflowResponse(_ProtoMessage):
    def __init__(self, workflow_version_id: str = ...) -> None: ...
    workflow_version_id: str

class WaitForInstanceRequest(_ProtoMessage):
    def __init__(
        self,
        poll_interval_secs: float = ...,
    ) -> None: ...
    poll_interval_secs: float

class WaitForInstanceResponse(_ProtoMessage):
    def __init__(self, payload: bytes = ...) -> None: ...
    payload: bytes
