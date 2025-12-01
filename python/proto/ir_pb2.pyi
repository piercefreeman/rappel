from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SourceLocation(_message.Message):
    __slots__ = ("lineno", "col_offset", "end_lineno", "end_col_offset")
    LINENO_FIELD_NUMBER: _ClassVar[int]
    COL_OFFSET_FIELD_NUMBER: _ClassVar[int]
    END_LINENO_FIELD_NUMBER: _ClassVar[int]
    END_COL_OFFSET_FIELD_NUMBER: _ClassVar[int]
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int
    def __init__(self, lineno: _Optional[int] = ..., col_offset: _Optional[int] = ..., end_lineno: _Optional[int] = ..., end_col_offset: _Optional[int] = ...) -> None: ...

class BackoffConfig(_message.Message):
    __slots__ = ("kind", "base_delay_ms", "multiplier")
    class Kind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        KIND_UNSPECIFIED: _ClassVar[BackoffConfig.Kind]
        KIND_LINEAR: _ClassVar[BackoffConfig.Kind]
        KIND_EXPONENTIAL: _ClassVar[BackoffConfig.Kind]
    KIND_UNSPECIFIED: BackoffConfig.Kind
    KIND_LINEAR: BackoffConfig.Kind
    KIND_EXPONENTIAL: BackoffConfig.Kind
    KIND_FIELD_NUMBER: _ClassVar[int]
    BASE_DELAY_MS_FIELD_NUMBER: _ClassVar[int]
    MULTIPLIER_FIELD_NUMBER: _ClassVar[int]
    kind: BackoffConfig.Kind
    base_delay_ms: int
    multiplier: float
    def __init__(self, kind: _Optional[_Union[BackoffConfig.Kind, str]] = ..., base_delay_ms: _Optional[int] = ..., multiplier: _Optional[float] = ...) -> None: ...

class RunActionConfig(_message.Message):
    __slots__ = ("timeout_seconds", "max_retries", "backoff")
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    MAX_RETRIES_FIELD_NUMBER: _ClassVar[int]
    BACKOFF_FIELD_NUMBER: _ClassVar[int]
    timeout_seconds: int
    max_retries: int
    backoff: BackoffConfig
    def __init__(self, timeout_seconds: _Optional[int] = ..., max_retries: _Optional[int] = ..., backoff: _Optional[_Union[BackoffConfig, _Mapping]] = ...) -> None: ...

class ActionCall(_message.Message):
    __slots__ = ("action", "module", "kwargs", "target", "config", "location")
    class KwargsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ACTION_FIELD_NUMBER: _ClassVar[int]
    MODULE_FIELD_NUMBER: _ClassVar[int]
    KWARGS_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    action: str
    module: str
    kwargs: _containers.ScalarMap[str, str]
    target: str
    config: RunActionConfig
    location: SourceLocation
    def __init__(self, action: _Optional[str] = ..., module: _Optional[str] = ..., kwargs: _Optional[_Mapping[str, str]] = ..., target: _Optional[str] = ..., config: _Optional[_Union[RunActionConfig, _Mapping]] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class SubgraphCall(_message.Message):
    __slots__ = ("method_name", "kwargs", "target", "location")
    class KwargsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    METHOD_NAME_FIELD_NUMBER: _ClassVar[int]
    KWARGS_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    method_name: str
    kwargs: _containers.ScalarMap[str, str]
    target: str
    location: SourceLocation
    def __init__(self, method_name: _Optional[str] = ..., kwargs: _Optional[_Mapping[str, str]] = ..., target: _Optional[str] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class GatherCall(_message.Message):
    __slots__ = ("action", "subgraph")
    ACTION_FIELD_NUMBER: _ClassVar[int]
    SUBGRAPH_FIELD_NUMBER: _ClassVar[int]
    action: ActionCall
    subgraph: SubgraphCall
    def __init__(self, action: _Optional[_Union[ActionCall, _Mapping]] = ..., subgraph: _Optional[_Union[SubgraphCall, _Mapping]] = ...) -> None: ...

class Gather(_message.Message):
    __slots__ = ("calls", "target", "location")
    CALLS_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    calls: _containers.RepeatedCompositeFieldContainer[GatherCall]
    target: str
    location: SourceLocation
    def __init__(self, calls: _Optional[_Iterable[_Union[GatherCall, _Mapping]]] = ..., target: _Optional[str] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class PythonBlock(_message.Message):
    __slots__ = ("code", "imports", "definitions", "inputs", "outputs", "location")
    CODE_FIELD_NUMBER: _ClassVar[int]
    IMPORTS_FIELD_NUMBER: _ClassVar[int]
    DEFINITIONS_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    code: str
    imports: _containers.RepeatedScalarFieldContainer[str]
    definitions: _containers.RepeatedScalarFieldContainer[str]
    inputs: _containers.RepeatedScalarFieldContainer[str]
    outputs: _containers.RepeatedScalarFieldContainer[str]
    location: SourceLocation
    def __init__(self, code: _Optional[str] = ..., imports: _Optional[_Iterable[str]] = ..., definitions: _Optional[_Iterable[str]] = ..., inputs: _Optional[_Iterable[str]] = ..., outputs: _Optional[_Iterable[str]] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class Loop(_message.Message):
    __slots__ = ("iterator_expr", "loop_var", "accumulator", "body", "location")
    ITERATOR_EXPR_FIELD_NUMBER: _ClassVar[int]
    LOOP_VAR_FIELD_NUMBER: _ClassVar[int]
    ACCUMULATOR_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    iterator_expr: str
    loop_var: str
    accumulator: str
    body: _containers.RepeatedCompositeFieldContainer[Statement]
    location: SourceLocation
    def __init__(self, iterator_expr: _Optional[str] = ..., loop_var: _Optional[str] = ..., accumulator: _Optional[str] = ..., body: _Optional[_Iterable[_Union[Statement, _Mapping]]] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class Branch(_message.Message):
    __slots__ = ("guard", "preamble", "actions", "postamble", "location")
    GUARD_FIELD_NUMBER: _ClassVar[int]
    PREAMBLE_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    POSTAMBLE_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    guard: str
    preamble: _containers.RepeatedCompositeFieldContainer[PythonBlock]
    actions: _containers.RepeatedCompositeFieldContainer[ActionCall]
    postamble: _containers.RepeatedCompositeFieldContainer[PythonBlock]
    location: SourceLocation
    def __init__(self, guard: _Optional[str] = ..., preamble: _Optional[_Iterable[_Union[PythonBlock, _Mapping]]] = ..., actions: _Optional[_Iterable[_Union[ActionCall, _Mapping]]] = ..., postamble: _Optional[_Iterable[_Union[PythonBlock, _Mapping]]] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class Conditional(_message.Message):
    __slots__ = ("branches", "target", "location")
    BRANCHES_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    branches: _containers.RepeatedCompositeFieldContainer[Branch]
    target: str
    location: SourceLocation
    def __init__(self, branches: _Optional[_Iterable[_Union[Branch, _Mapping]]] = ..., target: _Optional[str] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class ExceptHandler(_message.Message):
    __slots__ = ("exception_types", "preamble", "body", "postamble", "location")
    EXCEPTION_TYPES_FIELD_NUMBER: _ClassVar[int]
    PREAMBLE_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    POSTAMBLE_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    exception_types: _containers.RepeatedCompositeFieldContainer[ExceptionType]
    preamble: _containers.RepeatedCompositeFieldContainer[PythonBlock]
    body: _containers.RepeatedCompositeFieldContainer[ActionCall]
    postamble: _containers.RepeatedCompositeFieldContainer[PythonBlock]
    location: SourceLocation
    def __init__(self, exception_types: _Optional[_Iterable[_Union[ExceptionType, _Mapping]]] = ..., preamble: _Optional[_Iterable[_Union[PythonBlock, _Mapping]]] = ..., body: _Optional[_Iterable[_Union[ActionCall, _Mapping]]] = ..., postamble: _Optional[_Iterable[_Union[PythonBlock, _Mapping]]] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class ExceptionType(_message.Message):
    __slots__ = ("module", "name")
    MODULE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    module: str
    name: str
    def __init__(self, module: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class TryExcept(_message.Message):
    __slots__ = ("try_preamble", "try_body", "try_postamble", "handlers", "location")
    TRY_PREAMBLE_FIELD_NUMBER: _ClassVar[int]
    TRY_BODY_FIELD_NUMBER: _ClassVar[int]
    TRY_POSTAMBLE_FIELD_NUMBER: _ClassVar[int]
    HANDLERS_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    try_preamble: _containers.RepeatedCompositeFieldContainer[PythonBlock]
    try_body: _containers.RepeatedCompositeFieldContainer[ActionCall]
    try_postamble: _containers.RepeatedCompositeFieldContainer[PythonBlock]
    handlers: _containers.RepeatedCompositeFieldContainer[ExceptHandler]
    location: SourceLocation
    def __init__(self, try_preamble: _Optional[_Iterable[_Union[PythonBlock, _Mapping]]] = ..., try_body: _Optional[_Iterable[_Union[ActionCall, _Mapping]]] = ..., try_postamble: _Optional[_Iterable[_Union[PythonBlock, _Mapping]]] = ..., handlers: _Optional[_Iterable[_Union[ExceptHandler, _Mapping]]] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class Sleep(_message.Message):
    __slots__ = ("duration_expr", "location")
    DURATION_EXPR_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    duration_expr: str
    location: SourceLocation
    def __init__(self, duration_expr: _Optional[str] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class Return(_message.Message):
    __slots__ = ("expr", "action", "gather", "location")
    EXPR_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    GATHER_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    expr: str
    action: ActionCall
    gather: Gather
    location: SourceLocation
    def __init__(self, expr: _Optional[str] = ..., action: _Optional[_Union[ActionCall, _Mapping]] = ..., gather: _Optional[_Union[Gather, _Mapping]] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class Spread(_message.Message):
    __slots__ = ("action", "loop_var", "iterable", "target", "location")
    ACTION_FIELD_NUMBER: _ClassVar[int]
    LOOP_VAR_FIELD_NUMBER: _ClassVar[int]
    ITERABLE_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    action: ActionCall
    loop_var: str
    iterable: str
    target: str
    location: SourceLocation
    def __init__(self, action: _Optional[_Union[ActionCall, _Mapping]] = ..., loop_var: _Optional[str] = ..., iterable: _Optional[str] = ..., target: _Optional[str] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...

class Statement(_message.Message):
    __slots__ = ("action_call", "gather", "python_block", "loop", "conditional", "try_except", "sleep", "return_stmt", "spread")
    ACTION_CALL_FIELD_NUMBER: _ClassVar[int]
    GATHER_FIELD_NUMBER: _ClassVar[int]
    PYTHON_BLOCK_FIELD_NUMBER: _ClassVar[int]
    LOOP_FIELD_NUMBER: _ClassVar[int]
    CONDITIONAL_FIELD_NUMBER: _ClassVar[int]
    TRY_EXCEPT_FIELD_NUMBER: _ClassVar[int]
    SLEEP_FIELD_NUMBER: _ClassVar[int]
    RETURN_STMT_FIELD_NUMBER: _ClassVar[int]
    SPREAD_FIELD_NUMBER: _ClassVar[int]
    action_call: ActionCall
    gather: Gather
    python_block: PythonBlock
    loop: Loop
    conditional: Conditional
    try_except: TryExcept
    sleep: Sleep
    return_stmt: Return
    spread: Spread
    def __init__(self, action_call: _Optional[_Union[ActionCall, _Mapping]] = ..., gather: _Optional[_Union[Gather, _Mapping]] = ..., python_block: _Optional[_Union[PythonBlock, _Mapping]] = ..., loop: _Optional[_Union[Loop, _Mapping]] = ..., conditional: _Optional[_Union[Conditional, _Mapping]] = ..., try_except: _Optional[_Union[TryExcept, _Mapping]] = ..., sleep: _Optional[_Union[Sleep, _Mapping]] = ..., return_stmt: _Optional[_Union[Return, _Mapping]] = ..., spread: _Optional[_Union[Spread, _Mapping]] = ...) -> None: ...

class WorkflowParam(_message.Message):
    __slots__ = ("name", "type_annotation")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_ANNOTATION_FIELD_NUMBER: _ClassVar[int]
    name: str
    type_annotation: str
    def __init__(self, name: _Optional[str] = ..., type_annotation: _Optional[str] = ...) -> None: ...

class Workflow(_message.Message):
    __slots__ = ("name", "params", "body", "return_type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    RETURN_TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    params: _containers.RepeatedCompositeFieldContainer[WorkflowParam]
    body: _containers.RepeatedCompositeFieldContainer[Statement]
    return_type: str
    def __init__(self, name: _Optional[str] = ..., params: _Optional[_Iterable[_Union[WorkflowParam, _Mapping]]] = ..., body: _Optional[_Iterable[_Union[Statement, _Mapping]]] = ..., return_type: _Optional[str] = ...) -> None: ...

class ActionDefinition(_message.Message):
    __slots__ = ("name", "module", "param_names")
    NAME_FIELD_NUMBER: _ClassVar[int]
    MODULE_FIELD_NUMBER: _ClassVar[int]
    PARAM_NAMES_FIELD_NUMBER: _ClassVar[int]
    name: str
    module: str
    param_names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., module: _Optional[str] = ..., param_names: _Optional[_Iterable[str]] = ...) -> None: ...

class ParseError(_message.Message):
    __slots__ = ("message", "location")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    message: str
    location: SourceLocation
    def __init__(self, message: _Optional[str] = ..., location: _Optional[_Union[SourceLocation, _Mapping]] = ...) -> None: ...
