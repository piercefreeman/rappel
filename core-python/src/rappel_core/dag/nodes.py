"""DAG node definitions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional

from proto import ast_pb2 as ir


@dataclass(kw_only=True)
class DAGNode:
    """Base class for DAG nodes with computed labels and shared metadata.

    We keep rich node detail here so scheduling, validation, and visualization
    can share the same source of truth without re-deriving labels or intent.

    Visualization examples:
    - id="main_input_1", label="input: [x]"
    - id="action_4", label="@fetch() -> data"
    - id="join_7", label="join"
    """

    id: str
    node_uuid: uuid.UUID = field(default_factory=uuid.uuid4)
    function_name: Optional[str] = None

    NODE_TYPE: ClassVar[str] = "node"

    @property
    def node_type(self) -> str:
        return self.NODE_TYPE

    @property
    def label(self) -> str:
        raise NotImplementedError

    @property
    def is_input(self) -> bool:
        return False

    @property
    def is_output(self) -> bool:
        return False

    @property
    def is_aggregator(self) -> bool:
        return False

    @property
    def is_fn_call(self) -> bool:
        return False

    @property
    def is_spread(self) -> bool:
        return False


@dataclass(kw_only=True)
class InputNode(DAGNode):
    """Function entry node that declares input variables.

    Visualization example: label="input: [base, limit]"
    """

    io_vars: List[str] = field(default_factory=list)

    NODE_TYPE: ClassVar[str] = "input"

    @property
    def label(self) -> str:
        return "input: []" if not self.io_vars else f"input: [{', '.join(self.io_vars)}]"

    @property
    def is_input(self) -> bool:
        return True


@dataclass(kw_only=True)
class OutputNode(DAGNode):
    """Function output node that exposes return values.

    Visualization example: label="output: [result]"
    """

    io_vars: List[str] = field(default_factory=list)

    NODE_TYPE: ClassVar[str] = "output"

    @property
    def label(self) -> str:
        return f"output: [{', '.join(self.io_vars)}]"

    @property
    def is_output(self) -> bool:
        return True


@dataclass(kw_only=True)
class AssignmentNode(DAGNode):
    """Represents assignment statements in the DAG.

    Visualization example: label="total = ..."
    """

    targets: List[str] = field(default_factory=list)
    target: Optional[str] = None
    assign_expr: Optional[ir.Expr] = None
    label_hint: Optional[str] = None

    NODE_TYPE: ClassVar[str] = "assignment"

    def __post_init__(self) -> None:
        if self.target is None and self.targets:
            self.target = self.targets[0]

    @property
    def label(self) -> str:
        if self.label_hint:
            return self.label_hint
        if len(self.targets) > 1:
            return f"{', '.join(self.targets)} = ..."
        target = self.targets[0] if self.targets else self.target or "_"
        return f"{target} = ..."


@dataclass(kw_only=True)
class ActionCallNode(DAGNode):
    """Invokes an action via @action() with optional parallel/spread context.

    Visualization examples:
    - label="@double() -> value"
    - label="@fetch() [spread over item]"
    """

    action_name: str
    module_name: Optional[str] = None
    kwargs: Dict[str, str] = field(default_factory=dict)
    kwarg_exprs: Dict[str, ir.Expr] = field(default_factory=dict)
    policies: List[ir.PolicyBracket] = field(default_factory=list)
    targets: Optional[List[str]] = None
    target: Optional[str] = None
    parallel_index: Optional[int] = None
    aggregates_to: Optional[str] = None
    spread_loop_var: Optional[str] = None
    spread_collection_expr: Optional[ir.Expr] = None

    NODE_TYPE: ClassVar[str] = "action_call"

    def __post_init__(self) -> None:
        if self.target is None and self.targets:
            self.target = self.targets[0]

    @property
    def label(self) -> str:
        base = f"@{self.action_name}()"
        if self.spread_loop_var:
            return f"{base} [spread over {self.spread_loop_var}]"
        if self.parallel_index is not None:
            base = f"{base} [{self.parallel_index}]"
        if self.targets:
            if len(self.targets) == 1:
                return f"{base} -> {self.targets[0]}"
            return f"{base} -> ({', '.join(self.targets)})"
        if self.target:
            return f"{base} -> {self.target}"
        return base

    @property
    def is_spread(self) -> bool:
        return self.spread_loop_var is not None or self.spread_collection_expr is not None


@dataclass(kw_only=True)
class FnCallNode(DAGNode):
    """Invokes a user-defined function inside the DAG.

    Visualization example: label="helper() -> output"
    """

    called_function: str
    kwargs: Dict[str, str] = field(default_factory=dict)
    kwarg_exprs: Dict[str, ir.Expr] = field(default_factory=dict)
    targets: Optional[List[str]] = None
    target: Optional[str] = None
    assign_expr: Optional[ir.Expr] = None
    parallel_index: Optional[int] = None
    aggregates_to: Optional[str] = None

    NODE_TYPE: ClassVar[str] = "fn_call"

    def __post_init__(self) -> None:
        if self.target is None and self.targets:
            self.target = self.targets[0]

    @property
    def label(self) -> str:
        base = f"{self.called_function}()"
        if self.parallel_index is not None:
            base = f"{base} [{self.parallel_index}]"
        if self.targets:
            if len(self.targets) == 1:
                return f"{base} -> {self.targets[0]}"
            return f"{base} -> ({', '.join(self.targets)})"
        if self.target:
            return f"{base} -> {self.target}"
        return base

    @property
    def is_fn_call(self) -> bool:
        return True


@dataclass(kw_only=True)
class ParallelNode(DAGNode):
    """Parallel fan-out control node.

    Visualization example: label="parallel"
    """

    NODE_TYPE: ClassVar[str] = "parallel"

    @property
    def label(self) -> str:
        return "parallel"


@dataclass(kw_only=True)
class AggregatorNode(DAGNode):
    """Collects outputs from parallel or spread branches.

    Visualization examples:
    - label="aggregate -> result"
    - label="parallel_aggregate -> outputs"
    """

    aggregates_from: str
    targets: Optional[List[str]] = None
    target: Optional[str] = None
    aggregator_kind: str = "aggregate"

    NODE_TYPE: ClassVar[str] = "aggregator"

    def __post_init__(self) -> None:
        if self.target is None and self.targets:
            self.target = self.targets[0]

    @property
    def label(self) -> str:
        prefix = "parallel_aggregate" if self.aggregator_kind == "parallel" else "aggregate"
        if self.targets:
            if len(self.targets) == 1:
                return f"{prefix} -> {self.targets[0]}"
            return f"{prefix} -> ({', '.join(self.targets)})"
        if self.target:
            return f"{prefix} -> {self.target}"
        return prefix

    @property
    def is_aggregator(self) -> bool:
        return True


@dataclass(kw_only=True)
class BranchNode(DAGNode):
    """Conditional branch dispatch node.

    Visualization example: label="if guard"
    """

    description: str

    NODE_TYPE: ClassVar[str] = "branch"

    @property
    def label(self) -> str:
        return self.description


@dataclass(kw_only=True)
class JoinNode(DAGNode):
    """Converges multiple control-flow branches.

    Visualization example: label="join"
    """

    description: str
    targets: Optional[List[str]] = None
    target: Optional[str] = None

    NODE_TYPE: ClassVar[str] = "join"

    def __post_init__(self) -> None:
        if self.target is None and self.targets:
            self.target = self.targets[0]

    @property
    def label(self) -> str:
        return self.description


@dataclass(kw_only=True)
class ReturnNode(DAGNode):
    """Return node used when expanding nested functions.

    Visualization example: label="return"
    """

    assign_expr: Optional[ir.Expr] = None
    targets: Optional[List[str]] = None
    target: Optional[str] = None

    NODE_TYPE: ClassVar[str] = "return"

    def __post_init__(self) -> None:
        if self.target is None and self.targets:
            self.target = self.targets[0]

    @property
    def label(self) -> str:
        return "return"


@dataclass(kw_only=True)
class BreakNode(DAGNode):
    """Loop break node."""

    NODE_TYPE: ClassVar[str] = "break"

    @property
    def label(self) -> str:
        return "break"


@dataclass(kw_only=True)
class ContinueNode(DAGNode):
    """Loop continue node."""

    NODE_TYPE: ClassVar[str] = "continue"

    @property
    def label(self) -> str:
        return "continue"


@dataclass(kw_only=True)
class ExpressionNode(DAGNode):
    """Bare expression statement node."""

    NODE_TYPE: ClassVar[str] = "expression"

    @property
    def label(self) -> str:
        return "expr"
