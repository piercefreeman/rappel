"""Core DAG models and shared helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, NoReturn, Optional, Set

from proto import ast_pb2 as ir

from .nodes import DAGNode

EXCEPTION_SCOPE_VAR = "__rappel_exception__"


class DagConversionError(Exception):
    """Raised when IR -> DAG conversion fails."""


def assert_never(value: NoReturn) -> NoReturn:
    raise AssertionError(f"Unhandled value: {value!r}")


class EdgeType(str, Enum):
    """Classifies edges as control-flow (state machine) or data-flow.

    We keep the distinction so visualization and scheduling can render them
    differently (solid vs dashed) and so data dependencies can be computed
    independently of execution ordering.
    """

    STATE_MACHINE = "state_machine"
    DATA_FLOW = "data_flow"


@dataclass
class DAGEdge:
    """Directed edge between DAG nodes with execution or data semantics.

    We store rich metadata because both the runtime and the visualizer need to
    interpret the same graph: control-flow edges carry conditions/guards, while
    data-flow edges track which variable definition feeds which consumer.

    Visualization examples:
    - control: action_1 -> join_2 (condition="success")
    - control: branch_3 -> then_4 (condition="guarded")
    - data: assign_5 -> action_6 (variable="payload")
    """

    source: str
    target: str
    edge_type: EdgeType
    condition: Optional[str] = None
    variable: Optional[str] = None
    guard_expr: Optional[ir.Expr] = None
    is_else: bool = False
    exception_types: Optional[List[str]] = None
    exception_depth: Optional[int] = None
    is_loop_back: bool = False
    guard_string: Optional[str] = None

    @staticmethod
    def state_machine(source: str, target: str) -> "DAGEdge":
        return DAGEdge(
            source=source,
            target=target,
            edge_type=EdgeType.STATE_MACHINE,
        )

    @staticmethod
    def state_machine_with_condition(source: str, target: str, condition: str) -> "DAGEdge":
        return DAGEdge(
            source=source,
            target=target,
            edge_type=EdgeType.STATE_MACHINE,
            condition=condition,
        )

    @staticmethod
    def state_machine_with_guard(source: str, target: str, guard: ir.Expr) -> "DAGEdge":
        return DAGEdge(
            source=source,
            target=target,
            edge_type=EdgeType.STATE_MACHINE,
            condition="guarded",
            guard_expr=copy.deepcopy(guard),
        )

    @staticmethod
    def state_machine_else(source: str, target: str) -> "DAGEdge":
        return DAGEdge(
            source=source,
            target=target,
            edge_type=EdgeType.STATE_MACHINE,
            condition="else",
            is_else=True,
        )

    @staticmethod
    def state_machine_with_exception(
        source: str,
        target: str,
        exception_types: List[str],
    ) -> "DAGEdge":
        normalized = (
            []
            if len(exception_types) == 1 and exception_types[0] == "Exception"
            else list(exception_types)
        )
        condition = "except:*" if not normalized else f"except:{','.join(normalized)}"
        return DAGEdge(
            source=source,
            target=target,
            edge_type=EdgeType.STATE_MACHINE,
            condition=condition,
            exception_types=normalized,
        )

    @staticmethod
    def state_machine_success(source: str, target: str) -> "DAGEdge":
        return DAGEdge(
            source=source,
            target=target,
            edge_type=EdgeType.STATE_MACHINE,
            condition="success",
        )

    @staticmethod
    def data_flow(source: str, target: str, variable: str) -> "DAGEdge":
        return DAGEdge(
            source=source,
            target=target,
            edge_type=EdgeType.DATA_FLOW,
            variable=variable,
        )

    def with_loop_back(self, is_loop_back: bool) -> "DAGEdge":
        self.is_loop_back = is_loop_back
        return self

    def with_guard(self, guard: str) -> "DAGEdge":
        self.guard_string = guard
        return self


@dataclass
class DAG:
    """Container for DAG nodes/edges with helper queries.

    The DAG object is the common currency between conversion, scheduling, and
    visualization. We keep both node metadata and edge metadata so downstream
    tools can render a faithful control/data graph.

    Visualization example (pseudo):
    - nodes: input -> action -> output
    - edges: input -control-> action, action -data(var=x)-> output
    """

    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    edges: List[DAGEdge] = field(default_factory=list)
    entry_node: Optional[str] = None

    def add_node(self, node: DAGNode) -> None:
        if self.entry_node is None:
            self.entry_node = node.id
        self.nodes[node.id] = node

    def add_edge(self, edge: DAGEdge) -> None:
        self.edges.append(edge)

    def get_incoming_edges(self, node_id: str) -> List[DAGEdge]:
        return [edge for edge in self.edges if edge.target == node_id]

    def get_outgoing_edges(self, node_id: str) -> List[DAGEdge]:
        return [edge for edge in self.edges if edge.source == node_id]

    def get_state_machine_edges(self) -> List[DAGEdge]:
        return [edge for edge in self.edges if edge.edge_type == EdgeType.STATE_MACHINE]

    def get_data_flow_edges(self) -> List[DAGEdge]:
        return [edge for edge in self.edges if edge.edge_type == EdgeType.DATA_FLOW]

    def get_functions(self) -> List[str]:
        functions: Set[str] = set()
        for node in self.nodes.values():
            if node.function_name is not None:
                functions.add(node.function_name)
        return sorted(functions)

    def get_nodes_for_function(self, function_name: str) -> Dict[str, DAGNode]:
        return {
            node_id: node
            for node_id, node in self.nodes.items()
            if node.function_name == function_name
        }

    def get_edges_for_function(self, function_name: str) -> List[DAGEdge]:
        fn_nodes = set(self.get_nodes_for_function(function_name).keys())
        return [edge for edge in self.edges if edge.source in fn_nodes and edge.target in fn_nodes]


@dataclass
class ConvertedSubgraph:
    """Intermediate representation for stitching statement subgraphs.

    Every IR statement can expand into multiple DAG nodes. ConvertedSubgraph
    captures the "entry" and "exits" so the converter can wire the next
    statement without knowing the internal structure of the previous one.

    Examples:
    - Simple assignment: entry=assign_1, exits=[assign_1]
    - If/else: entry=branch_2, exits=[join_5]
    - Empty block: is_noop=True (frontier stays unchanged)
    """

    entry: Optional[str]
    exits: List[str]
    nodes: List[str]
    is_noop: bool

    @staticmethod
    def noop() -> "ConvertedSubgraph":
        return ConvertedSubgraph(entry=None, exits=[], nodes=[], is_noop=True)
