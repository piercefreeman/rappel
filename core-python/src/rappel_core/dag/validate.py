"""DAG validation utilities."""

from __future__ import annotations

from typing import Set

from .models import DAG, DagConversionError, EdgeType


def validate_dag(dag: DAG) -> None:
    validate_edges_reference_existing_nodes(dag)
    validate_output_nodes_have_no_outgoing_edges(dag)
    validate_loop_incr_edges(dag)
    validate_no_duplicate_state_machine_edges(dag)
    validate_input_nodes_have_no_incoming_edges(dag)


def validate_edges_reference_existing_nodes(dag: DAG) -> None:
    for edge in dag.edges:
        if edge.source not in dag.nodes:
            raise DagConversionError(
                f"DAG edge references non-existent source node '{edge.source}' -> '{edge.target}'"
            )
        if edge.target not in dag.nodes:
            raise DagConversionError(
                "DAG edge references non-existent target node "
                f"'{edge.target}' (from '{edge.source}', edge_type={edge.edge_type}, "
                f"exception_types={edge.exception_types})"
            )


def validate_output_nodes_have_no_outgoing_edges(dag: DAG) -> None:
    for node_id, node in dag.nodes.items():
        if node.node_type == "output" and ":" not in node_id:
            for edge in dag.edges:
                if (
                    edge.source == node_id
                    and edge.edge_type == EdgeType.STATE_MACHINE
                    and edge.exception_types is None
                ):
                    raise DagConversionError(
                        "Main output node "
                        f"'{node_id}' has non-exception outgoing state machine edge to "
                        f"'{edge.target}'"
                    )


def validate_loop_incr_edges(dag: DAG) -> None:
    for node_id in dag.nodes:
        if "loop_incr" not in node_id:
            continue
        for edge in dag.edges:
            if edge.source != node_id or edge.edge_type != EdgeType.STATE_MACHINE:
                continue
            if edge.exception_types is not None:
                continue
            if edge.is_loop_back and "loop_cond" in edge.target:
                continue
            raise DagConversionError(
                "Loop increment node "
                f"'{node_id}' has unexpected state machine edge to '{edge.target}'. "
                "Loop_incr should only have loop_back edges to loop_cond or exception edges. "
                "This suggests incorrect 'last_real_node' tracking during function expansion."
            )


def validate_no_duplicate_state_machine_edges(dag: DAG) -> None:
    seen: Set[str] = set()
    for edge in dag.edges:
        if edge.edge_type != EdgeType.STATE_MACHINE:
            continue
        if edge.exception_types is None:
            key = (
                f"{edge.source}->{edge.target}:loop_back={edge.is_loop_back},is_else={edge.is_else},"
                f"guard={edge.guard_string}"
            )
            if key in seen:
                raise DagConversionError(
                    "Duplicate state machine edge: "
                    f"{edge.source} -> {edge.target} (loop_back={edge.is_loop_back}, "
                    f"is_else={edge.is_else})"
                )
            seen.add(key)


def validate_input_nodes_have_no_incoming_edges(dag: DAG) -> None:
    for node_id, node in dag.nodes.items():
        if node.is_input:
            for edge in dag.edges:
                if edge.target == node_id and edge.edge_type == EdgeType.STATE_MACHINE:
                    raise DagConversionError(
                        f"Input node '{node_id}' has incoming state machine edge from '{edge.source}'"
                    )
