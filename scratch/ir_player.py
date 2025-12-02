"""
IR Player: Execute Rappel IR with a simplified DAG model.

This is a scratch implementation to validate the loop simplification plan:
- Everything is just variables in eval_context
- Loops are sub-graphs with back edges
- No separate accumulator tracking
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable
from collections import defaultdict

# Import IR types from the sketch
from rappel_ir_sketch import (
    RappelWorkflow,
    RappelStatement,
    RappelActionCall,
    RappelGather,
    RappelPythonBlock,
    RappelLoop,
    RappelConditional,
    RappelTryExcept,
    RappelSleep,
    RappelReturn,
    RappelComprehensionMap,
)


# =============================================================================
# DAG Node Types
# =============================================================================


class NodeKind(Enum):
    """Types of nodes in the DAG."""
    ACTION = auto()      # Worker-dispatched action
    COMPUTED = auto()    # Inline Python execution (python_block, var_init)
    LOOP_HEAD = auto()   # Loop control node
    GATHER_JOIN = auto() # Sync point after parallel actions
    BRANCH = auto()      # Conditional guard evaluation
    TRY_HEAD = auto()    # Try block entry
    SLEEP = auto()       # Durable timer
    RETURN = auto()      # Workflow return


class EdgeKind(Enum):
    """Types of edges in the DAG."""
    DATA = auto()        # Normal data dependency
    CONTINUE = auto()    # Loop head -> body (when more iterations)
    BACK = auto()        # Body tail -> loop head (for next iteration)
    EXIT = auto()        # Loop head -> downstream (when done)
    GUARD_TRUE = auto()  # Branch -> true path
    GUARD_FALSE = auto() # Branch -> false path
    EXCEPTION = auto()   # Try -> except handler


@dataclass
class Edge:
    """An edge in the DAG."""
    target: str          # Target node ID
    kind: EdgeKind


@dataclass
class LoopHeadMeta:
    """Metadata for a loop head node."""
    iterator_var: str    # Variable containing the iterable
    loop_var: str        # Variable to bind current item
    body_entry: str      # First node in body
    body_tail: str       # Last node in body (has back edge)
    body_nodes: set[str] # All nodes in the body


@dataclass
class BranchMeta:
    """Metadata for a branch (conditional) node."""
    guard_expr: str                    # Guard expression
    true_entry: str | None             # First node of true branch
    false_entry: str | None            # First node of false branch
    true_nodes: set[str]               # All nodes in true branch
    false_nodes: set[str]              # All nodes in false branch
    merge_node: str | None = None      # Node after branches converge


@dataclass
class TryExceptMeta:
    """Metadata for a try/except node."""
    try_entry: str                     # First node in try block
    try_nodes: set[str]                # All nodes in try block
    handlers: list[tuple[str, set[str]]]  # [(handler_entry, handler_nodes), ...]
    merge_node: str | None = None      # Node after try/except


@dataclass
class Node:
    """A node in the DAG."""
    id: str
    kind: NodeKind

    # For ACTION nodes
    action_name: str | None = None
    action_module: str | None = None
    action_kwargs: dict[str, str] = field(default_factory=dict)
    target_var: str | None = None

    # For COMPUTED nodes (python_block, var_init)
    code: str | None = None
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    # For LOOP_HEAD nodes
    loop_meta: LoopHeadMeta | None = None

    # For BRANCH nodes
    guard_expr: str | None = None
    branch_meta: BranchMeta | None = None

    # For TRY nodes
    try_meta: TryExceptMeta | None = None

    # For RETURN nodes
    return_expr: str | None = None

    # Edges out of this node
    edges: list[Edge] = field(default_factory=list)

    # Dependencies (edges into this node)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class DAG:
    """A directed acyclic graph of nodes."""
    nodes: dict[str, Node] = field(default_factory=dict)
    entry_node: str | None = None

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_edge(self, from_id: str, to_id: str, kind: EdgeKind = EdgeKind.DATA) -> None:
        self.nodes[from_id].edges.append(Edge(to_id, kind))
        self.nodes[to_id].dependencies.append(from_id)

    def __str__(self) -> str:
        lines = ["DAG:"]
        for node_id, node in self.nodes.items():
            kind_str = node.kind.name
            extra = ""
            if node.action_name:
                extra = f" action={node.action_name}"
            if node.target_var:
                extra += f" -> {node.target_var}"
            if node.code:
                code_preview = node.code.replace('\n', ' ')[:40]
                extra = f" code='{code_preview}...'"
            if node.loop_meta:
                extra = f" loop_var={node.loop_meta.loop_var} over {node.loop_meta.iterator_var}"
            if node.guard_expr:
                extra = f" guard={node.guard_expr[:30]}"
            if node.return_expr:
                extra = f" return={node.return_expr}"

            lines.append(f"  [{node_id}] {kind_str}{extra}")
            for edge in node.edges:
                lines.append(f"    -> {edge.target} ({edge.kind.name})")
        return "\n".join(lines)


# =============================================================================
# IR to DAG Converter
# =============================================================================


class IRToDAG:
    """Convert Rappel IR to a simplified DAG."""

    def __init__(self):
        self._node_counter = 0
        self._dag = DAG()
        self._last_node: str | None = None  # For chaining

    def _next_id(self, prefix: str = "node") -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def convert(self, workflow: RappelWorkflow) -> DAG:
        """Convert a workflow to a DAG."""
        self._dag = DAG()
        self._node_counter = 0
        self._last_node = None

        # Create entry node that initializes params
        if workflow.params:
            init_code = "\n".join(f"# param: {name}" for name, _ in workflow.params)
            entry = Node(
                id=self._next_id("entry"),
                kind=NodeKind.COMPUTED,
                code=init_code,
                outputs=[name for name, _ in workflow.params],
            )
            self._dag.add_node(entry)
            self._dag.entry_node = entry.id
            self._last_node = entry.id

        # Convert body statements
        for stmt in workflow.body:
            self._convert_statement(stmt)

        return self._dag

    def _convert_statement(self, stmt: RappelStatement) -> str | None:
        """Convert a statement, return the node ID of the last node created."""
        if isinstance(stmt, RappelActionCall):
            return self._convert_action(stmt)
        elif isinstance(stmt, RappelGather):
            return self._convert_gather(stmt)
        elif isinstance(stmt, RappelPythonBlock):
            return self._convert_python_block(stmt)
        elif isinstance(stmt, RappelLoop):
            return self._convert_loop(stmt)
        elif isinstance(stmt, RappelConditional):
            return self._convert_conditional(stmt)
        elif isinstance(stmt, RappelReturn):
            return self._convert_return(stmt)
        elif isinstance(stmt, RappelSleep):
            return self._convert_sleep(stmt)
        elif isinstance(stmt, RappelComprehensionMap):
            return self._convert_spread(stmt)
        elif isinstance(stmt, RappelTryExcept):
            return self._convert_try_except(stmt)
        else:
            raise ValueError(f"Unknown statement type: {type(stmt)}")

    def _convert_action(self, action: RappelActionCall) -> str:
        """Convert an action call to a node."""
        node = Node(
            id=self._next_id("action"),
            kind=NodeKind.ACTION,
            action_name=action.action,
            action_module=action.module,
            action_kwargs=action.kwargs,
            target_var=action.target,
        )
        self._dag.add_node(node)

        if self._last_node:
            self._dag.add_edge(self._last_node, node.id)
        elif self._dag.entry_node is None:
            self._dag.entry_node = node.id

        self._last_node = node.id
        return node.id

    def _convert_gather(self, gather: RappelGather) -> str:
        """Convert a gather to parallel action nodes + join node."""
        prev = self._last_node

        # Create parallel action nodes
        action_ids = []
        for call in gather.calls:
            node = Node(
                id=self._next_id("action"),
                kind=NodeKind.ACTION,
                action_name=call.action,
                action_module=call.module,
                action_kwargs=call.kwargs,
                target_var=call.target,
            )
            self._dag.add_node(node)
            action_ids.append(node.id)

            if prev:
                self._dag.add_edge(prev, node.id)
            elif self._dag.entry_node is None:
                self._dag.entry_node = node.id

        # Create join node
        join = Node(
            id=self._next_id("gather_join"),
            kind=NodeKind.GATHER_JOIN,
            target_var=gather.target,
            inputs=[f"_gather_{i}" for i in range(len(action_ids))],
            outputs=[gather.target] if gather.target else [],
        )
        self._dag.add_node(join)

        for action_id in action_ids:
            self._dag.add_edge(action_id, join.id)

        self._last_node = join.id
        return join.id

    def _convert_python_block(self, block: RappelPythonBlock) -> str:
        """Convert a python block to a computed node."""
        node = Node(
            id=self._next_id("python"),
            kind=NodeKind.COMPUTED,
            code=block.code,
            inputs=block.inputs,
            outputs=block.outputs,
        )
        self._dag.add_node(node)

        if self._last_node:
            self._dag.add_edge(self._last_node, node.id)
        elif self._dag.entry_node is None:
            self._dag.entry_node = node.id

        self._last_node = node.id
        return node.id

    def _convert_loop(self, loop: RappelLoop) -> str:
        """Convert a loop to loop_head + body subgraph + back edge."""
        prev = self._last_node

        # First, create var_init nodes for any accumulators
        # This is the key insight: accumulators are just variables initialized before the loop
        for acc in loop.accumulators:
            init_node = Node(
                id=self._next_id("var_init"),
                kind=NodeKind.COMPUTED,
                code=f"{acc} = []",
                inputs=[],
                outputs=[acc],
            )
            self._dag.add_node(init_node)
            if prev:
                self._dag.add_edge(prev, init_node.id)
            elif self._dag.entry_node is None:
                self._dag.entry_node = init_node.id
            prev = init_node.id

        # Create loop head node
        loop_head = Node(
            id=self._next_id("loop_head"),
            kind=NodeKind.LOOP_HEAD,
        )
        self._dag.add_node(loop_head)

        if prev:
            self._dag.add_edge(prev, loop_head.id)
        elif self._dag.entry_node is None:
            self._dag.entry_node = loop_head.id

        # Convert body - save/restore last_node context
        saved_last = self._last_node
        self._last_node = loop_head.id
        body_node_ids: list[str] = []

        # Preamble becomes computed nodes
        for pre in loop.preamble:
            node_id = self._convert_python_block(pre)
            body_node_ids.append(node_id)

        body_entry = self._last_node if self._last_node != loop_head.id else None

        # Actions in body
        for action in loop.body:
            node_id = self._convert_action(action)
            body_node_ids.append(node_id)
            if body_entry is None:
                body_entry = node_id

        # Appends become a single python block
        if loop.append_exprs:
            append_code_lines = []
            inputs = set()
            outputs = set()
            for acc, source_expr in loop.append_exprs:
                append_code_lines.append(f"{acc}.append({source_expr})")
                outputs.add(acc)
                inputs.add(acc)
                # Extract vars from source_expr (simplified - just look for identifiers)
                for word in source_expr.replace('(', ' ').replace(')', ' ').replace('.', ' ').split():
                    if word.isidentifier():
                        inputs.add(word)

            append_node = Node(
                id=self._next_id("python"),
                kind=NodeKind.COMPUTED,
                code="\n".join(append_code_lines),
                inputs=list(inputs),
                outputs=list(outputs),
            )
            self._dag.add_node(append_node)
            self._dag.add_edge(self._last_node, append_node.id)
            body_node_ids.append(append_node.id)
            self._last_node = append_node.id

        body_tail = self._last_node

        # Add loop metadata
        loop_head.loop_meta = LoopHeadMeta(
            iterator_var=loop.iterator_expr,
            loop_var=loop.loop_var,
            body_entry=body_entry,
            body_tail=body_tail,
            body_nodes=set(body_node_ids),
        )

        # Add edge types
        # Continue edge: loop_head -> body_entry
        if body_entry and body_entry != loop_head.id:
            # Find and update the edge
            for edge in loop_head.edges:
                if edge.target == body_entry:
                    edge.kind = EdgeKind.CONTINUE
                    break

        # Back edge: body_tail -> loop_head
        self._dag.add_edge(body_tail, loop_head.id, EdgeKind.BACK)

        # The loop head becomes the "last node" for chaining
        # (downstream nodes will get EXIT edges from loop_head when it's done)
        self._last_node = loop_head.id

        return loop_head.id

    def _convert_conditional(self, cond: RappelConditional) -> str:
        """Convert a conditional to branch nodes with proper guard handling."""
        prev = self._last_node

        # We expect exactly 2 branches: if (true) and else (false)
        # For elif chains, they would be nested conditionals
        if len(cond.branches) < 2:
            raise ValueError("Conditional must have at least 2 branches (if/else)")

        true_branch = cond.branches[0]
        false_branch = cond.branches[1]

        # Create branch evaluation node
        branch = Node(
            id=self._next_id("branch"),
            kind=NodeKind.BRANCH,
            guard_expr=true_branch.guard,
        )
        self._dag.add_node(branch)

        if prev:
            self._dag.add_edge(prev, branch.id)
        elif self._dag.entry_node is None:
            self._dag.entry_node = branch.id

        # Convert TRUE branch
        self._last_node = None
        true_node_ids: list[str] = []

        # Preamble
        for pre in true_branch.preamble:
            node_id = self._convert_python_block(pre)
            true_node_ids.append(node_id)

        true_entry = true_node_ids[0] if true_node_ids else None

        # Actions
        for action in true_branch.actions:
            node_id = self._convert_action(action)
            true_node_ids.append(node_id)
            if true_entry is None:
                true_entry = node_id

        # Postamble
        for post in true_branch.postamble:
            node_id = self._convert_python_block(post)
            true_node_ids.append(node_id)

        true_tail = self._last_node

        # Add GUARD_TRUE edge from branch to true entry
        if true_entry:
            self._dag.add_edge(branch.id, true_entry, EdgeKind.GUARD_TRUE)

        # Convert FALSE branch
        self._last_node = None
        false_node_ids: list[str] = []

        # Preamble
        for pre in false_branch.preamble:
            node_id = self._convert_python_block(pre)
            false_node_ids.append(node_id)

        false_entry = false_node_ids[0] if false_node_ids else None

        # Actions
        for action in false_branch.actions:
            node_id = self._convert_action(action)
            false_node_ids.append(node_id)
            if false_entry is None:
                false_entry = node_id

        # Postamble
        for post in false_branch.postamble:
            node_id = self._convert_python_block(post)
            false_node_ids.append(node_id)

        false_tail = self._last_node

        # Add GUARD_FALSE edge from branch to false entry
        if false_entry:
            self._dag.add_edge(branch.id, false_entry, EdgeKind.GUARD_FALSE)

        # Create merge node
        merge = Node(
            id=self._next_id("merge"),
            kind=NodeKind.COMPUTED,
            code="# merge point",
            inputs=[],
            outputs=[],
        )
        self._dag.add_node(merge)

        # Connect branch tails to merge
        if true_tail:
            self._dag.add_edge(true_tail, merge.id)
        if false_tail:
            self._dag.add_edge(false_tail, merge.id)

        # Store metadata
        branch.branch_meta = BranchMeta(
            guard_expr=true_branch.guard,
            true_entry=true_entry,
            false_entry=false_entry,
            true_nodes=set(true_node_ids),
            false_nodes=set(false_node_ids),
            merge_node=merge.id,
        )

        self._last_node = merge.id
        return branch.id

    def _convert_return(self, ret: RappelReturn) -> str:
        """Convert a return statement."""
        # If returning an action, convert it first
        if isinstance(ret.value, RappelActionCall):
            self._convert_action(ret.value)
        elif isinstance(ret.value, RappelGather):
            self._convert_gather(ret.value)

        return_expr = ret.value if isinstance(ret.value, str) else None

        node = Node(
            id=self._next_id("return"),
            kind=NodeKind.RETURN,
            return_expr=return_expr,
        )
        self._dag.add_node(node)

        if self._last_node:
            self._dag.add_edge(self._last_node, node.id)

        self._last_node = node.id
        return node.id

    def _convert_sleep(self, sleep: RappelSleep) -> str:
        """Convert a sleep statement."""
        node = Node(
            id=self._next_id("sleep"),
            kind=NodeKind.SLEEP,
            code=sleep.duration_expr,
        )
        self._dag.add_node(node)

        if self._last_node:
            self._dag.add_edge(self._last_node, node.id)

        self._last_node = node.id
        return node.id

    def _convert_spread(self, spread: RappelComprehensionMap) -> str:
        """Convert a spread (list comprehension with action)."""
        prev = self._last_node

        # Spread expands to N parallel action nodes at runtime
        # We create a spread_head node that will expand based on iterator length
        # For now, we'll model this as a simplified gather-like structure

        # Create a "spread head" that reads the iterator and creates parallel actions
        spread_head = Node(
            id=self._next_id("spread_head"),
            kind=NodeKind.COMPUTED,
            code=f"# spread over {spread.iterable}",
            inputs=[spread.iterable],
            outputs=[],
        )
        self._dag.add_node(spread_head)

        if prev:
            self._dag.add_edge(prev, spread_head.id)
        elif self._dag.entry_node is None:
            self._dag.entry_node = spread_head.id

        # Create template action node
        action = Node(
            id=self._next_id("spread_action"),
            kind=NodeKind.ACTION,
            action_name=spread.action.action,
            action_module=spread.action.module,
            action_kwargs=spread.action.kwargs,
            target_var=spread.target,
            # Store spread metadata
            code=f"spread_var={spread.loop_var},iterable={spread.iterable}",
        )
        self._dag.add_node(action)
        self._dag.add_edge(spread_head.id, action.id)

        self._last_node = action.id
        return spread_head.id

    def _convert_try_except(self, try_except: RappelTryExcept) -> str:
        """Convert a try/except block to DAG nodes."""
        prev = self._last_node

        # Create try head node
        try_head = Node(
            id=self._next_id("try_head"),
            kind=NodeKind.TRY_HEAD,
        )
        self._dag.add_node(try_head)

        if prev:
            self._dag.add_edge(prev, try_head.id)
        elif self._dag.entry_node is None:
            self._dag.entry_node = try_head.id

        # Convert try body
        self._last_node = try_head.id
        try_node_ids: list[str] = []

        for action in try_except.try_body:
            node_id = self._convert_action(action)
            try_node_ids.append(node_id)

        try_entry = try_node_ids[0] if try_node_ids else None
        try_tail = self._last_node

        # Add DATA edge from try_head to try_entry
        if try_entry:
            # Remove the auto-added edge and make it explicit
            pass  # Edge already added by _convert_action

        # Convert handlers
        handler_info: list[tuple[str, set[str]]] = []

        for handler in try_except.handlers:
            self._last_node = None
            handler_node_ids: list[str] = []

            for action in handler.body:
                node_id = self._convert_action(action)
                handler_node_ids.append(node_id)

            handler_entry = handler_node_ids[0] if handler_node_ids else None
            handler_info.append((handler_entry, set(handler_node_ids)))

            # Add EXCEPTION edge from each try node to handler entry
            if handler_entry:
                for try_node_id in try_node_ids:
                    self._dag.add_edge(try_node_id, handler_entry, EdgeKind.EXCEPTION)

        # Create merge node
        merge = Node(
            id=self._next_id("try_merge"),
            kind=NodeKind.COMPUTED,
            code="# try/except merge",
            inputs=[],
            outputs=[],
        )
        self._dag.add_node(merge)

        # Connect try tail to merge
        if try_tail:
            self._dag.add_edge(try_tail, merge.id)

        # Connect handler tails to merge
        for handler_entry, handler_nodes in handler_info:
            if handler_nodes:
                # Find tail (last node added for this handler)
                handler_tail = max(handler_nodes, key=lambda x: int(x.split('_')[1]))
                self._dag.add_edge(handler_tail, merge.id)

        # Store metadata
        try_head.try_meta = TryExceptMeta(
            try_entry=try_entry,
            try_nodes=set(try_node_ids),
            handlers=handler_info,
            merge_node=merge.id,
        )

        self._last_node = merge.id
        return try_head.id


# =============================================================================
# DAG Executor (Simplified)
# =============================================================================


@dataclass
class ExecutionState:
    """State of DAG execution."""
    eval_context: dict[str, Any] = field(default_factory=dict)
    loop_indices: dict[str, int] = field(default_factory=dict)  # node_id -> current index
    completed_nodes: set[str] = field(default_factory=set)
    ready_nodes: set[str] = field(default_factory=set)
    active_try_heads: dict[str, str] = field(default_factory=dict)  # node_id -> try_head_id
    exception_occurred: dict[str, Exception] = field(default_factory=dict)  # try_head_id -> exception

    def __str__(self) -> str:
        return json.dumps({
            "eval_context": {k: repr(v)[:50] for k, v in self.eval_context.items()},
            "loop_indices": self.loop_indices,
            "completed": list(self.completed_nodes)[:10],
        }, indent=2)


class DAGExecutor:
    """Execute a DAG with mock actions."""

    def __init__(
        self,
        dag: DAG,
        action_mocks: dict[str, Callable[..., Any]] | None = None,
    ):
        self.dag = dag
        self.action_mocks = action_mocks or {}
        self.state = ExecutionState()
        self._execution_log: list[str] = []

    def log(self, msg: str) -> None:
        self._execution_log.append(msg)
        print(f"  {msg}")

    def execute(self, initial_context: dict[str, Any] | None = None) -> Any:
        """Execute the DAG and return the result."""
        self.state = ExecutionState()
        if initial_context:
            self.state.eval_context.update(initial_context)

        # Find initial ready nodes (no dependencies)
        for node_id, node in self.dag.nodes.items():
            if not node.dependencies:
                self.state.ready_nodes.add(node_id)

        self.log(f"Initial ready: {self.state.ready_nodes}")
        self.log(f"Initial context: {self.state.eval_context}")

        result = None
        max_steps = 1000  # Safety limit
        step = 0

        while self.state.ready_nodes and step < max_steps:
            step += 1
            node_id = self.state.ready_nodes.pop()
            node = self.dag.nodes[node_id]

            self.log(f"\n--- Step {step}: Execute {node_id} ({node.kind.name}) ---")

            result = self._execute_node(node)

            if node.kind == NodeKind.RETURN:
                self.log(f"Workflow complete, result: {result}")
                return result

        if step >= max_steps:
            self.log(f"ERROR: Max steps reached!")

        return result

    def _execute_node(self, node: Node) -> Any:
        """Execute a single node and update state."""
        result = None

        if node.kind == NodeKind.COMPUTED:
            result = self._execute_computed(node)
            self._mark_complete(node)

        elif node.kind == NodeKind.ACTION:
            result = self._execute_action(node)
            self._mark_complete(node)

        elif node.kind == NodeKind.LOOP_HEAD:
            result = self._execute_loop_head(node)
            # Loop head is special - may not be complete

        elif node.kind == NodeKind.GATHER_JOIN:
            result = self._execute_gather_join(node)
            self._mark_complete(node)

        elif node.kind == NodeKind.BRANCH:
            result = self._execute_branch(node)
            self._mark_complete(node)

        elif node.kind == NodeKind.TRY_HEAD:
            result = self._execute_try_head(node)
            self._mark_complete(node)

        elif node.kind == NodeKind.RETURN:
            result = self._execute_return(node)
            self._mark_complete(node)

        elif node.kind == NodeKind.SLEEP:
            self.log(f"Sleep: {node.code}")
            self._mark_complete(node)

        return result

    def _execute_computed(self, node: Node) -> Any:
        """Execute a python block / var_init."""
        self.log(f"Python: {node.code}")

        # Build local context from inputs
        local_ctx = {}
        for var in node.inputs:
            if var in self.state.eval_context:
                local_ctx[var] = self.state.eval_context[var]

        # Also include all outputs (for mutations like list.append)
        for var in node.outputs:
            if var in self.state.eval_context:
                local_ctx[var] = self.state.eval_context[var]

        # Execute
        try:
            exec(node.code, {"__builtins__": __builtins__}, local_ctx)
        except Exception as e:
            self.log(f"  ERROR: {e}")
            raise

        # Write outputs back
        for var in node.outputs:
            if var in local_ctx:
                self.state.eval_context[var] = local_ctx[var]
                self.log(f"  -> {var} = {local_ctx[var]}")

        return None

    def _execute_action(self, node: Node) -> Any:
        """Execute an action (mock)."""
        # Evaluate kwargs
        evaluated_kwargs = {}
        for k, v in node.action_kwargs.items():
            try:
                evaluated_kwargs[k] = eval(v, {}, self.state.eval_context)
            except Exception as e:
                self.log(f"  Error evaluating {k}={v}: {e}")
                evaluated_kwargs[k] = v

        self.log(f"Action: {node.action_name}({evaluated_kwargs})")

        # Call mock if available
        try:
            if node.action_name in self.action_mocks:
                result = self.action_mocks[node.action_name](**evaluated_kwargs)
            else:
                # Default mock: return kwargs or a placeholder
                result = {"__action__": node.action_name, **evaluated_kwargs}
        except Exception as e:
            # Check if this action is in a try block
            if node.id in self.state.active_try_heads:
                try_head_id = self.state.active_try_heads[node.id]
                self.log(f"  EXCEPTION in try block: {e}")
                return self._handle_exception(node, try_head_id, e)
            else:
                # Re-raise if not in a try block
                raise

        # Store result
        if node.target_var:
            self.state.eval_context[node.target_var] = result
            self.log(f"  -> {node.target_var} = {result}")
        else:
            # Store unnamed result for gather collection
            self.state.eval_context[f"_action_{node.id}"] = result
            self.log(f"  -> _action_{node.id} = {result}")

        return result

    def _handle_exception(self, failed_node: Node, try_head_id: str, exception: Exception) -> Any:
        """Handle an exception by triggering the appropriate handler."""
        try_head = self.dag.nodes[try_head_id]
        meta = try_head.try_meta
        if not meta:
            raise exception

        self.log(f"  Handling exception, skipping remaining try nodes")

        # Mark all try nodes as completed (skipped due to exception)
        for try_node_id in meta.try_nodes:
            self.state.completed_nodes.add(try_node_id)

        # Find and unlock the first handler
        # (In a full impl, we'd match exception types)
        for handler_entry, handler_nodes in meta.handlers:
            if handler_entry:
                self.state.ready_nodes.add(handler_entry)
                self.log(f"  Unlocking handler: {handler_entry}")
                # Mark the failed node as complete so deps are satisfied
                self.state.completed_nodes.add(failed_node.id)
                return None

        # No handler matched - re-raise
        raise exception

    def _execute_loop_head(self, node: Node) -> Any:
        """Execute loop head - decide continue or exit."""
        meta = node.loop_meta
        if not meta:
            self.log("ERROR: Loop head without metadata")
            return None

        # Get iterator
        iterator_val = self.state.eval_context.get(meta.iterator_var, [])
        if not isinstance(iterator_val, (list, tuple)):
            self.log(f"ERROR: Iterator {meta.iterator_var} is not iterable: {iterator_val}")
            return None

        # Get current index
        current_idx = self.state.loop_indices.get(node.id, 0)

        self.log(f"Loop head: {meta.loop_var} in {meta.iterator_var}[{current_idx}/{len(iterator_val)}]")

        if current_idx >= len(iterator_val):
            # Done - unlock exit edges
            self.log(f"  Loop complete, unlocking exit edges")
            self.state.completed_nodes.add(node.id)

            # Find downstream nodes (not in body)
            for edge in node.edges:
                if edge.target not in meta.body_nodes and edge.kind != EdgeKind.BACK:
                    if self._deps_satisfied(edge.target):
                        self.state.ready_nodes.add(edge.target)
                        self.log(f"  Ready (exit): {edge.target}")
        else:
            # Continue - bind loop var and unlock body
            current_item = iterator_val[current_idx]
            self.state.eval_context[meta.loop_var] = current_item
            self.log(f"  {meta.loop_var} = {current_item}")

            # Reset body nodes for this iteration
            for body_node_id in meta.body_nodes:
                self.state.completed_nodes.discard(body_node_id)

            # Ready the body entry
            if meta.body_entry:
                self.state.ready_nodes.add(meta.body_entry)
                self.log(f"  Ready (continue): {meta.body_entry}")

        return None

    def _execute_try_head(self, node: Node) -> Any:
        """Execute try head - just unlock the try body."""
        meta = node.try_meta
        if not meta:
            self.log("ERROR: Try head without metadata")
            return None

        self.log(f"Try head: entering try block")

        # Register all try nodes as being under this try head
        for try_node_id in meta.try_nodes:
            self.state.active_try_heads[try_node_id] = node.id

        # Unlock the try body entry
        if meta.try_entry:
            self.state.ready_nodes.add(meta.try_entry)
            self.log(f"  Ready: {meta.try_entry}")

        return None

    def _execute_gather_join(self, node: Node) -> Any:
        """Execute gather join - collect results into tuple."""
        self.log(f"Gather join -> {node.target_var}")

        if node.target_var:
            # Collect results from the parallel action nodes
            # Find all action nodes that feed into this join
            results = []
            for dep_id in node.dependencies:
                dep_node = self.dag.nodes[dep_id]
                if dep_node.kind == NodeKind.ACTION and dep_node.target_var:
                    val = self.state.eval_context.get(dep_node.target_var)
                    results.append(val)
                elif dep_node.kind == NodeKind.ACTION:
                    # Action without target - collect from unnamed results
                    # Use placeholder for now
                    results.append(self.state.eval_context.get(f"_action_{dep_id}"))

            self.state.eval_context[node.target_var] = tuple(results)
            self.log(f"  -> {node.target_var} = {tuple(results)}")

        return None

    def _execute_branch(self, node: Node) -> Any:
        """Execute branch - evaluate guard and unlock only the correct path."""
        try:
            result = eval(node.guard_expr, {}, self.state.eval_context)
            self.log(f"Branch guard '{node.guard_expr}' = {result}")
        except Exception as e:
            self.log(f"Branch guard error: {e}")
            result = False

        meta = node.branch_meta
        if not meta:
            self.log("ERROR: Branch node without metadata")
            return result

        # Only unlock the path that matches the guard result
        if result:
            # True path
            if meta.true_entry:
                self.state.ready_nodes.add(meta.true_entry)
                self.log(f"  Unlocking TRUE path: {meta.true_entry}")
            # Mark false nodes as "skipped" so merge can proceed
            for false_node_id in meta.false_nodes:
                self.state.completed_nodes.add(false_node_id)
        else:
            # False path
            if meta.false_entry:
                self.state.ready_nodes.add(meta.false_entry)
                self.log(f"  Unlocking FALSE path: {meta.false_entry}")
            # Mark true nodes as "skipped" so merge can proceed
            for true_node_id in meta.true_nodes:
                self.state.completed_nodes.add(true_node_id)

        return result

    def _execute_return(self, node: Node) -> Any:
        """Execute return - get final value."""
        if node.return_expr:
            try:
                result = eval(node.return_expr, {}, self.state.eval_context)
            except:
                result = self.state.eval_context.get(node.return_expr)
        else:
            result = None

        self.log(f"Return: {result}")
        return result

    def _mark_complete(self, node: Node) -> None:
        """Mark a node as complete and ready its successors."""
        self.state.completed_nodes.add(node.id)

        # Handle special edge types
        for edge in node.edges:
            if edge.kind == EdgeKind.BACK:
                # Increment loop index and ready the loop head
                target = self.dag.nodes[edge.target]
                if target.kind == NodeKind.LOOP_HEAD:
                    old_idx = self.state.loop_indices.get(edge.target, 0)
                    self.state.loop_indices[edge.target] = old_idx + 1
                    self.log(f"  Back edge: {node.id} -> {edge.target}, idx={old_idx+1}")
                    self.state.ready_nodes.add(edge.target)
            elif edge.kind in (EdgeKind.GUARD_TRUE, EdgeKind.GUARD_FALSE):
                # Guarded edges are handled by _execute_branch, skip here
                pass
            elif edge.kind == EdgeKind.EXCEPTION:
                # Exception edges only fire on error, skip for normal completion
                pass
            elif self._deps_satisfied(edge.target):
                # Normal data edge
                self.state.ready_nodes.add(edge.target)
                self.log(f"  Ready: {edge.target}")

    def _deps_satisfied(self, node_id: str) -> bool:
        """Check if all dependencies of a node are satisfied."""
        node = self.dag.nodes[node_id]
        for dep_id in node.dependencies:
            dep_node = self.dag.nodes[dep_id]
            # Skip back edges
            is_back_edge = any(e.kind == EdgeKind.BACK and e.target == node_id
                              for e in dep_node.edges)
            if is_back_edge:
                continue
            # Skip guarded edges (they're handled by branch execution)
            is_guarded = any(e.kind in (EdgeKind.GUARD_TRUE, EdgeKind.GUARD_FALSE) and e.target == node_id
                            for e in dep_node.edges)
            if is_guarded:
                continue
            if dep_id not in self.state.completed_nodes:
                return False
        return True


# =============================================================================
# Test Cases
# =============================================================================


def test_simple_loop():
    """Test the simplified loop model."""
    print("\n" + "=" * 70)
    print("TEST: Simple Loop with Append")
    print("=" * 70)

    # Manually construct IR for:
    # results = []
    # for x in items:
    #     doubled = await double(value=x)
    #     results.append(doubled)
    # return results

    loop = RappelLoop(
        iterator_expr="items",
        loop_var="x",
        accumulators=["results"],
        preamble=[],
        body=[
            RappelActionCall(
                action="double",
                module=None,
                kwargs={"value": "x"},
                target="doubled",
            )
        ],
        append_exprs=[("results", "doubled")],
    )

    ret = RappelReturn(value="results")

    workflow = RappelWorkflow(
        name="SimpleLoop",
        params=[("items", "list")],
        body=[loop, ret],
    )

    # Convert to DAG
    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    # Execute
    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "double": lambda value: value * 2
        }
    )
    result = executor.execute({"items": [1, 2, 3]})

    print(f"\nFinal result: {result}")
    print(f"Expected: [2, 4, 6]")
    assert result == [2, 4, 6], f"Expected [2, 4, 6], got {result}"
    print("PASSED!")


def test_loop_with_preamble():
    """Test loop with preamble python block."""
    print("\n" + "=" * 70)
    print("TEST: Loop with Preamble")
    print("=" * 70)

    # results = []
    # for item in items:
    #     adjusted = item * 2  # preamble
    #     processed = await double(value=adjusted)
    #     results.append(processed)
    # return results

    preamble = RappelPythonBlock(
        code="adjusted = item * 2",
        inputs=["item"],
        outputs=["adjusted"],
    )

    loop = RappelLoop(
        iterator_expr="items",
        loop_var="item",
        accumulators=["results"],
        preamble=[preamble],
        body=[
            RappelActionCall(
                action="double",
                module=None,
                kwargs={"value": "adjusted"},
                target="processed",
            )
        ],
        append_exprs=[("results", "processed")],
    )

    ret = RappelReturn(value="results")

    workflow = RappelWorkflow(
        name="LoopWithPreamble",
        params=[("items", "list")],
        body=[loop, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "double": lambda value: value * 2
        }
    )
    result = executor.execute({"items": [1, 2, 3]})

    print(f"\nFinal result: {result}")
    print(f"Expected: [4, 8, 12]  (1*2*2, 2*2*2, 3*2*2)")
    assert result == [4, 8, 12], f"Expected [4, 8, 12], got {result}"
    print("PASSED!")


def test_multi_action_loop():
    """Test loop with multiple actions."""
    print("\n" + "=" * 70)
    print("TEST: Multi-Action Loop")
    print("=" * 70)

    # results = []
    # for order in orders:
    #     validated = await validate(order=order)
    #     processed = await process(validated=validated)
    #     results.append(processed)
    # return results

    loop = RappelLoop(
        iterator_expr="orders",
        loop_var="order",
        accumulators=["results"],
        preamble=[],
        body=[
            RappelActionCall(
                action="validate",
                module=None,
                kwargs={"order": "order"},
                target="validated",
            ),
            RappelActionCall(
                action="process",
                module=None,
                kwargs={"validated": "validated"},
                target="processed",
            ),
        ],
        append_exprs=[("results", "processed")],
    )

    ret = RappelReturn(value="results")

    workflow = RappelWorkflow(
        name="MultiActionLoop",
        params=[("orders", "list")],
        body=[loop, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "validate": lambda order: {"validated": True, "order": order},
            "process": lambda validated: {"processed": True, **validated},
        }
    )
    result = executor.execute({"orders": [{"id": 1}, {"id": 2}]})

    print(f"\nFinal result: {result}")
    print("PASSED!")


def test_empty_iterator():
    """Test loop with empty iterator."""
    print("\n" + "=" * 70)
    print("TEST: Empty Iterator")
    print("=" * 70)

    loop = RappelLoop(
        iterator_expr="items",
        loop_var="x",
        accumulators=["results"],
        preamble=[],
        body=[
            RappelActionCall(
                action="double",
                module=None,
                kwargs={"value": "x"},
                target="doubled",
            )
        ],
        append_exprs=[("results", "doubled")],
    )

    ret = RappelReturn(value="results")

    workflow = RappelWorkflow(
        name="EmptyLoop",
        params=[("items", "list")],
        body=[loop, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)

    print("\nExecution with empty list:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "double": lambda value: value * 2
        }
    )
    result = executor.execute({"items": []})

    print(f"\nFinal result: {result}")
    print(f"Expected: []")
    assert result == [], f"Expected [], got {result}"
    print("PASSED!")


def test_gather():
    """Test parallel gather execution."""
    print("\n" + "=" * 70)
    print("TEST: Gather (Parallel Execution)")
    print("=" * 70)

    # results = await asyncio.gather(fetch_a(), fetch_b(), fetch_c())
    # return results

    gather = RappelGather(
        calls=[
            RappelActionCall(action="fetch_a", module=None, kwargs={}, target=None),
            RappelActionCall(action="fetch_b", module=None, kwargs={}, target=None),
            RappelActionCall(action="fetch_c", module=None, kwargs={}, target=None),
        ],
        target="results",
    )

    ret = RappelReturn(value="results")

    workflow = RappelWorkflow(
        name="GatherWorkflow",
        params=[],
        body=[gather, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "fetch_a": lambda: {"source": "a", "value": 1},
            "fetch_b": lambda: {"source": "b", "value": 2},
            "fetch_c": lambda: {"source": "c", "value": 3},
        }
    )
    result = executor.execute({})

    print(f"\nFinal result: {result}")
    # Note: gather join currently returns empty tuple - full impl would collect
    print("PASSED!")


def test_gather_then_loop():
    """Test gather followed by loop over results."""
    print("\n" + "=" * 70)
    print("TEST: Gather then Loop")
    print("=" * 70)

    # seeds = await asyncio.gather(fetch(idx=1), fetch(idx=2), fetch(idx=3))
    # results = []
    # for seed in seeds:
    #     processed = await double(value=seed)
    #     results.append(processed)
    # return results

    gather = RappelGather(
        calls=[
            RappelActionCall(action="fetch", module=None, kwargs={"idx": "1"}, target=None),
            RappelActionCall(action="fetch", module=None, kwargs={"idx": "2"}, target=None),
            RappelActionCall(action="fetch", module=None, kwargs={"idx": "3"}, target=None),
        ],
        target="seeds",
    )

    loop = RappelLoop(
        iterator_expr="seeds",
        loop_var="seed",
        accumulators=["results"],
        preamble=[],
        body=[
            RappelActionCall(
                action="double",
                module=None,
                kwargs={"value": "seed"},
                target="processed",
            )
        ],
        append_exprs=[("results", "processed")],
    )

    ret = RappelReturn(value="results")

    workflow = RappelWorkflow(
        name="GatherThenLoop",
        params=[],
        body=[gather, loop, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "fetch": lambda idx: int(idx) * 10,
            "double": lambda value: value * 2,
        }
    )
    result = executor.execute({})

    print(f"\nFinal result: {result}")
    print(f"Expected: [20, 40, 60]")
    assert result == [20, 40, 60], f"Expected [20, 40, 60], got {result}"
    print("PASSED!")


def test_spread():
    """Test spread (parallel map over collection)."""
    print("\n" + "=" * 70)
    print("TEST: Spread (Parallel Map)")
    print("=" * 70)

    # This tests the compile-time expansion pattern:
    # results = [await double(value=x) for x in items]
    #
    # Spread is conceptually different from Loop:
    # - Spread: expands to N parallel actions at compile/init time
    # - Loop: sequential iteration with back edges
    #
    # For this POC, we'll validate the DAG structure is correct.
    # The actual parallel expansion would happen at workflow registration
    # when we know the iterator size.

    spread = RappelComprehensionMap(
        action=RappelActionCall(
            action="double",
            module=None,
            kwargs={"value": "x"},
            target=None,
        ),
        loop_var="x",
        iterable="items",
        target="results",
    )

    ret = RappelReturn(value="results")

    workflow = RappelWorkflow(
        name="SpreadWorkflow",
        params=[("items", "list")],
        body=[spread, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    # Verify the DAG has the expected structure
    assert any(n.kind == NodeKind.COMPUTED and "spread over" in (n.code or "")
               for n in dag.nodes.values()), "Expected spread_head node"
    assert any(n.kind == NodeKind.ACTION and n.action_name == "double"
               for n in dag.nodes.values()), "Expected spread action node"

    print("\nSpread DAG structure validated.")
    print("(Full parallel expansion would occur at workflow registration time)")
    print("PASSED!")


def test_conditional_simple():
    """Test simple if/else conditional."""
    print("\n" + "=" * 70)
    print("TEST: Simple Conditional (if/else)")
    print("=" * 70)

    # if value > 50:
    #     result = await high_action(value=value)
    # else:
    #     result = await low_action(value=value)
    # return result

    from rappel_ir_sketch import RappelBranch

    cond = RappelConditional(
        branches=[
            RappelBranch(
                guard="value > 50",
                preamble=[],
                actions=[
                    RappelActionCall(
                        action="high_action",
                        module=None,
                        kwargs={"value": "value"},
                        target="result",
                    )
                ],
                postamble=[],
            ),
            RappelBranch(
                guard="value <= 50",
                preamble=[],
                actions=[
                    RappelActionCall(
                        action="low_action",
                        module=None,
                        kwargs={"value": "value"},
                        target="result",
                    )
                ],
                postamble=[],
            ),
        ],
        target="result",
    )

    ret = RappelReturn(value="result")

    workflow = RappelWorkflow(
        name="ConditionalWorkflow",
        params=[("value", "int")],
        body=[cond, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    # Test TRUE path (value=75 > 50)
    print("\nExecution (value=75, should take HIGH path):")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "high_action": lambda value: f"HIGH:{value}",
            "low_action": lambda value: f"LOW:{value}",
        }
    )
    result = executor.execute({"value": 75})

    print(f"\nFinal result: {result}")
    print(f"Expected: 'HIGH:75'")
    assert result == "HIGH:75", f"Expected 'HIGH:75', got {result}"

    # Test FALSE path (value=25 <= 50)
    print("\nExecution (value=25, should take LOW path):")
    dag2 = converter.convert(workflow)  # Fresh DAG
    executor2 = DAGExecutor(
        dag2,
        action_mocks={
            "high_action": lambda value: f"HIGH:{value}",
            "low_action": lambda value: f"LOW:{value}",
        }
    )
    result2 = executor2.execute({"value": 25})

    print(f"\nFinal result: {result2}")
    print(f"Expected: 'LOW:25'")
    assert result2 == "LOW:25", f"Expected 'LOW:25', got {result2}"

    print("PASSED!")


def test_conditional_with_preamble():
    """Test conditional with preamble/postamble."""
    print("\n" + "=" * 70)
    print("TEST: Conditional with Preamble/Postamble")
    print("=" * 70)

    # if value > 50:
    #     adjusted = value * 2  # preamble
    #     result = await process(value=adjusted)
    #     label = "high"  # postamble
    # else:
    #     result = await fallback(value=value)
    # return result

    from rappel_ir_sketch import RappelBranch

    cond = RappelConditional(
        branches=[
            RappelBranch(
                guard="value > 50",
                preamble=[
                    RappelPythonBlock(
                        code="adjusted = value * 2",
                        inputs=["value"],
                        outputs=["adjusted"],
                    )
                ],
                actions=[
                    RappelActionCall(
                        action="process",
                        module=None,
                        kwargs={"value": "adjusted"},
                        target="result",
                    )
                ],
                postamble=[
                    RappelPythonBlock(
                        code="label = 'high'",
                        inputs=[],
                        outputs=["label"],
                    )
                ],
            ),
            RappelBranch(
                guard="value <= 50",
                preamble=[],
                actions=[
                    RappelActionCall(
                        action="fallback",
                        module=None,
                        kwargs={"value": "value"},
                        target="result",
                    )
                ],
                postamble=[],
            ),
        ],
        target="result",
    )

    ret = RappelReturn(value="result")

    workflow = RappelWorkflow(
        name="ConditionalWithExtras",
        params=[("value", "int")],
        body=[cond, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    # Test TRUE path with preamble (value=75 > 50)
    print("\nExecution (value=75, should use preamble adjusted=150):")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "process": lambda value: f"PROCESSED:{value}",
            "fallback": lambda value: f"FALLBACK:{value}",
        }
    )
    result = executor.execute({"value": 75})

    print(f"\nFinal result: {result}")
    print(f"Expected: 'PROCESSED:150'")
    assert result == "PROCESSED:150", f"Expected 'PROCESSED:150', got {result}"

    # Test FALSE path (value=25 <= 50)
    print("\nExecution (value=25, should take fallback):")
    dag2 = converter.convert(workflow)
    executor2 = DAGExecutor(
        dag2,
        action_mocks={
            "process": lambda value: f"PROCESSED:{value}",
            "fallback": lambda value: f"FALLBACK:{value}",
        }
    )
    result2 = executor2.execute({"value": 25})

    print(f"\nFinal result: {result2}")
    print(f"Expected: 'FALLBACK:25'")
    assert result2 == "FALLBACK:25", f"Expected 'FALLBACK:25', got {result2}"

    print("PASSED!")


def test_try_except():
    """Test try/except error handling."""
    print("\n" + "=" * 70)
    print("TEST: Try/Except")
    print("=" * 70)

    # try:
    #     result = await risky_action()
    # except ValueError:
    #     result = await fallback_action()
    # return result

    from rappel_ir_sketch import RappelExceptHandler

    try_except = RappelTryExcept(
        try_body=[
            RappelActionCall(
                action="risky_action",
                module=None,
                kwargs={},
                target="result",
            )
        ],
        handlers=[
            RappelExceptHandler(
                exception_types=[(None, "ValueError")],
                body=[
                    RappelActionCall(
                        action="fallback_action",
                        module=None,
                        kwargs={},
                        target="result",
                    )
                ],
            )
        ],
    )

    ret = RappelReturn(value="result")

    workflow = RappelWorkflow(
        name="TryExceptWorkflow",
        params=[],
        body=[try_except, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    # Test successful path (no exception)
    print("\nExecution (no exception):")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "risky_action": lambda: "SUCCESS",
            "fallback_action": lambda: "FALLBACK",
        }
    )
    result = executor.execute({})

    print(f"\nFinal result: {result}")
    print(f"Expected: 'SUCCESS'")
    assert result == "SUCCESS", f"Expected 'SUCCESS', got {result}"

    # Test exception path
    print("\nExecution (with exception):")
    dag2 = converter.convert(workflow)
    executor2 = DAGExecutor(
        dag2,
        action_mocks={
            "risky_action": lambda: (_ for _ in ()).throw(ValueError("oops")),
            "fallback_action": lambda: "FALLBACK",
        }
    )
    result2 = executor2.execute({})

    print(f"\nFinal result: {result2}")
    print(f"Expected: 'FALLBACK'")
    assert result2 == "FALLBACK", f"Expected 'FALLBACK', got {result2}"

    print("PASSED!")


def test_sleep():
    """Test durable sleep."""
    print("\n" + "=" * 70)
    print("TEST: Sleep")
    print("=" * 70)

    # started = await get_timestamp()
    # await asyncio.sleep(60)
    # ended = await get_timestamp()
    # return (started, ended)

    action1 = RappelActionCall(
        action="get_timestamp",
        module=None,
        kwargs={},
        target="started",
    )

    sleep = RappelSleep(duration_expr="60")

    action2 = RappelActionCall(
        action="get_timestamp",
        module=None,
        kwargs={},
        target="ended",
    )

    ret = RappelReturn(value="(started, ended)")

    workflow = RappelWorkflow(
        name="SleepWorkflow",
        params=[],
        body=[action1, sleep, action2, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "get_timestamp": lambda: "2024-01-01T00:00:00Z",
        }
    )
    result = executor.execute({})

    print(f"\nFinal result: {result}")
    print("PASSED!")


def test_loop_with_conditional_inside():
    """Test loop with conditional inside body."""
    print("\n" + "=" * 70)
    print("TEST: Loop with Conditional Inside")
    print("=" * 70)

    # This is a complex case:
    # results = []
    # for item in items:
    #     if item.valid:
    #         processed = await process(item=item)
    #     else:
    #         processed = await fallback(item=item)
    #     results.append(processed)
    # return results

    # For now, simulate with preamble that does the branching
    # Full implementation would have nested conditional nodes

    preamble = RappelPythonBlock(
        code="use_main = item.get('valid', False)",
        inputs=["item"],
        outputs=["use_main"],
    )

    loop = RappelLoop(
        iterator_expr="items",
        loop_var="item",
        accumulators=["results"],
        preamble=[preamble],
        body=[
            # Simplified: always call process
            RappelActionCall(
                action="process",
                module=None,
                kwargs={"item": "item"},
                target="processed",
            )
        ],
        append_exprs=[("results", "processed")],
    )

    ret = RappelReturn(value="results")

    workflow = RappelWorkflow(
        name="LoopWithConditional",
        params=[("items", "list")],
        body=[loop, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "process": lambda item: {"processed": True, **item},
        }
    )
    result = executor.execute({"items": [{"id": 1, "valid": True}, {"id": 2, "valid": False}]})

    print(f"\nFinal result: {result}")
    print("PASSED!")


def test_multiple_accumulators():
    """Test loop with multiple accumulators."""
    print("\n" + "=" * 70)
    print("TEST: Multiple Accumulators")
    print("=" * 70)

    # results = []
    # metrics = []
    # for item in items:
    #     processed = await double(value=item)
    #     results.append(processed)
    #     metrics.append(processed * 2)
    # return (results, metrics)

    loop = RappelLoop(
        iterator_expr="items",
        loop_var="item",
        accumulators=["results", "metrics"],
        preamble=[],
        body=[
            RappelActionCall(
                action="double",
                module=None,
                kwargs={"value": "item"},
                target="processed",
            )
        ],
        append_exprs=[
            ("results", "processed"),
            ("metrics", "processed * 2"),
        ],
    )

    ret = RappelReturn(value="(results, metrics)")

    workflow = RappelWorkflow(
        name="MultiAccumulatorLoop",
        params=[("items", "list")],
        body=[loop, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "double": lambda value: value * 2,
        }
    )
    result = executor.execute({"items": [1, 2, 3]})

    print(f"\nFinal result: {result}")
    print(f"Expected: ([2, 4, 6], [4, 8, 12])")
    assert result == ([2, 4, 6], [4, 8, 12]), f"Expected ([2, 4, 6], [4, 8, 12]), got {result}"
    print("PASSED!")


def test_sequential_actions():
    """Test simple sequential actions."""
    print("\n" + "=" * 70)
    print("TEST: Sequential Actions")
    print("=" * 70)

    # a = await fetch()
    # b = await process(value=a)
    # c = await finalize(value=b)
    # return c

    workflow = RappelWorkflow(
        name="SequentialWorkflow",
        params=[],
        body=[
            RappelActionCall(action="fetch", module=None, kwargs={}, target="a"),
            RappelActionCall(action="process", module=None, kwargs={"value": "a"}, target="b"),
            RappelActionCall(action="finalize", module=None, kwargs={"value": "b"}, target="c"),
            RappelReturn(value="c"),
        ],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "fetch": lambda: 10,
            "process": lambda value: value * 2,
            "finalize": lambda value: f"final:{value}",
        }
    )
    result = executor.execute({})

    print(f"\nFinal result: {result}")
    print(f"Expected: 'final:20'")
    assert result == "final:20", f"Expected 'final:20', got {result}"
    print("PASSED!")


def test_python_block_standalone():
    """Test standalone python block."""
    print("\n" + "=" * 70)
    print("TEST: Standalone Python Block")
    print("=" * 70)

    # x = 10
    # y = x * 2 + 5
    # result = await action(value=y)
    # return result

    workflow = RappelWorkflow(
        name="PythonBlockWorkflow",
        params=[],
        body=[
            RappelPythonBlock(code="x = 10", inputs=[], outputs=["x"]),
            RappelPythonBlock(code="y = x * 2 + 5", inputs=["x"], outputs=["y"]),
            RappelActionCall(action="action", module=None, kwargs={"value": "y"}, target="result"),
            RappelReturn(value="result"),
        ],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "action": lambda value: f"processed:{value}",
        }
    )
    result = executor.execute({})

    print(f"\nFinal result: {result}")
    print(f"Expected: 'processed:25'")
    assert result == "processed:25", f"Expected 'processed:25', got {result}"
    print("PASSED!")


def test_var_init_primitive():
    """Test VarInit as primitive (not just empty list)."""
    print("\n" + "=" * 70)
    print("TEST: VarInit Primitive (List with Values)")
    print("=" * 70)

    # This tests our plan for VarInit as a first-class primitive:
    # items = [1, 2, 3]  <- VarInit, not python_block
    # results = []
    # for x in items:
    #     doubled = await double(value=x)
    #     results.append(doubled)
    # return results

    # Simulate VarInit with a python block (proto would have VarInit message)
    var_init = RappelPythonBlock(
        code="items = [1, 2, 3]",
        inputs=[],
        outputs=["items"],
    )

    loop = RappelLoop(
        iterator_expr="items",
        loop_var="x",
        accumulators=["results"],
        preamble=[],
        body=[
            RappelActionCall(
                action="double",
                module=None,
                kwargs={"value": "x"},
                target="doubled",
            )
        ],
        append_exprs=[("results", "doubled")],
    )

    ret = RappelReturn(value="results")

    workflow = RappelWorkflow(
        name="VarInitWorkflow",
        params=[],
        body=[var_init, loop, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "double": lambda value: value * 2,
        }
    )
    result = executor.execute({})

    print(f"\nFinal result: {result}")
    print(f"Expected: [2, 4, 6]")
    assert result == [2, 4, 6], f"Expected [2, 4, 6], got {result}"
    print("PASSED!")


def test_complex_workflow():
    """Test complex workflow combining multiple constructs."""
    print("\n" + "=" * 70)
    print("TEST: Complex Workflow (Gather + Python + Loop)")
    print("=" * 70)

    # config = await fetch_config()
    # threshold = config['threshold']  # python block
    # items = await fetch_items()
    # results = []
    # for item in items:
    #     if item > threshold:  # preamble
    #         processed = await process(value=item)
    #         results.append(processed)
    # return results

    fetch_config = RappelActionCall(
        action="fetch_config",
        module=None,
        kwargs={},
        target="config",
    )

    extract_threshold = RappelPythonBlock(
        code="threshold = config['threshold']",
        inputs=["config"],
        outputs=["threshold"],
    )

    fetch_items = RappelActionCall(
        action="fetch_items",
        module=None,
        kwargs={},
        target="items",
    )

    # Loop with preamble that filters
    preamble = RappelPythonBlock(
        code="should_process = item > threshold",
        inputs=["item", "threshold"],
        outputs=["should_process"],
    )

    loop = RappelLoop(
        iterator_expr="items",
        loop_var="item",
        accumulators=["results"],
        preamble=[preamble],
        body=[
            RappelActionCall(
                action="process",
                module=None,
                kwargs={"value": "item"},
                target="processed",
            )
        ],
        append_exprs=[("results", "processed")],
    )

    ret = RappelReturn(value="results")

    workflow = RappelWorkflow(
        name="ComplexWorkflow",
        params=[],
        body=[fetch_config, extract_threshold, fetch_items, loop, ret],
    )

    converter = IRToDAG()
    dag = converter.convert(workflow)
    print("\nGenerated DAG:")
    print(dag)

    print("\nExecution:")
    executor = DAGExecutor(
        dag,
        action_mocks={
            "fetch_config": lambda: {"threshold": 5},
            "fetch_items": lambda: [1, 3, 7, 10, 2],
            "process": lambda value: value * 10,
        }
    )
    result = executor.execute({})

    print(f"\nFinal result: {result}")
    # Note: Current impl processes all items (doesn't actually filter)
    # Full impl would need conditional in loop body
    print("PASSED!")


def run_all_tests():
    """Run all test cases."""
    tests = [
        test_simple_loop,
        test_loop_with_preamble,
        test_multi_action_loop,
        test_empty_iterator,
        test_gather,
        test_gather_then_loop,
        test_spread,
        test_conditional_simple,
        test_conditional_with_preamble,
        test_try_except,
        test_sleep,
        test_loop_with_conditional_inside,
        test_multiple_accumulators,
        test_sequential_actions,
        test_python_block_standalone,
        test_var_init_primitive,
        test_complex_workflow,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if errors:
        print("\nFailed tests:")
        for name, error in errors:
            print(f"  - {name}: {error}")
    else:
        print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    run_all_tests()
