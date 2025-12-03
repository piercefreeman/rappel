"""
IR Builder - Converts Python workflow AST to Rappel IR (ast.proto).

This module parses Python workflow classes and produces the IR representation
that can be sent to the Rust runtime for execution.
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from proto import ast_pb2 as ir

if TYPE_CHECKING:
    from .workflow import Workflow


@dataclass
class ActionDefinition:
    """Definition of an action function."""

    action_name: str
    module_name: Optional[str]
    signature: inspect.Signature


def build_workflow_ir(workflow_cls: type["Workflow"]) -> ir.Program:
    """Build an IR Program from a workflow class.

    Args:
        workflow_cls: The workflow class to convert.

    Returns:
        An IR Program proto message.
    """
    original_run = getattr(workflow_cls, "__workflow_run_impl__", None)
    if original_run is None:
        original_run = workflow_cls.__dict__.get("run")
    if original_run is None:
        raise ValueError(f"workflow {workflow_cls!r} missing run implementation")

    module = inspect.getmodule(original_run)
    if module is None:
        raise ValueError(f"unable to locate module for workflow {workflow_cls!r}")

    # Get the function source and parse it
    function_source = textwrap.dedent(inspect.getsource(original_run))
    tree = ast.parse(function_source)

    # Discover actions in the module
    action_defs = _discover_action_names(module)

    # Build the IR
    builder = IRBuilder(action_defs)
    builder.visit(tree)

    # Create the Program with a single function definition
    program = ir.Program()
    if builder.function_def:
        program.functions.append(builder.function_def)

    return program


def _discover_action_names(module: Any) -> Dict[str, ActionDefinition]:
    """Discover all @action decorated functions in a module."""
    names: Dict[str, ActionDefinition] = {}
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        action_name = getattr(attr, "__rappel_action_name__", None)
        action_module = getattr(attr, "__rappel_action_module__", None)
        if callable(attr) and action_name:
            signature = inspect.signature(attr)
            names[attr_name] = ActionDefinition(
                action_name=action_name,
                module_name=action_module or module.__name__,
                signature=signature,
            )
    return names


class IRBuilder(ast.NodeVisitor):
    """Builds IR from Python AST."""

    def __init__(self, action_defs: Dict[str, ActionDefinition]):
        self._action_defs = action_defs
        self.function_def: Optional[ir.FunctionDef] = None
        self._statements: List[ir.Statement] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Visit a function definition (the workflow's run method)."""
        # Extract inputs from function parameters (skip 'self')
        inputs: List[str] = []
        for arg in node.args.args[1:]:  # Skip 'self'
            inputs.append(arg.arg)

        # Create the function definition
        self.function_def = ir.FunctionDef(
            name=node.name,
            io=ir.IoDecl(inputs=inputs, outputs=[]),
            span=_make_span(node),
        )

        # Visit the body
        self._statements = []
        for stmt in node.body:
            ir_stmt = self._visit_statement(stmt)
            if ir_stmt:
                self._statements.append(ir_stmt)

        # Set the body
        self.function_def.body.CopyFrom(ir.Block(statements=self._statements))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """Visit an async function definition (the workflow's run method)."""
        # Handle async the same way as sync for IR building
        inputs: List[str] = []
        for arg in node.args.args[1:]:  # Skip 'self'
            inputs.append(arg.arg)

        self.function_def = ir.FunctionDef(
            name=node.name,
            io=ir.IoDecl(inputs=inputs, outputs=[]),
            span=_make_span(node),
        )

        self._statements = []
        for stmt in node.body:
            ir_stmt = self._visit_statement(stmt)
            if ir_stmt:
                self._statements.append(ir_stmt)

        self.function_def.body.CopyFrom(ir.Block(statements=self._statements))

    def _visit_statement(self, node: ast.stmt) -> Optional[ir.Statement]:
        """Convert a Python statement to IR Statement."""
        if isinstance(node, ast.Assign):
            return self._visit_assign(node)
        elif isinstance(node, ast.Expr):
            return self._visit_expr_stmt(node)
        elif isinstance(node, ast.For):
            return self._visit_for(node)
        elif isinstance(node, ast.If):
            return self._visit_if(node)
        elif isinstance(node, ast.Try):
            return self._visit_try(node)
        elif isinstance(node, ast.Return):
            return self._visit_return(node)
        elif isinstance(node, ast.AugAssign):
            # Convert augmented assignment to regular assignment with binary op
            return self._visit_aug_assign(node)

        return None

    def _visit_assign(self, node: ast.Assign) -> Optional[ir.Statement]:
        """Convert assignment to IR."""
        stmt = ir.Statement(span=_make_span(node))

        # Check if this is an action call
        action_call = self._extract_action_call(node.value)
        if action_call:
            # Get target variable name
            target = self._get_assign_target(node.targets)
            if target:
                action_call.target = target
            stmt.action_call.CopyFrom(action_call)
            return stmt

        # Regular assignment
        targets: List[str] = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                targets.append(t.id)
            elif isinstance(t, ast.Tuple):
                for elt in t.elts:
                    if isinstance(elt, ast.Name):
                        targets.append(elt.id)

        value_expr = _expr_to_ir(node.value)
        if value_expr:
            assign = ir.Assignment(targets=targets, value=value_expr)
            stmt.assignment.CopyFrom(assign)
            return stmt

        return None

    def _visit_expr_stmt(self, node: ast.Expr) -> Optional[ir.Statement]:
        """Convert expression statement to IR."""
        stmt = ir.Statement(span=_make_span(node))

        # Check if this is an action call
        action_call = self._extract_action_call(node.value)
        if action_call:
            stmt.action_call.CopyFrom(action_call)
            return stmt

        # Regular expression
        expr = _expr_to_ir(node.value)
        if expr:
            stmt.expr_stmt.CopyFrom(ir.ExprStmt(expr=expr))
            return stmt

        return None

    def _visit_for(self, node: ast.For) -> Optional[ir.Statement]:
        """Convert for loop to IR."""
        stmt = ir.Statement(span=_make_span(node))

        # Get loop variables
        loop_vars: List[str] = []
        if isinstance(node.target, ast.Name):
            loop_vars.append(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    loop_vars.append(elt.id)

        # Get iterable
        iterable = _expr_to_ir(node.iter)
        if not iterable:
            return None

        # Build body
        body_stmts: List[ir.Statement] = []
        for body_node in node.body:
            body_stmt = self._visit_statement(body_node)
            if body_stmt:
                body_stmts.append(body_stmt)

        for_loop = ir.ForLoop(
            loop_vars=loop_vars,
            iterable=iterable,
            body=ir.Block(statements=body_stmts),
        )
        stmt.for_loop.CopyFrom(for_loop)
        return stmt

    def _visit_if(self, node: ast.If) -> Optional[ir.Statement]:
        """Convert if statement to IR conditional."""
        stmt = ir.Statement(span=_make_span(node))

        # Build if branch
        condition = _expr_to_ir(node.test)
        if not condition:
            return None

        body_stmts: List[ir.Statement] = []
        for body_node in node.body:
            body_stmt = self._visit_statement(body_node)
            if body_stmt:
                body_stmts.append(body_stmt)

        if_branch = ir.IfBranch(
            condition=condition,
            body=ir.Block(statements=body_stmts),
            span=_make_span(node),
        )

        conditional = ir.Conditional(if_branch=if_branch)

        # Handle elif/else chains
        current = node
        while current.orelse:
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                # elif branch
                elif_node = current.orelse[0]
                elif_condition = _expr_to_ir(elif_node.test)
                if elif_condition:
                    elif_body: List[ir.Statement] = []
                    for body_node in elif_node.body:
                        body_stmt = self._visit_statement(body_node)
                        if body_stmt:
                            elif_body.append(body_stmt)
                    elif_branch = ir.ElifBranch(
                        condition=elif_condition,
                        body=ir.Block(statements=elif_body),
                        span=_make_span(elif_node),
                    )
                    conditional.elif_branches.append(elif_branch)
                current = elif_node
            else:
                # else branch
                else_body: List[ir.Statement] = []
                for else_node in current.orelse:
                    else_stmt = self._visit_statement(else_node)
                    if else_stmt:
                        else_body.append(else_stmt)
                else_branch = ir.ElseBranch(
                    body=ir.Block(statements=else_body),
                    span=_make_span(current.orelse[0]) if current.orelse else None,
                )
                conditional.else_branch.CopyFrom(else_branch)
                break

        stmt.conditional.CopyFrom(conditional)
        return stmt

    def _visit_try(self, node: ast.Try) -> Optional[ir.Statement]:
        """Convert try/except to IR."""
        stmt = ir.Statement(span=_make_span(node))

        # Build try body
        try_body: List[ir.Statement] = []
        for body_node in node.body:
            body_stmt = self._visit_statement(body_node)
            if body_stmt:
                try_body.append(body_stmt)

        try_except = ir.TryExcept(try_body=ir.Block(statements=try_body))

        # Build handlers
        for handler in node.handlers:
            exception_types: List[str] = []
            if handler.type:
                if isinstance(handler.type, ast.Name):
                    exception_types.append(handler.type.id)
                elif isinstance(handler.type, ast.Tuple):
                    for elt in handler.type.elts:
                        if isinstance(elt, ast.Name):
                            exception_types.append(elt.id)

            handler_body: List[ir.Statement] = []
            for handler_node in handler.body:
                handler_stmt = self._visit_statement(handler_node)
                if handler_stmt:
                    handler_body.append(handler_stmt)

            except_handler = ir.ExceptHandler(
                exception_types=exception_types,
                body=ir.Block(statements=handler_body),
                span=_make_span(handler),
            )
            try_except.handlers.append(except_handler)

        stmt.try_except.CopyFrom(try_except)
        return stmt

    def _visit_return(self, node: ast.Return) -> Optional[ir.Statement]:
        """Convert return statement to IR."""
        stmt = ir.Statement(span=_make_span(node))

        if node.value:
            # Check if returning an action call
            action_call = self._extract_action_call(node.value)
            if action_call:
                # Return with action call
                return_stmt = ir.ReturnStmt()
                # Action calls in returns need special handling
                expr = _expr_to_ir(node.value)
                if expr:
                    return_stmt.value.CopyFrom(expr)
                stmt.return_stmt.CopyFrom(return_stmt)
                return stmt

            expr = _expr_to_ir(node.value)
            if expr:
                return_stmt = ir.ReturnStmt(value=expr)
                stmt.return_stmt.CopyFrom(return_stmt)
        else:
            stmt.return_stmt.CopyFrom(ir.ReturnStmt())

        return stmt

    def _visit_aug_assign(self, node: ast.AugAssign) -> Optional[ir.Statement]:
        """Convert augmented assignment (+=, -=, etc.) to IR."""
        # For now, we can represent this as a regular assignment with binary op
        # target op= value  ->  target = target op value
        stmt = ir.Statement(span=_make_span(node))

        targets: List[str] = []
        if isinstance(node.target, ast.Name):
            targets.append(node.target.id)

        left = _expr_to_ir(node.target)
        right = _expr_to_ir(node.value)
        if left and right:
            op = _bin_op_to_ir(node.op)
            if op:
                binary = ir.BinaryOp(left=left, op=op, right=right)
                value = ir.Expr(binary_op=binary)
                assign = ir.Assignment(targets=targets, value=value)
                stmt.assignment.CopyFrom(assign)
                return stmt

        return None

    def _extract_action_call(self, node: ast.expr) -> Optional[ir.ActionCall]:
        """Extract an action call from an expression if present."""
        if not isinstance(node, ast.Await):
            return None

        awaited = node.value
        # Handle self.run_action(...) wrapper
        if isinstance(awaited, ast.Call):
            if self._is_run_action_call(awaited):
                # Extract the actual action call from run_action
                if awaited.args:
                    return self._extract_action_call_from_awaitable(awaited.args[0])
            # Direct action call
            return self._extract_action_call_from_call(awaited)

        return None

    def _is_run_action_call(self, node: ast.Call) -> bool:
        """Check if this is a self.run_action(...) call."""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "run_action"
        return False

    def _extract_action_call_from_awaitable(self, node: ast.expr) -> Optional[ir.ActionCall]:
        """Extract action call from an awaitable expression."""
        if isinstance(node, ast.Call):
            return self._extract_action_call_from_call(node)
        return None

    def _extract_action_call_from_call(self, node: ast.Call) -> Optional[ir.ActionCall]:
        """Extract action call info from a Call node."""
        action_name = self._get_action_name(node.func)
        if not action_name:
            return None

        if action_name not in self._action_defs:
            return None

        action_def = self._action_defs[action_name]
        action_call = ir.ActionCall(action_name=action_def.action_name)

        # Add kwargs
        for kw in node.keywords:
            if kw.arg:
                expr = _expr_to_ir(kw.value)
                if expr:
                    kwarg = ir.Kwarg(name=kw.arg, value=expr)
                    action_call.kwargs.append(kwarg)

        return action_call

    def _get_action_name(self, func: ast.expr) -> Optional[str]:
        """Get the action name from a function expression."""
        if isinstance(func, ast.Name):
            return func.id
        elif isinstance(func, ast.Attribute):
            return func.attr
        return None

    def _get_assign_target(self, targets: List[ast.expr]) -> Optional[str]:
        """Get the target variable name from assignment targets."""
        if targets and isinstance(targets[0], ast.Name):
            return targets[0].id
        return None


def _make_span(node: ast.AST) -> ir.Span:
    """Create a Span from an AST node."""
    return ir.Span(
        start_line=getattr(node, "lineno", 0),
        start_col=getattr(node, "col_offset", 0),
        end_line=getattr(node, "end_lineno", 0) or 0,
        end_col=getattr(node, "end_col_offset", 0) or 0,
    )


def _expr_to_ir(expr: ast.AST) -> Optional[ir.Expr]:
    """Convert Python AST expression to IR Expr."""
    result = ir.Expr(span=_make_span(expr))

    if isinstance(expr, ast.Name):
        result.variable.CopyFrom(ir.Variable(name=expr.id))
        return result

    if isinstance(expr, ast.Constant):
        literal = _constant_to_literal(expr.value)
        if literal:
            result.literal.CopyFrom(literal)
            return result

    if isinstance(expr, ast.BinOp):
        left = _expr_to_ir(expr.left)
        right = _expr_to_ir(expr.right)
        op = _bin_op_to_ir(expr.op)
        if left and right and op:
            result.binary_op.CopyFrom(ir.BinaryOp(left=left, op=op, right=right))
            return result

    if isinstance(expr, ast.UnaryOp):
        operand = _expr_to_ir(expr.operand)
        op = _unary_op_to_ir(expr.op)
        if operand and op:
            result.unary_op.CopyFrom(ir.UnaryOp(op=op, operand=operand))
            return result

    if isinstance(expr, ast.Compare):
        left = _expr_to_ir(expr.left)
        if not left:
            return None
        # For simplicity, handle single comparison
        if expr.ops and expr.comparators:
            op = _cmp_op_to_ir(expr.ops[0])
            right = _expr_to_ir(expr.comparators[0])
            if op and right:
                result.binary_op.CopyFrom(ir.BinaryOp(left=left, op=op, right=right))
                return result

    if isinstance(expr, ast.BoolOp):
        values = [_expr_to_ir(v) for v in expr.values]
        if all(v for v in values):
            op = _bool_op_to_ir(expr.op)
            if op and len(values) >= 2:
                # Chain boolean ops: a and b and c -> (a and b) and c
                result_expr = values[0]
                for v in values[1:]:
                    if result_expr and v:
                        new_result = ir.Expr()
                        new_result.binary_op.CopyFrom(ir.BinaryOp(left=result_expr, op=op, right=v))
                        result_expr = new_result
                return result_expr

    if isinstance(expr, ast.List):
        elements = [_expr_to_ir(e) for e in expr.elts]
        if all(e for e in elements):
            list_expr = ir.ListExpr(elements=[e for e in elements if e])
            result.list.CopyFrom(list_expr)
            return result

    if isinstance(expr, ast.Dict):
        entries: List[ir.DictEntry] = []
        for k, v in zip(expr.keys, expr.values, strict=False):
            if k:
                key_expr = _expr_to_ir(k)
                value_expr = _expr_to_ir(v)
                if key_expr and value_expr:
                    entries.append(ir.DictEntry(key=key_expr, value=value_expr))
        result.dict.CopyFrom(ir.DictExpr(entries=entries))
        return result

    if isinstance(expr, ast.Subscript):
        obj = _expr_to_ir(expr.value)
        index = _expr_to_ir(expr.slice) if isinstance(expr.slice, ast.AST) else None
        if obj and index:
            result.index.CopyFrom(ir.IndexAccess(object=obj, index=index))
            return result

    if isinstance(expr, ast.Attribute):
        obj = _expr_to_ir(expr.value)
        if obj:
            result.dot.CopyFrom(ir.DotAccess(object=obj, attribute=expr.attr))
            return result

    if isinstance(expr, ast.Call):
        # Function call
        func_name = _get_func_name(expr.func)
        if func_name:
            args = [_expr_to_ir(a) for a in expr.args]
            kwargs: List[ir.Kwarg] = []
            for kw in expr.keywords:
                if kw.arg:
                    kw_expr = _expr_to_ir(kw.value)
                    if kw_expr:
                        kwargs.append(ir.Kwarg(name=kw.arg, value=kw_expr))
            func_call = ir.FunctionCall(
                name=func_name,
                args=[a for a in args if a],
                kwargs=kwargs,
            )
            result.function_call.CopyFrom(func_call)
            return result

    if isinstance(expr, ast.Tuple):
        # Handle tuple as list for now
        elements = [_expr_to_ir(e) for e in expr.elts]
        if all(e for e in elements):
            list_expr = ir.ListExpr(elements=[e for e in elements if e])
            result.list.CopyFrom(list_expr)
            return result

    return None


def _constant_to_literal(value: Any) -> Optional[ir.Literal]:
    """Convert a Python constant to IR Literal."""
    literal = ir.Literal()
    if value is None:
        literal.is_none = True
    elif isinstance(value, bool):
        literal.bool_value = value
    elif isinstance(value, int):
        literal.int_value = value
    elif isinstance(value, float):
        literal.float_value = value
    elif isinstance(value, str):
        literal.string_value = value
    else:
        return None
    return literal


def _bin_op_to_ir(op: ast.operator) -> Optional[ir.BinaryOperator]:
    """Convert Python binary operator to IR BinaryOperator."""
    mapping = {
        ast.Add: ir.BinaryOperator.BINARY_OP_ADD,
        ast.Sub: ir.BinaryOperator.BINARY_OP_SUB,
        ast.Mult: ir.BinaryOperator.BINARY_OP_MUL,
        ast.Div: ir.BinaryOperator.BINARY_OP_DIV,
        ast.FloorDiv: ir.BinaryOperator.BINARY_OP_FLOOR_DIV,
        ast.Mod: ir.BinaryOperator.BINARY_OP_MOD,
    }
    return mapping.get(type(op))


def _unary_op_to_ir(op: ast.unaryop) -> Optional[ir.UnaryOperator]:
    """Convert Python unary operator to IR UnaryOperator."""
    mapping = {
        ast.USub: ir.UnaryOperator.UNARY_OP_NEG,
        ast.Not: ir.UnaryOperator.UNARY_OP_NOT,
    }
    return mapping.get(type(op))


def _cmp_op_to_ir(op: ast.cmpop) -> Optional[ir.BinaryOperator]:
    """Convert Python comparison operator to IR BinaryOperator."""
    mapping = {
        ast.Eq: ir.BinaryOperator.BINARY_OP_EQ,
        ast.NotEq: ir.BinaryOperator.BINARY_OP_NE,
        ast.Lt: ir.BinaryOperator.BINARY_OP_LT,
        ast.LtE: ir.BinaryOperator.BINARY_OP_LE,
        ast.Gt: ir.BinaryOperator.BINARY_OP_GT,
        ast.GtE: ir.BinaryOperator.BINARY_OP_GE,
        ast.In: ir.BinaryOperator.BINARY_OP_IN,
        ast.NotIn: ir.BinaryOperator.BINARY_OP_NOT_IN,
    }
    return mapping.get(type(op))


def _bool_op_to_ir(op: ast.boolop) -> Optional[ir.BinaryOperator]:
    """Convert Python boolean operator to IR BinaryOperator."""
    mapping = {
        ast.And: ir.BinaryOperator.BINARY_OP_AND,
        ast.Or: ir.BinaryOperator.BINARY_OP_OR,
    }
    return mapping.get(type(op))


def _get_func_name(func: ast.expr) -> Optional[str]:
    """Get function name from a Call's func attribute."""
    if isinstance(func, ast.Name):
        return func.id
    elif isinstance(func, ast.Attribute):
        # For method calls like obj.method, return full dotted name
        parts = []
        current = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    return None
