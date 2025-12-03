"""
Rappel Pretty Printer - Converts IR back to formatted source code.
"""

from __future__ import annotations

from .ir import (
    RappelString,
    RappelProgram,
    RappelAssignment,
    RappelMultiAssignment,
    RappelLiteral,
    RappelVariable,
    RappelBinaryOp,
    RappelUnaryOp,
    RappelListExpr,
    RappelDictExpr,
    RappelIndexAccess,
    RappelDotAccess,
    RappelSpread,
    RappelCall,
    RappelActionCall,
    RappelSpreadAction,
    RappelFunctionDef,
    RappelForLoop,
    RappelIfStatement,
    RappelReturn,
    RappelExprStatement,
    RappelExpr,
    RappelStatement,
)


class RappelPrettyPrinter:
    """Pretty printer for Rappel IR."""

    def __init__(self):
        self.indent = 0

    def print(self, node: RappelProgram | RappelStatement | RappelExpr) -> str:
        """Print a node as formatted string."""
        return self._print_node(node)

    def _print_node(self, node) -> str:
        """Print any node."""
        method_name = f"_print_{type(node).__name__}"
        method = getattr(self, method_name, self._print_generic)
        return method(node)

    def _print_generic(self, node) -> str:
        return f"<{type(node).__name__}>"

    def _print_RappelProgram(self, node: RappelProgram) -> str:
        lines = []
        for stmt in node.statements:
            lines.append(self._print_node(stmt))
        return "\n".join(lines)

    def _print_RappelAssignment(self, node: RappelAssignment) -> str:
        return f"{node.target} = {self._print_node(node.value)}"

    def _print_RappelMultiAssignment(self, node: RappelMultiAssignment) -> str:
        targets = ", ".join(node.targets)
        return f"{targets} = {self._print_node(node.value)}"

    def _print_RappelLiteral(self, node: RappelLiteral) -> str:
        if isinstance(node.value, RappelString):
            return f'"{node.value.value}"'
        return str(node.value.value)

    def _print_RappelVariable(self, node: RappelVariable) -> str:
        return node.name

    def _print_RappelBinaryOp(self, node: RappelBinaryOp) -> str:
        return f"({self._print_node(node.left)} {node.op} {self._print_node(node.right)})"

    def _print_RappelUnaryOp(self, node: RappelUnaryOp) -> str:
        return f"({node.op} {self._print_node(node.operand)})"

    def _print_RappelListExpr(self, node: RappelListExpr) -> str:
        items = ", ".join(self._print_node(item) for item in node.items)
        return f"[{items}]"

    def _print_RappelDictExpr(self, node: RappelDictExpr) -> str:
        pairs = ", ".join(
            f"{self._print_node(k)}: {self._print_node(v)}"
            for k, v in node.pairs
        )
        return "{" + pairs + "}"

    def _print_RappelIndexAccess(self, node: RappelIndexAccess) -> str:
        return f"{self._print_node(node.target)}[{self._print_node(node.index)}]"

    def _print_RappelDotAccess(self, node: RappelDotAccess) -> str:
        return f"{self._print_node(node.target)}.{node.field}"

    def _print_RappelSpread(self, node: RappelSpread) -> str:
        return f"...{self._print_node(node.target)}"

    def _print_RappelCall(self, node: RappelCall) -> str:
        kwargs = [f"{k}={self._print_node(v)}" for k, v in node.kwargs]
        return f"{node.target}({', '.join(kwargs)})"

    def _print_RappelActionCall(self, node: RappelActionCall) -> str:
        kwargs = [f"{k}={self._print_node(v)}" for k, v in node.kwargs]
        return f"@{node.action_name}({', '.join(kwargs)})"

    def _print_RappelSpreadAction(self, node: RappelSpreadAction) -> str:
        action = self._print_RappelActionCall(node.action)
        spread_part = f"spread {self._print_node(node.source_list)}:{node.item_var} -> {action}"
        if node.target:
            return f"{node.target} = {spread_part}"
        return spread_part

    def _print_RappelFunctionDef(self, node: RappelFunctionDef) -> str:
        inputs = ", ".join(node.inputs)
        outputs = ", ".join(node.outputs)
        body = self._print_block(node.body)
        return f"fn {node.name}(input: [{inputs}], output: [{outputs}]):\n{body}"

    def _print_RappelForLoop(self, node: RappelForLoop) -> str:
        iterable = self._print_node(node.iterable)
        body = self._print_block(node.body)
        loop_vars_str = ", ".join(node.loop_vars)
        return f"for {loop_vars_str} in {iterable}:\n{body}"

    def _print_RappelIfStatement(self, node: RappelIfStatement) -> str:
        cond = self._print_node(node.condition)
        then_body = self._print_block(node.then_body)
        result = f"if {cond}:\n{then_body}"
        if node.else_body:
            else_body = self._print_block(node.else_body)
            result += f"\nelse:\n{else_body}"
        return result

    def _print_RappelReturn(self, node: RappelReturn) -> str:
        if len(node.values) == 0:
            return "return"
        elif len(node.values) == 1:
            return f"return {self._print_node(node.values[0])}"
        else:
            vals = ", ".join(self._print_node(v) for v in node.values)
            return f"return [{vals}]"

    def _print_RappelExprStatement(self, node: RappelExprStatement) -> str:
        return self._print_node(node.expr)

    def _print_block(self, stmts: tuple) -> str:
        lines = []
        for stmt in stmts:
            line = self._print_node(stmt)
            for subline in line.split("\n"):
                lines.append("    " + subline)
        return "\n".join(lines)
