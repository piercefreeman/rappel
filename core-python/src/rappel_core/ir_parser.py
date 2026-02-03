"""Parser for the IR source-like format."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from proto import ast_pb2 as ir

from .dag import assert_never


class IRParseError(Exception):
    """Raised when parsing the IR source representation fails."""


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


class _Tokenizer:
    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        self._length = len(text)

    def __iter__(self) -> Iterator[_Token]:
        while True:
            token = self._next_token()
            yield token
            if token.kind == "EOF":
                return

    def _next_token(self) -> _Token:
        self._skip_whitespace()
        if self._pos >= self._length:
            return _Token("EOF", "", self._pos)

        ch = self._text[self._pos]
        if ch.isalpha() or ch == "_":
            start = self._pos
            self._pos += 1
            while self._pos < self._length:
                ch = self._text[self._pos]
                if ch.isalnum() or ch == "_":
                    self._pos += 1
                else:
                    break
            value = self._text[start : self._pos]
            return _Token("NAME", value, start)

        if ch.isdigit():
            start = self._pos
            self._pos += 1
            while self._pos < self._length and self._text[self._pos].isdigit():
                self._pos += 1
            if self._pos < self._length and self._text[self._pos] == ".":
                self._pos += 1
                while self._pos < self._length and self._text[self._pos].isdigit():
                    self._pos += 1
            if self._pos < self._length and self._text[self._pos] in {"e", "E"}:
                self._pos += 1
                if self._pos < self._length and self._text[self._pos] in {"+", "-"}:
                    self._pos += 1
                while self._pos < self._length and self._text[self._pos].isdigit():
                    self._pos += 1
            value = self._text[start : self._pos]
            return _Token("NUMBER", value, start)

        if ch in {"'", '"'}:
            start = self._pos
            quote = ch
            self._pos += 1
            escaped = False
            while self._pos < self._length:
                ch = self._text[self._pos]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    self._pos += 1
                    break
                self._pos += 1
            value = self._text[start : self._pos]
            return _Token("STRING", value, start)

        two_char = {
            "==",
            "!=",
            ">=",
            "<=",
            "//",
            "->",
        }
        if self._pos + 1 < self._length:
            pair = self._text[self._pos : self._pos + 2]
            if pair in two_char:
                self._pos += 2
                return _Token(pair, pair, self._pos - 2)

        if ch in "()[]{}.,:+-*/%<>=@":
            self._pos += 1
            return _Token(ch, ch, self._pos - 1)

        raise IRParseError(f"Unexpected character '{ch}'")

    def _skip_whitespace(self) -> None:
        while self._pos < self._length and self._text[self._pos].isspace():
            self._pos += 1


class _TokenStream:
    def __init__(self, tokens: Iterable[_Token]) -> None:
        self._tokens = list(tokens)
        self._index = 0

    def peek(self) -> _Token:
        return self._tokens[self._index]

    def peek_next(self) -> _Token:
        if self._index + 1 < len(self._tokens):
            return self._tokens[self._index + 1]
        return _Token("EOF", "", self._tokens[-1].position if self._tokens else 0)

    def advance(self) -> _Token:
        token = self._tokens[self._index]
        if self._index < len(self._tokens) - 1:
            self._index += 1
        return token

    def expect(self, kind: str, value: Optional[str] = None) -> _Token:
        token = self.peek()
        if token.kind != kind:
            raise IRParseError(f"Expected {kind}, found {token.kind}")
        if value is not None and token.value != value:
            raise IRParseError(f"Expected {value}, found {token.value}")
        return self.advance()

    def match(self, kind: str, value: Optional[str] = None) -> bool:
        token = self.peek()
        if token.kind != kind:
            return False
        if value is not None and token.value != value:
            return False
        self.advance()
        return True


class _ExprParser:
    def __init__(self, text: str) -> None:
        tokens = list(_Tokenizer(text))
        self._stream = _TokenStream(tokens)

    def parse(self) -> ir.Expr:
        expr = self._parse_expr(0)
        if self._stream.peek().kind != "EOF":
            raise IRParseError("Unexpected tokens at end of expression")
        return expr

    def _parse_expr(self, min_prec: int) -> ir.Expr:
        left = self._parse_unary()

        while True:
            op_info = self._peek_binary_op()
            if op_info is None:
                break
            op_str, prec = op_info
            if prec < min_prec:
                break
            if op_str == "not in":
                self._stream.expect("NAME", "not")
                self._stream.expect("NAME", "in")
            else:
                token = self._stream.advance()
                op_str = token.value

            right = self._parse_expr(prec + 1)
            left = ir.Expr(
                binary_op=ir.BinaryOp(
                    left=left,
                    op=self._binary_operator(op_str),
                    right=right,
                )
            )

        return left

    def _parse_unary(self) -> ir.Expr:
        token = self._stream.peek()
        if token.kind == "NAME" and token.value == "not":
            self._stream.advance()
            operand = self._parse_expr(60)
            return ir.Expr(
                unary_op=ir.UnaryOp(
                    op=ir.UnaryOperator.UNARY_OP_NOT,
                    operand=operand,
                )
            )
        if token.kind == "-":
            self._stream.advance()
            operand = self._parse_expr(60)
            return ir.Expr(
                unary_op=ir.UnaryOp(
                    op=ir.UnaryOperator.UNARY_OP_NEG,
                    operand=operand,
                )
            )
        return self._parse_postfix()

    def _parse_postfix(self) -> ir.Expr:
        expr = self._parse_primary()
        while True:
            if self._stream.match("."):
                attr = self._stream.expect("NAME")
                expr = ir.Expr(dot=ir.DotAccess(object=expr, attribute=attr.value))
                continue
            if self._stream.match("["):
                index_expr = self._parse_expr(0)
                self._stream.expect("]")
                expr = ir.Expr(index=ir.IndexAccess(object=expr, index=index_expr))
                continue
            break
        return expr

    def _parse_primary(self) -> ir.Expr:
        token = self._stream.peek()
        if token.kind == "NUMBER":
            self._stream.advance()
            if any(char in token.value for char in ".eE"):
                return ir.Expr(literal=ir.Literal(float_value=float(token.value)))
            return ir.Expr(literal=ir.Literal(int_value=int(token.value)))

        if token.kind == "STRING":
            self._stream.advance()
            value = _decode_string(token.value)
            return ir.Expr(literal=ir.Literal(string_value=value))

        if token.kind == "NAME":
            if token.value == "True":
                self._stream.advance()
                return ir.Expr(literal=ir.Literal(bool_value=True))
            if token.value == "False":
                self._stream.advance()
                return ir.Expr(literal=ir.Literal(bool_value=False))
            if token.value == "None":
                self._stream.advance()
                return ir.Expr(literal=ir.Literal(is_none=True))
            if token.value == "spread":
                return self._parse_spread_expr()
            if token.value == "parallel" and self._stream.peek_next().kind == "(":
                return self._parse_parallel_expr()
            name = self._stream.advance().value
            if self._stream.peek().kind == "(":
                return ir.Expr(function_call=self._parse_function_call(name))
            return ir.Expr(variable=ir.Variable(name=name))

        if token.kind == "@":
            action = self._parse_action_call()
            return ir.Expr(action_call=action)

        if token.kind == "(":
            self._stream.advance()
            expr = self._parse_expr(0)
            self._stream.expect(")")
            return expr

        if token.kind == "[":
            return self._parse_list()

        if token.kind == "{":
            return self._parse_dict()

        raise IRParseError(f"Unexpected token {token.kind}")

    def _parse_list(self) -> ir.Expr:
        self._stream.expect("[")
        elements: list[ir.Expr] = []
        if self._stream.match("]"):
            return ir.Expr(list=ir.ListExpr(elements=elements))
        while True:
            elements.append(self._parse_expr(0))
            if self._stream.match(","):
                if self._stream.peek().kind == "]":
                    break
                continue
            break
        self._stream.expect("]")
        return ir.Expr(list=ir.ListExpr(elements=elements))

    def _parse_dict(self) -> ir.Expr:
        self._stream.expect("{")
        entries: list[ir.DictEntry] = []
        if self._stream.match("}"):
            return ir.Expr(dict=ir.DictExpr(entries=entries))
        while True:
            key = self._parse_expr(0)
            self._stream.expect(":")
            value = self._parse_expr(0)
            entries.append(ir.DictEntry(key=key, value=value))
            if self._stream.match(","):
                if self._stream.peek().kind == "}":
                    break
                continue
            break
        self._stream.expect("}")
        return ir.Expr(dict=ir.DictExpr(entries=entries))

    def _parse_function_call(self, name: str) -> ir.FunctionCall:
        self._stream.expect("(")
        args: list[ir.Expr] = []
        kwargs: list[ir.Kwarg] = []
        if self._stream.match(")"):
            return self._build_function_call(name, args, kwargs)
        while True:
            token = self._stream.peek()
            if token.kind == "NAME" and self._stream.peek_next().kind == "=":
                key = self._stream.advance().value
                self._stream.expect("=")
                value = self._parse_expr(0)
                kwargs.append(ir.Kwarg(name=key, value=value))
            else:
                args.append(self._parse_expr(0))
            if self._stream.match(","):
                if self._stream.peek().kind == ")":
                    break
                continue
            break
        self._stream.expect(")")
        return self._build_function_call(name, args, kwargs)

    def _build_function_call(
        self, name: str, args: list[ir.Expr], kwargs: list[ir.Kwarg]
    ) -> ir.FunctionCall:
        global_map = {
            "range": ir.GlobalFunction.GLOBAL_FUNCTION_RANGE,
            "len": ir.GlobalFunction.GLOBAL_FUNCTION_LEN,
            "enumerate": ir.GlobalFunction.GLOBAL_FUNCTION_ENUMERATE,
            "isexception": ir.GlobalFunction.GLOBAL_FUNCTION_ISEXCEPTION,
        }
        if name in global_map:
            return ir.FunctionCall(global_function=global_map[name], args=args, kwargs=kwargs)
        return ir.FunctionCall(name=name, args=args, kwargs=kwargs)

    def _parse_parallel_expr(self) -> ir.Expr:
        self._stream.expect("NAME", "parallel")
        self._stream.expect("(")
        calls: list[ir.Call] = []
        if self._stream.match(")"):
            return ir.Expr(parallel_expr=ir.ParallelExpr(calls=calls))
        while True:
            calls.append(self._parse_call())
            if self._stream.match(","):
                if self._stream.peek().kind == ")":
                    break
                continue
            break
        self._stream.expect(")")
        return ir.Expr(parallel_expr=ir.ParallelExpr(calls=calls))

    def _parse_call(self) -> ir.Call:
        token = self._stream.peek()
        if token.kind == "@":
            action = self._parse_action_call()
            return ir.Call(action=action)
        if token.kind == "NAME" and self._stream.peek_next().kind == "(":
            name = self._stream.advance().value
            func = self._parse_function_call(name)
            return ir.Call(function=func)
        raise IRParseError("Expected action or function call")

    def _parse_action_call(self) -> ir.ActionCall:
        self._stream.expect("@")
        name_parts: list[str] = []
        name_parts.append(self._stream.expect("NAME").value)
        while self._stream.match("."):
            name_parts.append(self._stream.expect("NAME").value)
        if not name_parts:
            raise IRParseError("Action call missing name")
        if len(name_parts) == 1:
            module_name = None
            action_name = name_parts[0]
        else:
            module_name = ".".join(name_parts[:-1])
            action_name = name_parts[-1]

        self._stream.expect("(")
        kwargs: list[ir.Kwarg] = []
        if not self._stream.match(")"):
            while True:
                token = self._stream.peek()
                if token.kind != "NAME" or self._stream.peek_next().kind != "=":
                    raise IRParseError("Action calls require keyword arguments")
                key = self._stream.advance().value
                self._stream.expect("=")
                value = self._parse_expr(0)
                kwargs.append(ir.Kwarg(name=key, value=value))
                if self._stream.match(","):
                    if self._stream.peek().kind == ")":
                        break
                    continue
                break
            self._stream.expect(")")

        policies: list[ir.PolicyBracket] = []
        while self._stream.peek().kind == "[":
            policies.append(self._parse_policy())

        action = ir.ActionCall(action_name=action_name, kwargs=kwargs, policies=policies)
        if module_name:
            action.module_name = module_name
        return action

    def _parse_policy(self) -> ir.PolicyBracket:
        self._stream.expect("[")
        exception_types: list[str] = []
        if self._has_exception_header():
            while True:
                exception_types.append(self._stream.expect("NAME").value)
                if self._stream.match(","):
                    continue
                break
            self._stream.expect("->")

        kind_token = self._stream.expect("NAME")
        kind = kind_token.value
        self._stream.expect(":")
        if kind == "retry":
            max_retries = self._parse_int_value()
            backoff: ir.Duration | None = None
            if self._stream.match(","):
                key = self._stream.expect("NAME").value
                self._stream.expect(":")
                if key != "backoff":
                    raise IRParseError(f"Unsupported retry policy field: {key}")
                backoff = self._parse_duration()
            self._stream.expect("]")
            policy = ir.RetryPolicy(max_retries=max_retries)
            if exception_types:
                policy.exception_types.extend(exception_types)
            if backoff is not None:
                policy.backoff.CopyFrom(backoff)
            return ir.PolicyBracket(retry=policy)

        if kind == "timeout":
            if exception_types:
                raise IRParseError("Timeout policy cannot specify exception types")
            duration = self._parse_duration()
            self._stream.expect("]")
            return ir.PolicyBracket(timeout=ir.TimeoutPolicy(timeout=duration))

        raise IRParseError(f"Unsupported policy kind: {kind}")

    def _parse_int_value(self) -> int:
        token = self._stream.expect("NUMBER")
        if any(char in token.value for char in ".eE"):
            raise IRParseError("Expected integer value")
        return int(token.value)

    def _parse_duration(self) -> ir.Duration:
        value = self._parse_int_value()
        unit = "s"
        if self._stream.peek().kind == "NAME" and self._stream.peek().value in {"s", "m", "h"}:
            unit = self._stream.advance().value
        seconds = value
        if unit == "m":
            seconds *= 60
        elif unit == "h":
            seconds *= 3600
        return ir.Duration(seconds=seconds)

    def _has_exception_header(self) -> bool:
        idx = self._stream._index
        while idx < len(self._stream._tokens):
            token = self._stream._tokens[idx]
            if token.kind in {"]", "EOF"}:
                return False
            if token.kind == ":":
                return False
            if token.kind == "->":
                return True
            idx += 1
        return False

    def _parse_spread_expr(self) -> ir.Expr:
        self._stream.expect("NAME", "spread")
        collection = self._parse_expr(0)
        self._stream.expect(":")
        loop_var = self._stream.expect("NAME").value
        self._stream.expect("->")
        action = self._parse_action_call()
        return ir.Expr(
            spread_expr=ir.SpreadExpr(collection=collection, loop_var=loop_var, action=action)
        )

    def _peek_binary_op(self) -> Optional[tuple[str, int]]:
        token = self._stream.peek()
        if token.kind == "NAME":
            if token.value == "or":
                return ("or", 10)
            if token.value == "and":
                return ("and", 20)
            if token.value == "in":
                return ("in", 30)
            if token.value == "not" and self._stream.peek_next().value == "in":
                return ("not in", 30)
        if token.kind in {"==", "!=", "<", "<=", ">", ">=", "+", "-", "*", "/", "//", "%"}:
            return (token.value, self._binary_precedence(token.value))
        return None

    def _binary_precedence(self, op: str) -> int:
        match op:
            case "or":
                return 10
            case "and":
                return 20
            case "==" | "!=" | "<" | "<=" | ">" | ">=" | "in" | "not in":
                return 30
            case "+" | "-":
                return 40
            case "*" | "/" | "//" | "%":
                return 50
            case _:
                return 0

    def _binary_operator(self, op: str) -> int:
        match op:
            case "or":
                return ir.BinaryOperator.BINARY_OP_OR
            case "and":
                return ir.BinaryOperator.BINARY_OP_AND
            case "==":
                return ir.BinaryOperator.BINARY_OP_EQ
            case "!=":
                return ir.BinaryOperator.BINARY_OP_NE
            case "<":
                return ir.BinaryOperator.BINARY_OP_LT
            case "<=":
                return ir.BinaryOperator.BINARY_OP_LE
            case ">":
                return ir.BinaryOperator.BINARY_OP_GT
            case ">=":
                return ir.BinaryOperator.BINARY_OP_GE
            case "in":
                return ir.BinaryOperator.BINARY_OP_IN
            case "not in":
                return ir.BinaryOperator.BINARY_OP_NOT_IN
            case "+":
                return ir.BinaryOperator.BINARY_OP_ADD
            case "-":
                return ir.BinaryOperator.BINARY_OP_SUB
            case "*":
                return ir.BinaryOperator.BINARY_OP_MUL
            case "/":
                return ir.BinaryOperator.BINARY_OP_DIV
            case "//":
                return ir.BinaryOperator.BINARY_OP_FLOOR_DIV
            case "%":
                return ir.BinaryOperator.BINARY_OP_MOD
            case _:
                assert_never(op)


class IRParser:
    """Parse IR source strings into protobuf AST structures."""

    def __init__(self, indent: str = "    ") -> None:
        self._indent = indent
        self._lines: list[str] = []
        self._index = 0

    def parse_program(self, source: str) -> ir.Program:
        self._lines = source.splitlines()
        self._index = 0
        functions: list[ir.FunctionDef] = []

        while self._index < len(self._lines):
            line = self._current_line().strip()
            if not line:
                self._index += 1
                continue
            if not line.startswith("fn "):
                raise IRParseError(f"Expected function definition, found: {line}")
            functions.append(self._parse_function())

        return ir.Program(functions=functions)

    def parse_expr(self, source: str) -> ir.Expr:
        return _ExprParser(source).parse()

    def _parse_function(self) -> ir.FunctionDef:
        line = self._current_line()
        name, inputs, outputs = _parse_function_header(line)
        self._index += 1

        body = self._parse_block(self._indent_level(line) + 1)
        fn_def = ir.FunctionDef(name=name)
        fn_def.io.inputs.extend(inputs)
        fn_def.io.outputs.extend(outputs)
        fn_def.body.CopyFrom(body)
        return fn_def

    def _parse_block(self, indent_level: int) -> ir.Block:
        statements: list[ir.Statement] = []
        saw_pass = False
        while self._index < len(self._lines):
            line = self._current_line()
            if not line.strip():
                self._index += 1
                continue
            level = self._indent_level(line)
            if level < indent_level:
                break
            if level > indent_level:
                raise IRParseError("Unexpected indentation")

            content = line.strip()
            if content == "pass":
                saw_pass = True
                self._index += 1
                continue

            stmt = self._parse_statement(indent_level)
            if stmt is not None:
                statements.append(stmt)

        if not statements and saw_pass:
            return ir.Block()
        return ir.Block(statements=statements)

    def _parse_statement(self, indent_level: int) -> Optional[ir.Statement]:
        line = self._current_line().strip()
        if line.startswith("return"):
            self._index += 1
            parts = line.split(" ", 1)
            if len(parts) == 1:
                return ir.Statement(return_stmt=ir.ReturnStmt())
            expr = self.parse_expr(parts[1])
            return ir.Statement(return_stmt=ir.ReturnStmt(value=expr))

        if line == "break":
            self._index += 1
            return ir.Statement(break_stmt=ir.BreakStmt())

        if line == "continue":
            self._index += 1
            return ir.Statement(continue_stmt=ir.ContinueStmt())

        if line.startswith("for "):
            return self._parse_for_loop(indent_level)

        if line.startswith("while "):
            return self._parse_while_loop(indent_level)

        if line.startswith("if "):
            return self._parse_conditional(indent_level)

        if line.startswith("try:"):
            return self._parse_try_except(indent_level)

        if line.startswith("parallel:") and line.rstrip() == "parallel:":
            return self._parse_parallel_block(indent_level, targets=None)

        if line.startswith("spread "):
            self._index += 1
            spread = _parse_spread(line)
            return ir.Statement(spread_action=spread)

        if line.startswith("@"):
            self._index += 1
            expr = _ExprParser(line).parse()
            if not expr.HasField("action_call"):
                raise IRParseError("Expected action call statement")
            return ir.Statement(action_call=expr.action_call)

        assignment = _split_assignment(line)
        if assignment is not None:
            targets_str, rhs = assignment
            targets = _parse_targets(targets_str)
            if rhs == "parallel:":
                return self._parse_parallel_block(indent_level, targets=targets)
            self._index += 1
            expr = self.parse_expr(rhs)
            return ir.Statement(assignment=ir.Assignment(targets=targets, value=expr))

        self._index += 1
        expr = self.parse_expr(line)
        return ir.Statement(expr_stmt=ir.ExprStmt(expr=expr))

    def _parse_for_loop(self, indent_level: int) -> ir.Statement:
        line = self._current_line().strip()
        match = re.match(r"for\s+(.+)\s+in\s+(.+):$", line)
        if not match:
            raise IRParseError("Invalid for-loop header")
        vars_str = match.group(1)
        iterable_str = match.group(2)
        loop_vars = _parse_targets(vars_str)
        iterable = self.parse_expr(iterable_str)
        self._index += 1
        block = self._parse_block(indent_level + 1)
        loop = ir.ForLoop(loop_vars=loop_vars, iterable=iterable, block_body=block)
        return ir.Statement(for_loop=loop)

    def _parse_while_loop(self, indent_level: int) -> ir.Statement:
        line = self._current_line().strip()
        match = re.match(r"while\s+(.+):$", line)
        if not match:
            raise IRParseError("Invalid while-loop header")
        condition = self.parse_expr(match.group(1))
        self._index += 1
        block = self._parse_block(indent_level + 1)
        loop = ir.WhileLoop(condition=condition, block_body=block)
        return ir.Statement(while_loop=loop)

    def _parse_conditional(self, indent_level: int) -> ir.Statement:
        line = self._current_line().strip()
        match = re.match(r"if\s+(.+):$", line)
        if not match:
            raise IRParseError("Invalid if header")
        if_cond = self.parse_expr(match.group(1))
        self._index += 1
        if_block = self._parse_block(indent_level + 1)
        conditional = ir.Conditional(
            if_branch=ir.IfBranch(condition=if_cond, block_body=if_block)
        )

        while self._index < len(self._lines):
            line = self._current_line()
            if not line.strip():
                self._index += 1
                continue
            level = self._indent_level(line)
            if level != indent_level:
                break
            stripped = line.strip()
            if stripped.startswith("elif "):
                match = re.match(r"elif\s+(.+):$", stripped)
                if not match:
                    raise IRParseError("Invalid elif header")
                cond = self.parse_expr(match.group(1))
                self._index += 1
                block = self._parse_block(indent_level + 1)
                conditional.elif_branches.append(
                    ir.ElifBranch(condition=cond, block_body=block)
                )
                continue
            if stripped == "else:":
                self._index += 1
                block = self._parse_block(indent_level + 1)
                conditional.else_branch.CopyFrom(ir.ElseBranch(block_body=block))
                break
            break

        return ir.Statement(conditional=conditional)

    def _parse_try_except(self, indent_level: int) -> ir.Statement:
        self._index += 1
        try_block = self._parse_block(indent_level + 1)
        handlers: list[ir.ExceptHandler] = []

        while self._index < len(self._lines):
            line = self._current_line()
            if not line.strip():
                self._index += 1
                continue
            level = self._indent_level(line)
            if level != indent_level:
                break
            stripped = line.strip()
            if not stripped.startswith("except"):
                break
            header = stripped[len("except") :].strip()
            if not header.endswith(":"):
                raise IRParseError("Invalid except header")
            header = header[:-1].strip()
            exc_types: list[str] = []
            exc_var = ""
            if header:
                if " as " in header:
                    parts = header.split(" as ", 1)
                    header = parts[0].strip()
                    exc_var = parts[1].strip()
                if header:
                    exc_types = [part.strip() for part in header.split(",") if part.strip()]
            self._index += 1
            block = self._parse_block(indent_level + 1)
            handler = ir.ExceptHandler(exception_types=exc_types, block_body=block)
            if exc_var:
                handler.exception_var = exc_var
            handlers.append(handler)

        if not handlers:
            raise IRParseError("try block missing except handlers")

        return ir.Statement(try_except=ir.TryExcept(try_block=try_block, handlers=handlers))

    def _parse_parallel_block(
        self, indent_level: int, targets: Optional[list[str]]
    ) -> ir.Statement:
        self._index += 1
        calls: list[ir.Call] = []
        while self._index < len(self._lines):
            line = self._current_line()
            if not line.strip():
                self._index += 1
                continue
            level = self._indent_level(line)
            if level < indent_level + 1:
                break
            if level > indent_level + 1:
                raise IRParseError("Unexpected indentation in parallel block")
            content = line.strip()
            if content == "pass":
                self._index += 1
                continue
            expr = _ExprParser(content).parse()
            if expr.HasField("action_call"):
                calls.append(ir.Call(action=expr.action_call))
            elif expr.HasField("function_call"):
                calls.append(ir.Call(function=expr.function_call))
            else:
                raise IRParseError("Parallel block expects action or function calls")
            self._index += 1

        if targets is None:
            return ir.Statement(parallel_block=ir.ParallelBlock(calls=calls))
        expr = ir.Expr(parallel_expr=ir.ParallelExpr(calls=calls))
        return ir.Statement(assignment=ir.Assignment(targets=targets, value=expr))

    def _current_line(self) -> str:
        return self._lines[self._index]

    def _indent_level(self, line: str) -> int:
        count = 0
        for ch in line:
            if ch == " ":
                count += 1
            elif ch == "\t":
                raise IRParseError("Tabs are not supported in indentation")
            else:
                break
        if count % len(self._indent) != 0:
            raise IRParseError("Indentation is not aligned to the indent width")
        return count // len(self._indent)


def _parse_function_header(line: str) -> tuple[str, list[str], list[str]]:
    pattern = r"^fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*input:\s*\[(.*?)\]\s*,\s*output:\s*\[(.*?)\]\s*\)\s*:\s*$"
    match = re.match(pattern, line.strip())
    if not match:
        raise IRParseError(f"Invalid function header: {line}")
    name = match.group(1)
    inputs = _parse_targets(match.group(2))
    outputs = _parse_targets(match.group(3))
    return name, inputs, outputs


def _parse_targets(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped or stripped == "_":
        return []
    targets = [part.strip() for part in stripped.split(",") if part.strip()]
    return targets


def _split_assignment(line: str) -> Optional[tuple[str, str]]:
    depth = 0
    in_string = False
    escape = False
    quote = ""
    for idx, ch in enumerate(line):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == quote:
                in_string = False
            continue
        if ch in {"'", '"'}:
            in_string = True
            quote = ch
            continue
        if ch in "([{":
            depth += 1
            continue
        if ch in ")]}":
            depth -= 1
            continue
        if ch == "=" and depth == 0:
            prev = line[idx - 1] if idx > 0 else ""
            nxt = line[idx + 1] if idx + 1 < len(line) else ""
            if prev in {"=", "<", ">", "!"} or nxt == "=":
                continue
            return line[:idx].strip(), line[idx + 1 :].strip()
    return None


def _parse_spread(line: str) -> ir.SpreadAction:
    if not line.startswith("spread "):
        raise IRParseError("Spread statement must start with 'spread'")
    remainder = line[len("spread ") :]
    colon_index = _find_top_level_char(remainder, ":")
    if colon_index is None:
        raise IRParseError("Spread missing loop variable separator ':'")
    collection_str = remainder[:colon_index].strip()
    tail = remainder[colon_index + 1 :].strip()
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*->\s*(.+)$", tail)
    if not match:
        raise IRParseError("Spread missing '->' action")
    loop_var = match.group(1)
    action_str = match.group(2).strip()
    collection = _ExprParser(collection_str).parse()
    action_expr = _ExprParser(action_str).parse()
    if not action_expr.HasField("action_call"):
        raise IRParseError("Spread action must target an action call")
    return ir.SpreadAction(collection=collection, loop_var=loop_var, action=action_expr.action_call)


def _find_top_level_char(text: str, ch: str) -> Optional[int]:
    depth = 0
    in_string = False
    escape = False
    quote = ""
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == quote:
                in_string = False
            continue
        if char in {"'", '"'}:
            in_string = True
            quote = char
            continue
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth -= 1
            continue
        if char == ch and depth == 0:
            return idx
    return None


def _decode_string(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise IRParseError(f"Invalid string literal: {value}") from exc
    if not isinstance(parsed, str):
        raise IRParseError(f"Expected string literal, got {type(parsed).__name__}")
    return parsed


def parse_program(source: str) -> ir.Program:
    """Convenience wrapper to parse a program."""
    return IRParser().parse_program(source)
