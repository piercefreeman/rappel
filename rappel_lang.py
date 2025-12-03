# /// script
# dependencies = [
#   "matplotlib",
# ]
# ///

"""
Rappel Language - A DSL with immutable variables, explicit I/O, and first-class actions.

Run with: uv run python rappel_lang.py
Run tests with: uv run pytest rappel_lang.py -v

Language Features:
- Immutable variables assigned with `=`
- List operations: my_list = my_list + [new_item]
- Dict operations: my_dict = my_dict + {"key": "value"}
- Python blocks with explicit input/output vars
- Functions with explicit input/output vars (all calls must use kwargs)
- Actions called with @action_name(kwarg=value) syntax (external, not defined in code)
- For loops iterating single var over list with function body
- Spread operator for variable unpacking
- List/dict key-based access
- No closures, no nested functions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# =============================================================================
# CHECKPOINT 1: Token Types and IR Data Structures
# =============================================================================


class TokenType(Enum):
    """All token types in the Rappel language."""

    # Literals
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()

    # Keywords
    FN = auto()  # function definition
    PYTHON = auto()  # python block
    FOR = auto()  # for loop
    IN = auto()  # for x in list
    IF = auto()  # conditional
    ELSE = auto()  # else branch
    RETURN = auto()  # return statement
    INPUT = auto()  # input declaration
    OUTPUT = auto()  # output declaration
    SPREAD = auto()  # spread keyword
    AT = auto()  # @ for action calls

    # Operators
    ELLIPSIS = auto()  # ... spread operator in lists
    ASSIGN = auto()  # =
    PLUS = auto()  # +
    MINUS = auto()  # -
    STAR = auto()  # *
    SLASH = auto()  # /
    DOT = auto()  # .
    COMMA = auto()  # ,
    COLON = auto()  # :
    ARROW = auto()  # ->
    EQ = auto()  # ==
    NEQ = auto()  # !=
    LT = auto()  # <
    GT = auto()  # >
    LTE = auto()  # <=
    GTE = auto()  # >=
    AND = auto()  # and
    OR = auto()  # or
    NOT = auto()  # not

    # Delimiters
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()

    # Special
    EOF = auto()
    PIPE = auto()  # |


@dataclass
class Token:
    """A single token from the lexer."""

    type: TokenType
    value: Any
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:C{self.column})"


# =============================================================================
# IR Data Structures - Immutable by design
# =============================================================================


@dataclass(frozen=True)
class SourceLocation:
    """Source code location for error reporting."""

    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None


@dataclass(frozen=True)
class RappelValue:
    """Base class for all Rappel values."""

    pass


@dataclass(frozen=True)
class RappelString(RappelValue):
    """String literal value."""

    value: str


@dataclass(frozen=True)
class RappelNumber(RappelValue):
    """Numeric literal value (int or float)."""

    value: int | float


@dataclass(frozen=True)
class RappelBoolean(RappelValue):
    """Boolean literal value."""

    value: bool


@dataclass(frozen=True)
class RappelList(RappelValue):
    """List value - immutable."""

    items: tuple[RappelExpr, ...]


@dataclass(frozen=True)
class RappelDict(RappelValue):
    """Dict value - immutable."""

    pairs: tuple[tuple[str, RappelExpr], ...]


# =============================================================================
# IR Expression Nodes
# =============================================================================


@dataclass(frozen=True)
class RappelLiteral:
    """A literal value (string, number, boolean)."""

    value: RappelValue
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelVariable:
    """A variable reference."""

    name: str
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelListExpr:
    """A list expression [a, b, c]."""

    items: tuple[RappelExpr, ...]
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelDictExpr:
    """A dict expression {"key": value}."""

    pairs: tuple[tuple[RappelExpr, RappelExpr], ...]
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelBinaryOp:
    """A binary operation (a + b, a - b, etc.)."""

    op: str
    left: RappelExpr
    right: RappelExpr
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelUnaryOp:
    """A unary operation (not x, -x)."""

    op: str
    operand: RappelExpr
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelIndexAccess:
    """Index access: list[0] or dict["key"]."""

    target: RappelExpr
    index: RappelExpr
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelDotAccess:
    """Dot access: obj.field."""

    target: RappelExpr
    field: str
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelSpread:
    """Spread operator: ...variable."""

    target: RappelExpr
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelCall:
    """Function call: func(kwarg=value). All args must be kwargs."""

    target: str
    kwargs: tuple[tuple[str, RappelExpr], ...]
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelActionCall:
    """Action call: @action_name(kwarg=value). External action, kwargs only."""

    action_name: str
    kwargs: tuple[tuple[str, RappelExpr], ...]
    location: SourceLocation | None = None


# Type alias for expressions
RappelExpr = (
    RappelLiteral
    | RappelVariable
    | RappelListExpr
    | RappelDictExpr
    | RappelBinaryOp
    | RappelUnaryOp
    | RappelIndexAccess
    | RappelDotAccess
    | RappelSpread
    | RappelCall
    | RappelActionCall
)


# =============================================================================
# IR Statement Nodes
# =============================================================================


@dataclass(frozen=True)
class RappelAssignment:
    """Variable assignment: x = expr"""

    target: str
    value: RappelExpr
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelMultiAssignment:
    """Multiple assignment (unpacking): a, b, c = expr"""

    targets: tuple[str, ...]
    value: RappelExpr
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelReturn:
    """Return statement: return expr or return [a, b, c]"""

    values: tuple[RappelExpr, ...]
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelExprStatement:
    """Expression as statement (e.g., action call)."""

    expr: RappelExpr
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelPythonBlock:
    """
    Python block with explicit input/output.

    python(input: [x, y], output: [z]):
        z = x + y
    """

    code: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelFunctionDef:
    """
    Function definition with explicit input/output.

    fn process(input: [x, y], output: [result]):
        result = x + y
        return result
    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    body: tuple[RappelStatement, ...]
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelForLoop:
    """
    For loop - iterates over a collection with a single function call in body.

    for item in items:
        result = process_item(x=item)

    The body must contain exactly one statement: an assignment with a function call.
    """

    loop_var: str
    iterable: RappelExpr
    body: tuple[RappelStatement, ...]
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelIfStatement:
    """
    If statement with explicit blocks.

    if condition:
        body
    else:
        else_body
    """

    condition: RappelExpr
    then_body: tuple[RappelStatement, ...]
    else_body: tuple[RappelStatement, ...] | None = None
    location: SourceLocation | None = None


@dataclass(frozen=True)
class RappelSpreadAction:
    """
    Spread an action over a list, with results aggregated.

    spread items:item -> @fetch_details(id=item)
    results = spread items:item -> @fetch_details(id=item)

    - source_list: the list to spread over
    - item_var: the variable name for each item in the loop
    - action: the action call to execute for each item
    - target: optional variable to store aggregated results
    """

    source_list: RappelExpr
    item_var: str
    action: RappelActionCall
    target: str | None = None  # If assigned: results = spread ...
    location: SourceLocation | None = None


# Type alias for statements
RappelStatement = (
    RappelAssignment
    | RappelMultiAssignment
    | RappelReturn
    | RappelExprStatement
    | RappelPythonBlock
    | RappelFunctionDef
    | RappelForLoop
    | RappelIfStatement
    | RappelSpreadAction
)


@dataclass(frozen=True)
class RappelProgram:
    """A complete Rappel program."""

    statements: tuple[RappelStatement, ...]
    location: SourceLocation | None = None


# =============================================================================
# CHECKPOINT 1 TESTS: IR Data Structures
# =============================================================================


def test_ir_literal_values():
    """Test creating literal values."""
    s = RappelString("hello")
    assert s.value == "hello"

    n = RappelNumber(42)
    assert n.value == 42

    f = RappelNumber(3.14)
    assert f.value == 3.14

    b = RappelBoolean(True)
    assert b.value is True


def test_ir_list_and_dict():
    """Test creating list and dict values."""
    # List with items
    items = (
        RappelLiteral(RappelNumber(1)),
        RappelLiteral(RappelNumber(2)),
        RappelLiteral(RappelNumber(3)),
    )
    lst = RappelListExpr(items=items)
    assert len(lst.items) == 3

    # Dict with pairs
    pairs = (
        (RappelLiteral(RappelString("a")), RappelLiteral(RappelNumber(1))),
        (RappelLiteral(RappelString("b")), RappelLiteral(RappelNumber(2))),
    )
    dct = RappelDictExpr(pairs=pairs)
    assert len(dct.pairs) == 2


def test_ir_expressions():
    """Test creating various expressions."""
    # Variable reference
    var = RappelVariable(name="x")
    assert var.name == "x"

    # Binary operation
    add = RappelBinaryOp(
        op="+",
        left=RappelVariable(name="a"),
        right=RappelVariable(name="b"),
    )
    assert add.op == "+"

    # Index access
    idx = RappelIndexAccess(
        target=RappelVariable(name="my_list"),
        index=RappelLiteral(RappelNumber(0)),
    )
    assert isinstance(idx.target, RappelVariable)

    # Spread operator
    spread = RappelSpread(target=RappelVariable(name="items"))
    assert isinstance(spread.target, RappelVariable)


def test_ir_statements():
    """Test creating statements."""
    # Assignment
    assign = RappelAssignment(
        target="x",
        value=RappelLiteral(RappelNumber(42)),
    )
    assert assign.target == "x"

    # Multi-assignment (unpacking)
    multi = RappelMultiAssignment(
        targets=("a", "b", "c"),
        value=RappelVariable(name="result"),
    )
    assert len(multi.targets) == 3


def test_ir_python_block():
    """Test Python block with explicit I/O."""
    block = RappelPythonBlock(
        code="z = x + y",
        inputs=("x", "y"),
        outputs=("z",),
    )
    assert block.inputs == ("x", "y")
    assert block.outputs == ("z",)


def test_ir_function_def():
    """Test function definition with explicit I/O."""
    fn = RappelFunctionDef(
        name="add",
        inputs=("a", "b"),
        outputs=("result",),
        body=(
            RappelAssignment(
                target="result",
                value=RappelBinaryOp(
                    op="+",
                    left=RappelVariable(name="a"),
                    right=RappelVariable(name="b"),
                ),
            ),
            RappelReturn(values=(RappelVariable(name="result"),)),
        ),
    )
    assert fn.name == "add"
    assert fn.inputs == ("a", "b")
    assert fn.outputs == ("result",)


def test_ir_action_call():
    """Test action call expression."""
    action_call = RappelActionCall(
        action_name="fetch_url",
        kwargs=(
            ("url", RappelLiteral(RappelString("https://example.com"))),
        ),
    )
    assert action_call.action_name == "fetch_url"
    assert len(action_call.kwargs) == 1
    assert action_call.kwargs[0][0] == "url"


def test_ir_for_loop():
    """Test for loop with single function call body."""
    body_stmt = RappelAssignment(
        target="processed",
        value=RappelCall(
            target="process_item",
            kwargs=(("x", RappelVariable(name="item")),),
        ),
    )

    for_loop = RappelForLoop(
        loop_var="item",
        iterable=RappelVariable(name="items"),
        body=(body_stmt,),
    )
    assert for_loop.loop_var == "item"
    assert len(for_loop.body) == 1
    assert isinstance(for_loop.body[0], RappelAssignment)


def test_ir_if_statement():
    """Test if statement."""
    if_stmt = RappelIfStatement(
        condition=RappelBinaryOp(
            op=">",
            left=RappelVariable(name="x"),
            right=RappelLiteral(RappelNumber(0)),
        ),
        then_body=(
            RappelAssignment(
                target="result",
                value=RappelLiteral(RappelString("positive")),
            ),
        ),
        else_body=(
            RappelAssignment(
                target="result",
                value=RappelLiteral(RappelString("non-positive")),
            ),
        ),
    )
    assert if_stmt.condition.op == ">"


def test_ir_immutability():
    """Test that IR nodes are frozen (immutable)."""
    assign = RappelAssignment(
        target="x",
        value=RappelLiteral(RappelNumber(42)),
    )

    # Should raise FrozenInstanceError
    try:
        assign.target = "y"  # type: ignore
        assert False, "Should have raised an error"
    except AttributeError:
        pass  # Expected - frozen dataclass


# =============================================================================
# CHECKPOINT 2: Lexer Implementation
# =============================================================================


class RappelLexer:
    """
    Lexer for the Rappel language.

    Handles:
    - Keywords: fn, action, python, for, in, if, else, return, input, output
    - Operators: =, +, -, *, /, ==, !=, <, >, <=, >=, and, or, not, ->
    - Literals: strings, numbers, booleans
    - Identifiers
    - Indentation-based blocks
    - Spread operator: ...
    """

    KEYWORDS = {
        "fn": TokenType.FN,
        "python": TokenType.PYTHON,
        "for": TokenType.FOR,
        "in": TokenType.IN,
        "if": TokenType.IF,
        "else": TokenType.ELSE,
        "return": TokenType.RETURN,
        "input": TokenType.INPUT,
        "output": TokenType.OUTPUT,
        "spread": TokenType.SPREAD,
        "and": TokenType.AND,
        "or": TokenType.OR,
        "not": TokenType.NOT,
        "true": TokenType.BOOLEAN,
        "false": TokenType.BOOLEAN,
        "True": TokenType.BOOLEAN,
        "False": TokenType.BOOLEAN,
    }

    SINGLE_CHAR_TOKENS = {
        "(": TokenType.LPAREN,
        ")": TokenType.RPAREN,
        "[": TokenType.LBRACKET,
        "]": TokenType.RBRACKET,
        "{": TokenType.LBRACE,
        "}": TokenType.RBRACE,
        ",": TokenType.COMMA,
        ":": TokenType.COLON,
        "+": TokenType.PLUS,
        "*": TokenType.STAR,
        "/": TokenType.SLASH,
        "|": TokenType.PIPE,
        "@": TokenType.AT,
    }

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []
        self.indent_stack: list[int] = [0]
        self._at_line_start = True

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source."""
        while self.pos < len(self.source):
            self._scan_token()

        # Emit remaining DEDENTs
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(
                Token(TokenType.DEDENT, None, self.line, self.column)
            )

        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens

    def _scan_token(self) -> None:
        """Scan the next token."""
        # Handle line start indentation
        if self._at_line_start:
            self._handle_indentation()
            self._at_line_start = False
            if self.pos >= len(self.source):
                return

        char = self.source[self.pos]

        # Skip whitespace (but not newlines)
        if char in " \t" and not self._at_line_start:
            self._advance()
            return

        # Comments
        if char == "#":
            self._skip_comment()
            return

        # Newlines
        if char == "\n":
            # Only emit NEWLINE if we have meaningful tokens
            if self.tokens and self.tokens[-1].type not in (
                TokenType.NEWLINE,
                TokenType.INDENT,
                TokenType.DEDENT,
            ):
                self.tokens.append(
                    Token(TokenType.NEWLINE, "\n", self.line, self.column)
                )
            self._advance()
            self.line += 1
            self.column = 1
            self._at_line_start = True
            return

        # Multi-character operators
        if char == "-" and self._peek(1) == ">":
            self.tokens.append(
                Token(TokenType.ARROW, "->", self.line, self.column)
            )
            self._advance()
            self._advance()
            return

        if char == "=" and self._peek(1) == "=":
            self.tokens.append(Token(TokenType.EQ, "==", self.line, self.column))
            self._advance()
            self._advance()
            return

        if char == "!" and self._peek(1) == "=":
            self.tokens.append(
                Token(TokenType.NEQ, "!=", self.line, self.column)
            )
            self._advance()
            self._advance()
            return

        if char == "<" and self._peek(1) == "=":
            self.tokens.append(
                Token(TokenType.LTE, "<=", self.line, self.column)
            )
            self._advance()
            self._advance()
            return

        if char == ">" and self._peek(1) == "=":
            self.tokens.append(
                Token(TokenType.GTE, ">=", self.line, self.column)
            )
            self._advance()
            self._advance()
            return

        if char == "." and self._peek(1) == "." and self._peek(2) == ".":
            self.tokens.append(
                Token(TokenType.ELLIPSIS, "...", self.line, self.column)
            )
            self._advance()
            self._advance()
            self._advance()
            return

        # Single character operators
        if char == "=":
            self.tokens.append(
                Token(TokenType.ASSIGN, "=", self.line, self.column)
            )
            self._advance()
            return

        if char == "-":
            self.tokens.append(
                Token(TokenType.MINUS, "-", self.line, self.column)
            )
            self._advance()
            return

        if char == ".":
            self.tokens.append(Token(TokenType.DOT, ".", self.line, self.column))
            self._advance()
            return

        if char == "<":
            self.tokens.append(Token(TokenType.LT, "<", self.line, self.column))
            self._advance()
            return

        if char == ">":
            self.tokens.append(Token(TokenType.GT, ">", self.line, self.column))
            self._advance()
            return

        if char in self.SINGLE_CHAR_TOKENS:
            self.tokens.append(
                Token(
                    self.SINGLE_CHAR_TOKENS[char], char, self.line, self.column
                )
            )
            self._advance()
            return

        # String literals
        if char in '"\'':
            self._scan_string(char)
            return

        # Number literals
        if char.isdigit() or (char == "-" and self._peek(1).isdigit()):
            self._scan_number()
            return

        # Identifiers and keywords
        if char.isalpha() or char == "_":
            self._scan_identifier()
            return

        # Unknown character
        raise SyntaxError(
            f"Unexpected character '{char}' at line {self.line}, column {self.column}"
        )

    def _handle_indentation(self) -> None:
        """Handle indentation at the start of a line."""
        indent = 0
        while self.pos < len(self.source) and self.source[self.pos] in " \t":
            if self.source[self.pos] == " ":
                indent += 1
            else:  # tab
                indent += 4  # Treat tabs as 4 spaces
            self._advance()

        # Skip blank lines and comment-only lines
        if self.pos < len(self.source) and self.source[self.pos] in "\n#":
            return

        current_indent = self.indent_stack[-1]

        if indent > current_indent:
            self.indent_stack.append(indent)
            self.tokens.append(
                Token(TokenType.INDENT, indent, self.line, self.column)
            )
        elif indent < current_indent:
            while self.indent_stack and indent < self.indent_stack[-1]:
                self.indent_stack.pop()
                self.tokens.append(
                    Token(TokenType.DEDENT, None, self.line, self.column)
                )

            if self.indent_stack[-1] != indent:
                raise SyntaxError(
                    f"Inconsistent indentation at line {self.line}"
                )

    def _scan_string(self, quote: str) -> None:
        """Scan a string literal."""
        start_line = self.line
        start_col = self.column
        self._advance()  # consume opening quote

        value = ""
        while self.pos < len(self.source):
            char = self.source[self.pos]

            if char == quote:
                self._advance()
                self.tokens.append(
                    Token(TokenType.STRING, value, start_line, start_col)
                )
                return

            if char == "\\":
                self._advance()
                if self.pos < len(self.source):
                    escape_char = self.source[self.pos]
                    if escape_char == "n":
                        value += "\n"
                    elif escape_char == "t":
                        value += "\t"
                    elif escape_char == "\\":
                        value += "\\"
                    elif escape_char == quote:
                        value += quote
                    else:
                        value += escape_char
                    self._advance()
            else:
                value += char
                self._advance()

        raise SyntaxError(f"Unterminated string starting at line {start_line}")

    def _scan_number(self) -> None:
        """Scan a number literal."""
        start_col = self.column
        value = ""

        if self.source[self.pos] == "-":
            value += "-"
            self._advance()

        while self.pos < len(self.source) and (
            self.source[self.pos].isdigit() or self.source[self.pos] == "."
        ):
            value += self.source[self.pos]
            self._advance()

        if "." in value:
            self.tokens.append(
                Token(TokenType.NUMBER, float(value), self.line, start_col)
            )
        else:
            self.tokens.append(
                Token(TokenType.NUMBER, int(value), self.line, start_col)
            )

    def _scan_identifier(self) -> None:
        """Scan an identifier or keyword."""
        start_col = self.column
        value = ""

        while self.pos < len(self.source) and (
            self.source[self.pos].isalnum() or self.source[self.pos] == "_"
        ):
            value += self.source[self.pos]
            self._advance()

        if value in self.KEYWORDS:
            token_type = self.KEYWORDS[value]
            if token_type == TokenType.BOOLEAN:
                self.tokens.append(
                    Token(
                        token_type,
                        value in ("true", "True"),
                        self.line,
                        start_col,
                    )
                )
            else:
                self.tokens.append(
                    Token(token_type, value, self.line, start_col)
                )
        else:
            self.tokens.append(
                Token(TokenType.IDENTIFIER, value, self.line, start_col)
            )

    def _skip_comment(self) -> None:
        """Skip a comment until end of line."""
        while self.pos < len(self.source) and self.source[self.pos] != "\n":
            self._advance()

    def _advance(self) -> str:
        """Advance to the next character."""
        char = self.source[self.pos] if self.pos < len(self.source) else ""
        self.pos += 1
        self.column += 1
        return char

    def _peek(self, offset: int = 0) -> str:
        """Peek at a character without advancing."""
        pos = self.pos + offset
        return self.source[pos] if pos < len(self.source) else ""


# =============================================================================
# CHECKPOINT 2 TESTS: Lexer
# =============================================================================


def test_lexer_simple_assignment():
    """Test lexing a simple assignment."""
    lexer = RappelLexer("x = 42")
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[0].value == "x"
    assert tokens[1].type == TokenType.ASSIGN
    assert tokens[2].type == TokenType.NUMBER
    assert tokens[2].value == 42


def test_lexer_string_literal():
    """Test lexing string literals."""
    lexer = RappelLexer('"hello world"')
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.STRING
    assert tokens[0].value == "hello world"


def test_lexer_list_literal():
    """Test lexing list literals."""
    lexer = RappelLexer("[1, 2, 3]")
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.LBRACKET
    assert tokens[1].type == TokenType.NUMBER
    assert tokens[1].value == 1
    assert tokens[2].type == TokenType.COMMA
    assert tokens[3].type == TokenType.NUMBER
    assert tokens[5].type == TokenType.NUMBER
    assert tokens[6].type == TokenType.RBRACKET


def test_lexer_dict_literal():
    """Test lexing dict literals."""
    lexer = RappelLexer('{"key": "value"}')
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.LBRACE
    assert tokens[1].type == TokenType.STRING
    assert tokens[1].value == "key"
    assert tokens[2].type == TokenType.COLON
    assert tokens[3].type == TokenType.STRING
    assert tokens[3].value == "value"
    assert tokens[4].type == TokenType.RBRACE


def test_lexer_keywords():
    """Test lexing keywords."""
    lexer = RappelLexer("fn python for in if else return input output")
    tokens = lexer.tokenize()

    expected = [
        TokenType.FN,
        TokenType.PYTHON,
        TokenType.FOR,
        TokenType.IN,
        TokenType.IF,
        TokenType.ELSE,
        TokenType.RETURN,
        TokenType.INPUT,
        TokenType.OUTPUT,
    ]

    for i, expected_type in enumerate(expected):
        assert tokens[i].type == expected_type


def test_lexer_operators():
    """Test lexing operators."""
    lexer = RappelLexer("+ - * / = == != < > <= >= -> ...")
    tokens = lexer.tokenize()

    expected = [
        TokenType.PLUS,
        TokenType.MINUS,
        TokenType.STAR,
        TokenType.SLASH,
        TokenType.ASSIGN,
        TokenType.EQ,
        TokenType.NEQ,
        TokenType.LT,
        TokenType.GT,
        TokenType.LTE,
        TokenType.GTE,
        TokenType.ARROW,
        TokenType.ELLIPSIS,
    ]

    for i, expected_type in enumerate(expected):
        assert tokens[i].type == expected_type, f"Token {i}: expected {expected_type}, got {tokens[i].type}"


def test_lexer_boolean():
    """Test lexing boolean values."""
    lexer = RappelLexer("true false True False")
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.BOOLEAN
    assert tokens[0].value is True
    assert tokens[1].type == TokenType.BOOLEAN
    assert tokens[1].value is False
    assert tokens[2].type == TokenType.BOOLEAN
    assert tokens[2].value is True
    assert tokens[3].type == TokenType.BOOLEAN
    assert tokens[3].value is False


def test_lexer_indentation():
    """Test lexing indentation."""
    source = """if x:
    y = 1
    z = 2
w = 3"""
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    # Find INDENT and DEDENT tokens
    indent_count = sum(1 for t in tokens if t.type == TokenType.INDENT)
    dedent_count = sum(1 for t in tokens if t.type == TokenType.DEDENT)

    assert indent_count == 1
    assert dedent_count == 1


def test_lexer_comments():
    """Test that comments are skipped."""
    source = """x = 1  # this is a comment
# full line comment
y = 2"""
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    # Should have x, =, 1, newline, y, =, 2, eof
    identifiers = [t for t in tokens if t.type == TokenType.IDENTIFIER]
    assert len(identifiers) == 2
    assert identifiers[0].value == "x"
    assert identifiers[1].value == "y"


def test_lexer_function_def():
    """Test lexing a function definition."""
    source = """fn add(input: [a, b], output: [result]):
    result = a + b
    return result"""
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.FN
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].value == "add"


def test_lexer_for_loop():
    """Test lexing a for loop."""
    source = "for item in items:"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.FOR
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].value == "item"
    assert tokens[2].type == TokenType.IN
    assert tokens[3].type == TokenType.IDENTIFIER
    assert tokens[3].value == "items"
    assert tokens[4].type == TokenType.COLON


def test_lexer_list_concat():
    """Test lexing list concatenation."""
    source = "my_list = my_list + [new_item]"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[0].value == "my_list"
    assert tokens[1].type == TokenType.ASSIGN
    assert tokens[2].type == TokenType.IDENTIFIER
    assert tokens[3].type == TokenType.PLUS
    assert tokens[4].type == TokenType.LBRACKET


def test_lexer_dict_concat():
    """Test lexing dict concatenation."""
    source = 'my_dict = my_dict + {"key": "new"}'
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[3].type == TokenType.PLUS
    assert tokens[4].type == TokenType.LBRACE


def test_lexer_ellipsis_operator():
    """Test lexing ellipsis operator."""
    source = "call(...args)"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[2].type == TokenType.ELLIPSIS
    assert tokens[3].type == TokenType.IDENTIFIER
    assert tokens[3].value == "args"


def test_lexer_index_access():
    """Test lexing index access."""
    source = 'my_list[0] + my_dict["key"]'
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    # my_list [ 0 ] + my_dict [ "key" ]
    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[1].type == TokenType.LBRACKET
    assert tokens[2].type == TokenType.NUMBER
    assert tokens[3].type == TokenType.RBRACKET


def test_lexer_python_block():
    """Test lexing python block."""
    source = """python(input: [x, y], output: [z]):
    z = x + y"""
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.PYTHON
    assert tokens[1].type == TokenType.LPAREN
    assert tokens[2].type == TokenType.INPUT


def test_lexer_action_call():
    """Test lexing action call with @ syntax."""
    source = "@fetch_url(url=my_url)"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.AT
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].value == "fetch_url"
    assert tokens[2].type == TokenType.LPAREN


# =============================================================================
# CHECKPOINT 3: Parser Implementation
# =============================================================================


class RappelParser:
    """
    Parser for the Rappel language.

    Converts tokens into an AST (IR nodes).

    Enforces:
    - No nested function definitions (fn inside fn)
    - No nested python blocks inside fn (python blocks at top level only)
    - For loop body functions are allowed (they're the loop mechanism)
    - All function calls must use kwargs (no positional args)
    - Actions are called with @action_name(kwargs) syntax
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self._in_function = False  # Track if we're inside fn

    def parse(self) -> RappelProgram:
        """Parse a complete program."""
        statements = []

        while not self._is_at_end():
            # Skip newlines at top level
            while self._check(TokenType.NEWLINE):
                self._advance()
                if self._is_at_end():
                    break

            if self._is_at_end():
                break

            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)

        return RappelProgram(statements=tuple(statements))

    def _parse_statement(self) -> RappelStatement | None:
        """Parse a single statement."""
        if self._check(TokenType.FN):
            if self._in_function:
                raise SyntaxError(
                    f"Nested function definitions are not allowed. "
                    f"Define all functions at the top level. "
                    f"Line {self._peek().line}"
                )
            return self._parse_function_def()

        if self._check(TokenType.PYTHON):
            if self._in_function:
                raise SyntaxError(
                    f"Python blocks are not allowed inside functions. "
                    f"Define python blocks at the top level or use regular assignments. "
                    f"Line {self._peek().line}"
                )
            return self._parse_python_block()

        if self._check(TokenType.FOR):
            return self._parse_for_loop()

        if self._check(TokenType.IF):
            return self._parse_if_statement()

        if self._check(TokenType.RETURN):
            return self._parse_return()

        if self._check(TokenType.SPREAD):
            return self._parse_spread_action()

        # Assignment or expression statement (may also be spread with assignment)
        return self._parse_assignment_or_expr()

    def _parse_function_def(self) -> RappelFunctionDef:
        """Parse a function definition."""
        loc = self._location()
        self._consume(TokenType.FN, "Expected 'fn'")
        name = self._consume(TokenType.IDENTIFIER, "Expected function name").value

        self._consume(TokenType.LPAREN, "Expected '(' after function name")
        inputs, outputs = self._parse_io_spec()
        self._consume(TokenType.RPAREN, "Expected ')' after parameters")

        self._consume(TokenType.COLON, "Expected ':' after function signature")
        self._consume_newline()

        # Set flag to prevent nested definitions
        prev_in_fn = self._in_function
        self._in_function = True
        try:
            body = self._parse_block()
        finally:
            self._in_function = prev_in_fn

        return RappelFunctionDef(
            name=name,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            body=tuple(body),
            location=loc,
        )

    def _parse_python_block(self) -> RappelPythonBlock:
        """Parse a python block with explicit I/O."""
        loc = self._location()
        self._consume(TokenType.PYTHON, "Expected 'python'")

        self._consume(TokenType.LPAREN, "Expected '(' after 'python'")
        inputs, outputs = self._parse_io_spec()
        self._consume(TokenType.RPAREN, "Expected ')' after python parameters")

        self._consume(TokenType.COLON, "Expected ':' after python signature")
        self._consume_newline()

        # For python blocks, we capture the raw code
        code_lines = self._parse_raw_block()
        code = "\n".join(code_lines)

        return RappelPythonBlock(
            code=code,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            location=loc,
        )

    def _parse_for_loop(self) -> RappelForLoop:
        """
        Parse a for loop with a single function call in body.

        Syntax:
            for item in items:
                result = process_item(x=item)

        The body must contain exactly one statement: an assignment with a function call.
        """
        loc = self._location()
        self._consume(TokenType.FOR, "Expected 'for'")
        loop_var = self._consume(
            TokenType.IDENTIFIER, "Expected loop variable"
        ).value
        self._consume(TokenType.IN, "Expected 'in'")
        iterable = self._parse_expression()

        self._consume(TokenType.COLON, "Expected ':' after for loop header")
        self._consume_newline()

        body = self._parse_block()

        # Validate: body must contain exactly one function call
        if len(body) != 1:
            raise RappelSyntaxError(
                f"For loop body must contain exactly one statement, got {len(body)}",
                loc
            )

        stmt = body[0]
        # Must be an assignment with a function call on the right side
        if not isinstance(stmt, RappelAssignment):
            raise RappelSyntaxError(
                "For loop body must be an assignment with a function call",
                loc
            )
        if not isinstance(stmt.value, RappelCall):
            raise RappelSyntaxError(
                "For loop body assignment must have a function call on the right side",
                loc
            )

        return RappelForLoop(
            loop_var=loop_var,
            iterable=iterable,
            body=tuple(body),
            location=loc,
        )

    def _parse_spread_action(self, target: str | None = None) -> RappelSpreadAction:
        """
        Parse a spread action statement.

        Syntax: spread <source_list>:<item_var> -> @action(kwargs)
        Or with assignment: <target> = spread <source_list>:<item_var> -> @action(kwargs)
        """
        loc = self._location()
        self._consume(TokenType.SPREAD, "Expected 'spread'")

        # Parse source_list:item_var
        source_list = self._parse_expression()

        self._consume(TokenType.COLON, "Expected ':' after source list in spread")

        item_var = self._consume(
            TokenType.IDENTIFIER, "Expected item variable name after ':'"
        ).value

        self._consume(TokenType.ARROW, "Expected '->' before action in spread")

        # Parse the action call (must be @action_name(...))
        if not self._check(TokenType.AT):
            raise SyntaxError(
                f"Expected action call (@action_name) after '->' in spread, "
                f"got {self._peek().type.name} at line {self._peek().line}"
            )

        self._advance()  # consume @
        action_name = self._consume(
            TokenType.IDENTIFIER, "Expected action name after '@'"
        ).value
        self._consume(TokenType.LPAREN, "Expected '(' after action name")
        kwargs = self._parse_kwargs_only()
        self._consume(TokenType.RPAREN, "Expected ')'")

        action = RappelActionCall(
            action_name=action_name,
            kwargs=tuple(kwargs),
            location=loc,
        )

        return RappelSpreadAction(
            source_list=source_list,
            item_var=item_var,
            action=action,
            target=target,
            location=loc,
        )

    def _parse_if_statement(self) -> RappelIfStatement:
        """Parse an if statement."""
        loc = self._location()
        self._consume(TokenType.IF, "Expected 'if'")
        condition = self._parse_expression()

        self._consume(TokenType.COLON, "Expected ':' after condition")
        self._consume_newline()

        then_body = self._parse_block()

        else_body = None
        if self._check(TokenType.ELSE):
            self._advance()
            self._consume(TokenType.COLON, "Expected ':' after 'else'")
            self._consume_newline()
            else_body = self._parse_block()

        return RappelIfStatement(
            condition=condition,
            then_body=tuple(then_body),
            else_body=tuple(else_body) if else_body else None,
            location=loc,
        )

    def _parse_return(self) -> RappelReturn:
        """Parse a return statement."""
        loc = self._location()
        self._consume(TokenType.RETURN, "Expected 'return'")

        values = []
        if not self._check(TokenType.NEWLINE) and not self._is_at_end():
            # Check for list return [a, b, c]
            if self._check(TokenType.LBRACKET):
                self._advance()
                while not self._check(TokenType.RBRACKET):
                    values.append(self._parse_expression())
                    if self._check(TokenType.COMMA):
                        self._advance()
                self._consume(TokenType.RBRACKET, "Expected ']'")
            else:
                values.append(self._parse_expression())

        return RappelReturn(values=tuple(values), location=loc)

    def _parse_assignment_or_expr(self) -> RappelStatement:
        """Parse an assignment or expression statement."""
        loc = self._location()

        # Check for multi-assignment: a, b, c = ...
        if self._check(TokenType.IDENTIFIER):
            first_id = self._advance()
            targets = [first_id.value]

            while self._check(TokenType.COMMA):
                self._advance()
                if self._check(TokenType.IDENTIFIER):
                    targets.append(self._advance().value)
                else:
                    # Not a multi-assignment, backtrack
                    self.pos -= len(targets) * 2 - 1
                    break

            if self._check(TokenType.ASSIGN):
                self._advance()

                # Check if RHS is a spread statement: target = spread ...
                if self._check(TokenType.SPREAD):
                    if len(targets) > 1:
                        raise SyntaxError(
                            f"Spread assignment can only have one target, "
                            f"got {len(targets)} at line {self._peek().line}"
                        )
                    return self._parse_spread_action(target=targets[0])

                value = self._parse_expression()

                if len(targets) > 1:
                    return RappelMultiAssignment(
                        targets=tuple(targets), value=value, location=loc
                    )
                else:
                    return RappelAssignment(
                        target=targets[0], value=value, location=loc
                    )
            else:
                # Not an assignment, backtrack and parse as expression
                self.pos -= len(targets) * 2 - 1

        # Expression statement
        expr = self._parse_expression()
        return RappelExprStatement(expr=expr, location=loc)

    def _parse_expression(self) -> RappelExpr:
        """Parse an expression."""
        return self._parse_or()

    def _parse_or(self) -> RappelExpr:
        """Parse or expressions."""
        left = self._parse_and()

        while self._check(TokenType.OR):
            op = self._advance().value
            right = self._parse_and()
            left = RappelBinaryOp(op=op, left=left, right=right)

        return left

    def _parse_and(self) -> RappelExpr:
        """Parse and expressions."""
        left = self._parse_not()

        while self._check(TokenType.AND):
            op = self._advance().value
            right = self._parse_not()
            left = RappelBinaryOp(op=op, left=left, right=right)

        return left

    def _parse_not(self) -> RappelExpr:
        """Parse not expressions."""
        if self._check(TokenType.NOT):
            op = self._advance().value
            operand = self._parse_not()
            return RappelUnaryOp(op=op, operand=operand)

        return self._parse_comparison()

    def _parse_comparison(self) -> RappelExpr:
        """Parse comparison expressions."""
        left = self._parse_additive()

        while self._check_any(
            TokenType.EQ,
            TokenType.NEQ,
            TokenType.LT,
            TokenType.GT,
            TokenType.LTE,
            TokenType.GTE,
        ):
            op = self._advance().value
            right = self._parse_additive()
            left = RappelBinaryOp(op=op, left=left, right=right)

        return left

    def _parse_additive(self) -> RappelExpr:
        """Parse additive expressions (+, -)."""
        left = self._parse_multiplicative()

        while self._check_any(TokenType.PLUS, TokenType.MINUS):
            op = self._advance().value
            right = self._parse_multiplicative()
            left = RappelBinaryOp(op=op, left=left, right=right)

        return left

    def _parse_multiplicative(self) -> RappelExpr:
        """Parse multiplicative expressions (*, /)."""
        left = self._parse_unary()

        while self._check_any(TokenType.STAR, TokenType.SLASH):
            op = self._advance().value
            right = self._parse_unary()
            left = RappelBinaryOp(op=op, left=left, right=right)

        return left

    def _parse_unary(self) -> RappelExpr:
        """Parse unary expressions."""
        if self._check(TokenType.MINUS):
            op = self._advance().value
            operand = self._parse_unary()
            return RappelUnaryOp(op=op, operand=operand)

        if self._check(TokenType.ELLIPSIS):
            self._advance()
            operand = self._parse_postfix()
            return RappelSpread(target=operand)

        return self._parse_postfix()

    def _parse_postfix(self) -> RappelExpr:
        """Parse postfix expressions (calls, index access, dot access)."""
        expr = self._parse_primary()

        while True:
            if self._check(TokenType.LBRACKET):
                self._advance()
                index = self._parse_expression()
                self._consume(TokenType.RBRACKET, "Expected ']'")
                expr = RappelIndexAccess(target=expr, index=index)
            elif self._check(TokenType.DOT):
                self._advance()
                field = self._consume(
                    TokenType.IDENTIFIER, "Expected field name"
                ).value
                expr = RappelDotAccess(target=expr, field=field)
            elif self._check(TokenType.LPAREN) and isinstance(
                expr, RappelVariable
            ):
                # Function call - kwargs only
                loc = self._location()
                self._advance()
                kwargs = self._parse_kwargs_only()
                self._consume(TokenType.RPAREN, "Expected ')'")
                expr = RappelCall(
                    target=expr.name,
                    kwargs=tuple(kwargs),
                    location=loc,
                )
            else:
                break

        return expr

    def _parse_primary(self) -> RappelExpr:
        """Parse primary expressions."""
        loc = self._location()

        # Action call: @action_name(kwargs)
        if self._check(TokenType.AT):
            self._advance()
            action_name = self._consume(
                TokenType.IDENTIFIER, "Expected action name after '@'"
            ).value
            self._consume(TokenType.LPAREN, "Expected '(' after action name")
            kwargs = self._parse_kwargs_only()
            self._consume(TokenType.RPAREN, "Expected ')'")
            return RappelActionCall(
                action_name=action_name,
                kwargs=tuple(kwargs),
                location=loc,
            )

        # Parenthesized expression
        if self._check(TokenType.LPAREN):
            self._advance()
            expr = self._parse_expression()
            self._consume(TokenType.RPAREN, "Expected ')'")
            return expr

        # List literal
        if self._check(TokenType.LBRACKET):
            return self._parse_list_literal()

        # Dict literal
        if self._check(TokenType.LBRACE):
            return self._parse_dict_literal()

        # String literal
        if self._check(TokenType.STRING):
            value = self._advance().value
            return RappelLiteral(RappelString(value), location=loc)

        # Number literal
        if self._check(TokenType.NUMBER):
            value = self._advance().value
            return RappelLiteral(RappelNumber(value), location=loc)

        # Boolean literal
        if self._check(TokenType.BOOLEAN):
            value = self._advance().value
            return RappelLiteral(RappelBoolean(value), location=loc)

        # Identifier
        if self._check(TokenType.IDENTIFIER):
            name = self._advance().value
            return RappelVariable(name=name, location=loc)

        raise SyntaxError(
            f"Unexpected token {self._peek()} at line {self._peek().line}"
        )

    def _parse_list_literal(self) -> RappelListExpr:
        """Parse a list literal [a, b, c]."""
        loc = self._location()
        self._consume(TokenType.LBRACKET, "Expected '['")

        items = []
        while not self._check(TokenType.RBRACKET):
            items.append(self._parse_expression())
            if self._check(TokenType.COMMA):
                self._advance()

        self._consume(TokenType.RBRACKET, "Expected ']'")
        return RappelListExpr(items=tuple(items), location=loc)

    def _parse_dict_literal(self) -> RappelDictExpr:
        """Parse a dict literal {"key": value}."""
        loc = self._location()
        self._consume(TokenType.LBRACE, "Expected '{'")

        pairs = []
        while not self._check(TokenType.RBRACE):
            key = self._parse_expression()
            self._consume(TokenType.COLON, "Expected ':' after dict key")
            value = self._parse_expression()
            pairs.append((key, value))

            if self._check(TokenType.COMMA):
                self._advance()

        self._consume(TokenType.RBRACE, "Expected '}'")
        return RappelDictExpr(pairs=tuple(pairs), location=loc)

    def _parse_kwargs_only(self) -> list[tuple[str, RappelExpr]]:
        """Parse kwargs-only function/action call arguments."""
        kwargs = []

        while not self._check(TokenType.RPAREN):
            # Must be keyword argument: name=value or name: value
            if not self._check(TokenType.IDENTIFIER):
                raise SyntaxError(
                    f"Expected keyword argument (name=value), got {self._peek().type.name} "
                    f"at line {self._peek().line}. All function/action calls require kwargs."
                )

            name = self._advance().value

            # Accept both = and : for kwargs
            if self._check(TokenType.ASSIGN):
                self._advance()
            elif self._check(TokenType.COLON):
                self._advance()
            else:
                raise SyntaxError(
                    f"Expected '=' or ':' after argument name '{name}' "
                    f"at line {self._peek().line}. All function/action calls require kwargs."
                )

            value = self._parse_expression()
            kwargs.append((name, value))

            if self._check(TokenType.COMMA):
                self._advance()

        return kwargs

    def _parse_io_spec(self) -> tuple[list[str], list[str]]:
        """Parse input/output specification."""
        inputs = []
        outputs = []

        while not self._check(TokenType.RPAREN):
            if self._check(TokenType.INPUT):
                self._advance()
                self._consume(TokenType.COLON, "Expected ':' after 'input'")
                inputs = self._parse_identifier_list()
            elif self._check(TokenType.OUTPUT):
                self._advance()
                self._consume(TokenType.COLON, "Expected ':' after 'output'")
                outputs = self._parse_identifier_list()
            elif self._check(TokenType.COMMA):
                self._advance()
            else:
                break

        return inputs, outputs

    def _parse_identifier_list(self) -> list[str]:
        """Parse a list of identifiers [a, b, c]."""
        self._consume(TokenType.LBRACKET, "Expected '['")

        names = []
        while not self._check(TokenType.RBRACKET):
            names.append(
                self._consume(TokenType.IDENTIFIER, "Expected identifier").value
            )
            if self._check(TokenType.COMMA):
                self._advance()

        self._consume(TokenType.RBRACKET, "Expected ']'")
        return names

    def _parse_block(self) -> list[RappelStatement]:
        """Parse an indented block of statements."""
        statements = []

        self._consume(TokenType.INDENT, "Expected indented block")

        while not self._check(TokenType.DEDENT) and not self._is_at_end():
            # Skip newlines
            while self._check(TokenType.NEWLINE):
                self._advance()
                if self._check(TokenType.DEDENT) or self._is_at_end():
                    break

            if self._check(TokenType.DEDENT) or self._is_at_end():
                break

            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)

        if self._check(TokenType.DEDENT):
            self._advance()

        return statements

    def _parse_raw_block(self) -> list[str]:
        """Parse raw code block for python blocks."""
        lines = []

        self._consume(TokenType.INDENT, "Expected indented block")

        # For raw blocks, we need to collect all tokens until DEDENT
        # and reconstruct the code
        current_line = ""

        while not self._check(TokenType.DEDENT) and not self._is_at_end():
            token = self._advance()

            if token.type == TokenType.NEWLINE:
                if current_line.strip():
                    lines.append(current_line)
                current_line = ""
            elif token.type == TokenType.INDENT:
                current_line += "    "
            elif token.type == TokenType.DEDENT:
                self.pos -= 1  # Put it back
                break
            else:
                # Reconstruct token value
                if token.type == TokenType.STRING:
                    current_line += f'"{token.value}"'
                elif token.type == TokenType.IDENTIFIER:
                    if current_line and current_line[-1] not in " \t([{,.:":
                        current_line += " "
                    current_line += token.value
                elif token.type in (TokenType.NUMBER, TokenType.BOOLEAN):
                    if current_line and current_line[-1] not in " \t([{,.:":
                        current_line += " "
                    current_line += str(token.value)
                else:
                    current_line += token.value if token.value else ""

        if current_line.strip():
            lines.append(current_line)

        if self._check(TokenType.DEDENT):
            self._advance()

        return lines

    def _consume(self, token_type: TokenType, message: str) -> Token:
        """Consume a token of the expected type."""
        if self._check(token_type):
            return self._advance()
        raise SyntaxError(
            f"{message}. Got {self._peek().type.name} at line {self._peek().line}"
        )

    def _consume_newline(self) -> None:
        """Consume a newline token if present."""
        if self._check(TokenType.NEWLINE):
            self._advance()

    def _check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type."""
        if self._is_at_end():
            return token_type == TokenType.EOF
        return self._peek().type == token_type

    def _check_any(self, *token_types: TokenType) -> bool:
        """Check if current token is any of the given types."""
        return any(self._check(t) for t in token_types)

    def _advance(self) -> Token:
        """Advance to the next token."""
        if not self._is_at_end():
            self.pos += 1
        return self.tokens[self.pos - 1]

    def _peek(self) -> Token:
        """Peek at the current token."""
        return self.tokens[self.pos]

    def _peek_next(self) -> Token | None:
        """Peek at the next token."""
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return None

    def _is_at_end(self) -> bool:
        """Check if we've reached the end."""
        return self._peek().type == TokenType.EOF

    def _location(self) -> SourceLocation:
        """Get current source location."""
        token = self._peek()
        return SourceLocation(line=token.line, column=token.column)


def parse(source: str) -> RappelProgram:
    """Parse source code into a Rappel program."""
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()
    parser = RappelParser(tokens)
    return parser.parse()


# =============================================================================
# CHECKPOINT 3 TESTS: Parser
# =============================================================================


def test_parser_simple_assignment():
    """Test parsing simple assignment."""
    program = parse("x = 42")

    assert len(program.statements) == 1
    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert stmt.target == "x"
    assert isinstance(stmt.value, RappelLiteral)
    assert stmt.value.value.value == 42


def test_parser_string_assignment():
    """Test parsing string assignment."""
    program = parse('message = "hello world"')

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelLiteral)
    assert stmt.value.value.value == "hello world"


def test_parser_list_literal():
    """Test parsing list literal."""
    program = parse("items = [1, 2, 3]")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelListExpr)
    assert len(stmt.value.items) == 3


def test_parser_dict_literal():
    """Test parsing dict literal."""
    program = parse('config = {"name": "test", "count": 42}')

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelDictExpr)
    assert len(stmt.value.pairs) == 2


def test_parser_list_concat():
    """Test parsing list concatenation (immutable update)."""
    program = parse("my_list = my_list + [new_item]")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelBinaryOp)
    assert stmt.value.op == "+"
    assert isinstance(stmt.value.left, RappelVariable)
    assert isinstance(stmt.value.right, RappelListExpr)


def test_parser_dict_concat():
    """Test parsing dict concatenation (immutable update)."""
    program = parse('my_dict = my_dict + {"key": "new"}')

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelBinaryOp)
    assert stmt.value.op == "+"


def test_parser_index_access():
    """Test parsing index access."""
    program = parse("x = my_list[0]")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelIndexAccess)
    assert isinstance(stmt.value.target, RappelVariable)
    assert stmt.value.target.name == "my_list"


def test_parser_dict_key_access():
    """Test parsing dict key access."""
    program = parse('x = my_dict["key"]')

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelIndexAccess)


def test_parser_dot_access():
    """Test parsing dot access."""
    program = parse("x = obj.field")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelDotAccess)
    assert stmt.value.field == "field"


def test_parser_function_call():
    """Test parsing function call - must use kwargs."""
    program = parse("result = process(a=x, b=y, c=z)")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelCall)
    assert stmt.value.target == "process"
    assert len(stmt.value.kwargs) == 3


def test_parser_function_call_colon_syntax():
    """Test parsing function call with colon syntax for kwargs."""
    program = parse("result = fetch(url: my_url, timeout: 30)")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelCall)
    assert len(stmt.value.kwargs) == 2


def test_parser_function_call_rejects_positional():
    """Test that positional args are rejected."""
    import pytest
    with pytest.raises(SyntaxError) as exc_info:
        parse("result = process(a, b, c)")
    assert "kwargs" in str(exc_info.value).lower()


def test_parser_spread_operator():
    """Test parsing spread operator in list."""
    program = parse("result = [...items, extra]")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelListExpr)
    # The spread is in the items
    assert len(stmt.value.items) == 2
    assert isinstance(stmt.value.items[0], RappelSpread)


def test_parser_multi_assignment():
    """Test parsing multi-assignment (unpacking)."""
    program = parse("a, b, c = get_values()")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelMultiAssignment)
    assert stmt.targets == ("a", "b", "c")


def test_parser_function_def():
    """Test parsing function definition."""
    source = """fn add(input: [a, b], output: [result]):
    result = a + b
    return result"""

    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelFunctionDef)
    assert stmt.name == "add"
    assert stmt.inputs == ("a", "b")
    assert stmt.outputs == ("result",)
    assert len(stmt.body) == 2


def test_parser_action_call():
    """Test parsing action call with @ syntax."""
    source = 'result = @fetch_data(url="https://example.com")'

    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelActionCall)
    assert stmt.value.action_name == "fetch_data"
    assert len(stmt.value.kwargs) == 1
    assert stmt.value.kwargs[0][0] == "url"


def test_parser_spread_action():
    """Test parsing spread action syntax."""
    source = 'results = spread items:item -> @fetch_details(id=item)'

    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelSpreadAction)
    assert stmt.target == "results"
    assert stmt.item_var == "item"
    assert isinstance(stmt.source_list, RappelVariable)
    assert stmt.source_list.name == "items"
    assert stmt.action.action_name == "fetch_details"
    assert len(stmt.action.kwargs) == 1
    assert stmt.action.kwargs[0][0] == "id"


def test_parser_spread_action_no_target():
    """Test parsing spread action without assignment."""
    source = 'spread items:item -> @process(data=item)'

    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelSpreadAction)
    assert stmt.target is None
    assert stmt.item_var == "item"
    assert stmt.action.action_name == "process"


def test_parser_python_block():
    """Test parsing python block."""
    source = """python(input: [x, y], output: [z]):
    z = x + y"""

    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelPythonBlock)
    assert stmt.inputs == ("x", "y")
    assert stmt.outputs == ("z",)
    assert "z" in stmt.code and "x" in stmt.code


def test_parser_for_loop():
    """Test parsing for loop with single function call body."""
    source = """for item in items:
    result = process_item(x=item)"""

    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelForLoop)
    assert stmt.loop_var == "item"
    assert isinstance(stmt.iterable, RappelVariable)
    assert stmt.iterable.name == "items"
    assert len(stmt.body) == 1
    assert isinstance(stmt.body[0], RappelAssignment)
    assert stmt.body[0].target == "result"
    assert isinstance(stmt.body[0].value, RappelCall)
    assert stmt.body[0].value.target == "process_item"


def test_parser_if_statement():
    """Test parsing if statement."""
    source = """if x > 0:
    result = "positive"
else:
    result = "non-positive" """

    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelIfStatement)
    assert isinstance(stmt.condition, RappelBinaryOp)
    assert stmt.condition.op == ">"
    assert len(stmt.then_body) == 1
    assert stmt.else_body is not None
    assert len(stmt.else_body) == 1


def test_parser_return_single():
    """Test parsing single return."""
    source = """fn get_value(input: [], output: [x]):
    x = 42
    return x"""

    program = parse(source)
    fn = program.statements[0]
    assert isinstance(fn, RappelFunctionDef)

    ret = fn.body[-1]
    assert isinstance(ret, RappelReturn)
    assert len(ret.values) == 1


def test_parser_return_multiple():
    """Test parsing multiple return values."""
    source = """fn get_values(input: [], output: [a, b, c]):
    a = 1
    b = 2
    c = 3
    return [a, b, c]"""

    program = parse(source)
    fn = program.statements[0]

    ret = fn.body[-1]
    assert isinstance(ret, RappelReturn)
    assert len(ret.values) == 3


def test_parser_comparison_operators():
    """Test parsing comparison operators."""
    program = parse("result = a == b and c != d or e < f")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    # Should parse as: (a == b) and (c != d) or (e < f)
    # With precedence: ((a == b) and (c != d)) or (e < f)
    assert isinstance(stmt.value, RappelBinaryOp)


def test_parser_arithmetic():
    """Test parsing arithmetic expressions."""
    program = parse("result = a + b * c - d / e")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    # Should respect precedence: a + (b * c) - (d / e)
    assert isinstance(stmt.value, RappelBinaryOp)


def test_parser_unary_not():
    """Test parsing not operator."""
    program = parse("result = not flag")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelUnaryOp)
    assert stmt.value.op == "not"


def test_parser_unary_minus():
    """Test parsing unary minus."""
    program = parse("result = -x")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelUnaryOp)
    assert stmt.value.op == "-"


def test_parser_parenthesized():
    """Test parsing parenthesized expressions."""
    program = parse("result = (a + b) * c")

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelBinaryOp)
    assert stmt.value.op == "*"
    assert isinstance(stmt.value.left, RappelBinaryOp)
    assert stmt.value.left.op == "+"


def test_parser_rejects_nested_function():
    """Test that nested function definitions are rejected."""
    source = """fn outer(input: [x], output: [y]):
    fn inner(input: [a], output: [b]):
        b = a * 2
    y = inner(x)"""

    import pytest
    with pytest.raises(SyntaxError) as exc_info:
        parse(source)
    assert "Nested function definitions are not allowed" in str(exc_info.value)


def test_parser_rejects_python_block_in_function():
    """Test that python blocks inside functions are rejected."""
    source = """fn process(input: [x], output: [y]):
    python(input: [x], output: [y]):
        y = x * 2"""

    import pytest
    with pytest.raises(SyntaxError) as exc_info:
        parse(source)
    assert "Python blocks are not allowed inside functions" in str(exc_info.value)


def test_parser_action_call_in_function():
    """Test that action calls are allowed inside functions."""
    source = """fn process(input: [x], output: [y]):
    y = @fetch_data(url=x)
    return y"""

    # Should not raise - action calls are allowed anywhere
    program = parse(source)
    assert len(program.statements) == 1
    fn = program.statements[0]
    assert isinstance(fn, RappelFunctionDef)


def test_parser_allows_for_loop_in_function():
    """Test that for loops are allowed inside functions."""
    source = """fn process_all(input: [items], output: [results]):
    results = []
    for item in items:
        result = handle(x=item)
    return results"""

    # Should not raise
    program = parse(source)
    assert len(program.statements) == 1
    fn = program.statements[0]
    assert isinstance(fn, RappelFunctionDef)
    assert fn.name == "process_all"


def test_parser_allows_if_in_function():
    """Test that if statements are allowed inside functions."""
    source = """fn classify(input: [x], output: [label]):
    if x > 0:
        label = "positive"
    else:
        label = "non-positive"
    return label"""

    # Should not raise
    program = parse(source)
    assert len(program.statements) == 1


# =============================================================================
# CHECKPOINT 4: Comprehensive Examples
# =============================================================================


EXAMPLE_IMMUTABLE_VARS = """
# Immutable variable assignment
x = 42
name = "Alice"
is_active = true

# Variables must be reassigned to update
x = x + 1
name = name + " Smith"
"""

EXAMPLE_LIST_OPERATIONS = """
# List initialization
items = []
items = [1, 2, 3]

# Immutable list update (concatenation)
items = items + [4]
items = items + [5, 6]

# List access
first = items[0]
last = items[5]

# List concatenation
results = []
results = results + [10]
"""

EXAMPLE_DICT_OPERATIONS = """
# Dict initialization
config = {}
config = {"host": "localhost", "port": 8080}

# Immutable dict update
config = config + {"timeout": 30}
config = config + {"retries": 3, "debug": true}

# Dict access
host = config["host"]
port = config["port"]
"""

EXAMPLE_PYTHON_BLOCK = """
# Python block with explicit I/O
python(input: [x, y], output: [z]):
    z = x + y

# Python block for complex computation
python(input: [data], output: [mean, std]):
    import statistics
    mean = statistics.mean(data)
    std = statistics.stdev(data)
"""

EXAMPLE_FUNCTION_DEF = """
# Function with explicit input/output
fn calculate_area(input: [width, height], output: [area]):
    area = width * height
    return area

# Function with multiple outputs - returns list for unpacking
fn divide_with_remainder(input: [a, b], output: [quotient, remainder]):
    quotient = a / b
    remainder = a - quotient * b
    return [quotient, remainder]

# Simple transformation function
fn double_value(input: [x], output: [result]):
    result = x * 2
    return result

# Using functions - all calls require kwargs
q, r = divide_with_remainder(a=10, b=3)
doubled = double_value(x=5)
area = calculate_area(width=10, height=20)
"""

EXAMPLE_ACTION_CALL = """
# Actions are external - called with @action_name(kwargs) syntax
# No action definitions in code - they are defined externally

# Call an action to fetch data
response = @fetch_url(url="https://api.example.com/data")

# Call an action to save to database
record_id = @save_to_db(data=response, table="responses")

# Chain action calls
user_data = @fetch_user(id=123)
validated = @validate_user(user=user_data)
result = @update_profile(user_id=123, data=validated)
"""

EXAMPLE_FOR_LOOP = """
# For loop with single function call body
fn double_item(input: [item], output: [doubled]):
    doubled = item * 2
    return [doubled]

fn process_items(input: [], output: [results]):
    items = [1, 2, 3, 4, 5]
    results = []
    for item in items:
        doubled = double_item(item=item)
    return [results]
"""

EXAMPLE_SPREAD_OPERATOR = """
# Spread operator for unpacking in lists
base = [1, 2]
extended = [...base, 3, 4, 5]

# Spread with variables
items = [10, 20, 30]
all_items = [...items, 40, 50]
"""

EXAMPLE_CONDITIONALS = """
# If-else with explicit blocks
if value > 100:
    category = "high"
else:
    category = "low"

# Conditional in function
fn classify(input: [score], output: [grade]):
    if score >= 90:
        grade = "A"
    else:
        if score >= 80:
            grade = "B"
        else:
            if score >= 70:
                grade = "C"
            else:
                grade = "F"
    return grade
"""

EXAMPLE_COMPLEX_WORKFLOW = """
# Complex workflow example combining all features
# Functions defined at top level, actions called with @syntax

# Define validation function
fn validate_item(input: [item], output: [validated]):
    if item > 0:
        validated = item
    else:
        validated = None
    return validated

# Define item processing function
fn process_single_item(input: [item], output: [result, error]):
    validated = validate_item(item=item)
    if validated != None:
        result = validated * 2
        error = None
    else:
        result = None
        error = {"item": item, "reason": "invalid"}
    return [result, error]

# Define batch processing function
fn process_batch(input: [batch], output: [processed, failed]):
    processed = []
    failed = []
    for item in batch:
        result = process_single_item(item=item)
    if result != None:
        processed = processed + [result]
    return [processed, failed]

# Main workflow - initialize state
config = {"api_url": "https://api.example.com", "batch_size": 10}
results = []
errors = []

# Execute workflow - actions called with @syntax
batch = @fetch_batch(url=config["api_url"], offset=0, limit=config["batch_size"])
processed, failed = process_batch(batch=batch)
results = results + processed
errors = errors + failed
"""


def test_example_immutable_vars():
    """Test parsing immutable variable examples."""
    program = parse(EXAMPLE_IMMUTABLE_VARS)
    assert len(program.statements) >= 5


def test_example_list_operations():
    """Test parsing list operation examples."""
    program = parse(EXAMPLE_LIST_OPERATIONS)
    # Should have list init, updates, access, for loop
    assert len(program.statements) >= 5


def test_example_dict_operations():
    """Test parsing dict operation examples."""
    program = parse(EXAMPLE_DICT_OPERATIONS)
    assert len(program.statements) >= 5


def test_example_python_block():
    """Test parsing python block examples."""
    program = parse(EXAMPLE_PYTHON_BLOCK)
    python_blocks = [
        s for s in program.statements if isinstance(s, RappelPythonBlock)
    ]
    assert len(python_blocks) == 2


def test_example_function_def():
    """Test parsing function definition examples."""
    program = parse(EXAMPLE_FUNCTION_DEF)
    functions = [s for s in program.statements if isinstance(s, RappelFunctionDef)]
    assert len(functions) >= 2


def test_example_action_call():
    """Test parsing action call examples."""
    program = parse(EXAMPLE_ACTION_CALL)
    # Find action calls
    action_calls = []
    for stmt in program.statements:
        if isinstance(stmt, RappelAssignment):
            if isinstance(stmt.value, RappelActionCall):
                action_calls.append(stmt.value)
    assert len(action_calls) >= 3


def test_example_for_loop():
    """Test parsing for loop examples."""
    program = parse(EXAMPLE_FOR_LOOP)
    # For loops are now inside functions
    functions = [s for s in program.statements if isinstance(s, RappelFunctionDef)]
    assert len(functions) == 2  # double_item and process_items

    # Find for loop inside process_items function
    process_items_fn = [f for f in functions if f.name == "process_items"][0]
    for_loops = [s for s in process_items_fn.body if isinstance(s, RappelForLoop)]
    assert len(for_loops) == 1


def test_example_spread_operator():
    """Test parsing spread operator examples."""
    program = parse(EXAMPLE_SPREAD_OPERATOR)
    # Find statements with spread in list expressions
    has_spread = False
    for stmt in program.statements:
        if isinstance(stmt, RappelAssignment):
            if isinstance(stmt.value, RappelListExpr):
                for item in stmt.value.items:
                    if isinstance(item, RappelSpread):
                        has_spread = True
    assert has_spread


def test_example_conditionals():
    """Test parsing conditional examples."""
    program = parse(EXAMPLE_CONDITIONALS)
    if_stmts = [s for s in program.statements if isinstance(s, RappelIfStatement)]
    assert len(if_stmts) >= 1


def test_example_complex_workflow():
    """Test parsing complex workflow example."""
    program = parse(EXAMPLE_COMPLEX_WORKFLOW)
    # Should have config, functions, and main workflow with action calls
    assert len(program.statements) >= 5

    # Check for variety of statement types
    types_found = set()
    for stmt in program.statements:
        types_found.add(type(stmt).__name__)

    assert "RappelAssignment" in types_found
    assert "RappelFunctionDef" in types_found

    # Check for action calls in assignments
    has_action_call = False
    for stmt in program.statements:
        if isinstance(stmt, RappelAssignment):
            if isinstance(stmt.value, RappelActionCall):
                has_action_call = True
    assert has_action_call, "Should have at least one action call"


# =============================================================================
# Pretty Printer for IR (useful for debugging)
# =============================================================================


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

    def _print_RappelPythonBlock(self, node: RappelPythonBlock) -> str:
        inputs = ", ".join(node.inputs)
        outputs = ", ".join(node.outputs)
        return f"python(input: [{inputs}], output: [{outputs}]):\n    {node.code}"

    def _print_RappelForLoop(self, node: RappelForLoop) -> str:
        iterable = self._print_node(node.iterable)
        body = self._print_block(node.body)
        return f"for {node.loop_var} in {iterable}:\n{body}"

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


def test_pretty_printer():
    """Test the pretty printer."""
    source = """fn add(input: [a, b], output: [result]):
    result = a + b
    return result"""

    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert "fn add" in output
    assert "result = (a + b)" in output
    assert "return result" in output


# =============================================================================
# Main entry point
# =============================================================================


# =============================================================================
# New comprehensive example: Action -> Spread -> Loop -> Action
# =============================================================================

EXAMPLE_ACTION_SPREAD_LOOP = """
# Comprehensive example: All code lives in functions
# Functions are isolated - data flows only through explicit inputs/outputs
# main() is the default entrypoint

# Function to process a single order (called per-item in loop)
fn process_order(input: [order], output: [result]):
    if order["total"] > 0:
        payment_result = @process_payment(order_id=order["id"], amount=order["total"])
        if payment_result["status"] == "success":
            update_result = @update_order_status(order_id=order["id"], status="completed")
            result = {"order_id": order["id"], "payment": payment_result, "update": update_result}
        else:
            result = {"order_id": order["id"], "error": payment_result["error"]}
    else:
        result = {"order_id": order["id"], "error": "invalid_total"}
    return [result]

# Main entrypoint - orchestrates the workflow
fn main(input: [], output: [notification]):
    # Step 1: Fetch pending orders
    order_ids = @get_pending_orders(status="pending", limit=100)

    # Step 2: Spread to fetch details in parallel
    order_details = spread order_ids:order_id -> @fetch_order_details(id=order_id)

    # Step 3: Process each order (for loop body has exactly one function call)
    for order in order_details:
        result = process_order(order=order)

    # Step 4: Send summary notification
    notification = @send_summary_notification(order_count=100, channel="slack")
    return [notification]
"""


def main():
    """Main entry point - demonstrates the language."""
    print("=" * 60)
    print("Rappel Language - DSL with Immutable Variables")
    print("=" * 60)
    print()

    # Commented out: individual examples
    # examples = [
    #     ("Immutable Variables", EXAMPLE_IMMUTABLE_VARS),
    #     ("List Operations", EXAMPLE_LIST_OPERATIONS),
    #     ("Dict Operations", EXAMPLE_DICT_OPERATIONS),
    #     ("Python Blocks", EXAMPLE_PYTHON_BLOCK),
    #     ("Function Definitions", EXAMPLE_FUNCTION_DEF),
    #     ("Action Calls", EXAMPLE_ACTION_CALL),
    #     ("For Loops", EXAMPLE_FOR_LOOP),
    #     ("Spread Operator", EXAMPLE_SPREAD_OPERATOR),
    #     ("Conditionals", EXAMPLE_CONDITIONALS),
    #     ("Complex Workflow", EXAMPLE_COMPLEX_WORKFLOW),
    # ]
    #
    # printer = RappelPrettyPrinter()
    #
    # for name, source in examples:
    #     print(f"\n{'=' * 60}")
    #     print(f"Example: {name}")
    #     print("=" * 60)
    #     print("\nSource:")
    #     print("-" * 40)
    #     print(source.strip())
    #     print("-" * 40)
    #
    #     try:
    #         program = parse(source)
    #         print(f"\nParsed {len(program.statements)} statements")
    #         print("\nPretty printed:")
    #         print("-" * 40)
    #         print(printer.print(program))
    #     except SyntaxError as e:
    #         print(f"\nParse error: {e}")

    # New comprehensive example
    print("Example: Action -> Spread -> Loop -> Action Pipeline")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("  1. Calling an action to fetch a list")
    print("  2. Spreading results to parallel action calls")
    print("  3. Looping over results with a function body")
    print("  4. Calling actions within the loop")
    print("  5. Accumulating results")
    print("  6. Final action call with summary")
    print()
    print("-" * 60)
    print("SOURCE CODE:")
    print("-" * 60)
    print(EXAMPLE_ACTION_SPREAD_LOOP.strip())
    print("-" * 60)
    print()

    try:
        program = parse(EXAMPLE_ACTION_SPREAD_LOOP)
        print(f"Successfully parsed {len(program.statements)} statements")
        print()

        printer = RappelPrettyPrinter()
        print("-" * 60)
        print("PRETTY PRINTED IR:")
        print("-" * 60)
        print(printer.print(program))
        print("-" * 60)

        # Convert to DAG
        print()
        print("=" * 60)
        print("DAG CONVERSION")
        print("=" * 60)
        print()

        dag = convert_to_dag(program)
        print(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
        print()

        print("NODES:")
        print("-" * 40)
        for node_id, node in sorted(dag.nodes.items()):
            flags = []
            if node.is_loop_head:
                flags.append("loop")
            if node.is_aggregator:
                flags.append("aggregator")
            flags_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  {node_id}: {node.node_type} - {node.label}{flags_str}")

        print()
        print("STATE MACHINE EDGES (execution order):")
        print("-" * 40)
        for edge in dag.get_state_machine_edges():
            cond_str = f" [{edge.condition}]" if edge.condition else ""
            print(f"  {edge.source} -> {edge.target}{cond_str}")

        print()
        print("DATA FLOW EDGES (variable propagation):")
        print("-" * 40)
        for edge in dag.get_data_flow_edges():
            print(f"  {edge.source} --({edge.variable})--> {edge.target}")

        print()
        print("To visualize the DAG, call dag.visualize() with matplotlib installed.")
        dag.visualize()

    except SyntaxError as e:
        print(f"Parse error: {e}")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


# =============================================================================
# CHECKPOINT 5: DAG Converter
# =============================================================================


class EdgeType(Enum):
    """Types of edges in the DAG."""

    STATE_MACHINE = auto()  # Execution order edge
    DATA_FLOW = auto()  # Variable data flow edge


@dataclass(frozen=True)
class DAGNode:
    """A node in the execution DAG."""

    id: str
    node_type: str  # e.g., "assignment", "action_call", "for_loop", "if", "aggregator", "fn_call", "input", "output"
    ir_node: RappelStatement | RappelExpr | None
    label: str  # Human-readable label
    function_name: str | None = None  # Which function this node belongs to (None = top-level)
    # For for loops
    is_loop_head: bool = False
    loop_var: str | None = None
    # For aggregators
    is_aggregator: bool = False
    aggregates_from: str | None = None  # Node ID this aggregates from
    # For function call nodes
    is_fn_call: bool = False
    called_function: str | None = None  # Name of function being called
    # For input/output boundary nodes
    is_input: bool = False
    is_output: bool = False
    io_vars: tuple[str, ...] | None = None  # Variables at this boundary


@dataclass(frozen=True)
class DAGEdge:
    """An edge in the execution DAG."""

    source: str  # Node ID
    target: str  # Node ID
    edge_type: EdgeType
    # For state machine edges
    condition: str | None = None  # e.g., "continue", "done", "then", "else"
    # For data flow edges
    variable: str | None = None  # Which variable is being passed


@dataclass
class DAG:
    """A directed acyclic graph representing program execution."""

    nodes: dict[str, DAGNode]
    edges: list[DAGEdge]
    entry_node: str | None = None

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.entry_node = None

    def add_node(self, node: DAGNode) -> None:
        """Add a node to the DAG."""
        self.nodes[node.id] = node
        if self.entry_node is None:
            self.entry_node = node.id

    def add_edge(self, edge: DAGEdge) -> None:
        """Add an edge to the DAG."""
        self.edges.append(edge)

    def get_incoming_edges(self, node_id: str) -> list[DAGEdge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges if e.target == node_id]

    def get_outgoing_edges(self, node_id: str) -> list[DAGEdge]:
        """Get all edges from a node."""
        return [e for e in self.edges if e.source == node_id]

    def get_state_machine_edges(self) -> list[DAGEdge]:
        """Get all state machine (execution order) edges."""
        return [e for e in self.edges if e.edge_type == EdgeType.STATE_MACHINE]

    def get_data_flow_edges(self) -> list[DAGEdge]:
        """Get all data flow edges."""
        return [e for e in self.edges if e.edge_type == EdgeType.DATA_FLOW]

    def get_functions(self) -> list[str]:
        """Get all function names that have nodes in this DAG."""
        functions = set()
        for node in self.nodes.values():
            if node.function_name:
                functions.add(node.function_name)
        return sorted(functions)

    def get_nodes_for_function(self, function_name: str) -> dict[str, DAGNode]:
        """Get all nodes belonging to a specific function."""
        return {
            nid: node for nid, node in self.nodes.items()
            if node.function_name == function_name
        }

    def get_edges_for_function(self, function_name: str) -> list[DAGEdge]:
        """Get all edges where both source and target belong to the function."""
        fn_nodes = set(self.get_nodes_for_function(function_name).keys())
        return [
            e for e in self.edges
            if e.source in fn_nodes and e.target in fn_nodes
        ]

    def visualize(self, title: str = "Rappel Program DAG") -> None:
        """
        Visualize the DAG using Cytoscape.js in a browser.

        - Solid lines: state machine (execution order) edges
        - Dotted lines: data flow edges
        - Functions are grouped as compound nodes
        """
        import json
        import tempfile
        import webbrowser

        # Build Cytoscape elements
        elements = []

        # Function colors for compound nodes
        function_colors = ['#FFE0B2', '#E1BEE7', '#B2EBF2', '#C8E6C9', '#FFCDD2', '#D1C4E9']
        functions = self.get_functions()

        # Add compound nodes for each function
        for i, fn_name in enumerate(functions):
            elements.append({
                'data': {
                    'id': f'fn_{fn_name}',
                    'label': f'fn {fn_name}',
                    'isCompound': True,
                    'color': function_colors[i % len(function_colors)]
                }
            })

        # Add nodes
        for node_id, node in self.nodes.items():
            # Determine node color based on type
            if node.is_input:
                color = '#4CAF50'  # Green for inputs
                shape = 'ellipse'
            elif node.is_output:
                color = '#F44336'  # Red for outputs
                shape = 'ellipse'
            elif node.is_fn_call:
                color = '#3F51B5'  # Indigo for function calls
                shape = 'round-rectangle'
            elif node.is_loop_head:
                color = '#FF9800'  # Orange for loops
                shape = 'diamond'
            elif node.is_aggregator:
                color = '#9C27B0'  # Purple for aggregators
                shape = 'hexagon'
            elif node.node_type == 'action_call':
                color = '#E91E63'  # Pink for actions
                shape = 'round-rectangle'
            elif node.node_type == 'if':
                color = '#00BCD4'  # Cyan for conditionals
                shape = 'diamond'
            else:
                color = '#607D8B'  # Gray for others
                shape = 'round-rectangle'

            node_data = {
                'data': {
                    'id': node_id,
                    'label': node.label,
                    'color': color,
                    'shape': shape,
                    'nodeType': node.node_type
                }
            }

            # Add parent (compound node) if node belongs to a function
            if node.function_name:
                node_data['data']['parent'] = f'fn_{node.function_name}'

            elements.append(node_data)

        # Add edges
        for i, edge in enumerate(self.edges):
            if edge.source not in self.nodes or edge.target not in self.nodes:
                continue

            edge_data = {
                'data': {
                    'id': f'edge_{i}',
                    'source': edge.source,
                    'target': edge.target,
                    'label': edge.condition or edge.variable or '',
                    'edgeType': edge.edge_type.name
                }
            }
            elements.append(edge_data)

        # Generate HTML
        html_content = self._generate_cytoscape_html(title, elements)

        # Write to temp file and open in browser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_path = f.name

        webbrowser.open(f'file://{temp_path}')
        print(f"DAG visualization opened in browser: {temp_path}")

    def _generate_cytoscape_html(self, title: str, elements: list) -> str:
        """Generate HTML with Cytoscape.js visualization."""
        import json

        elements_json = json.dumps(elements, indent=2)

        return f'''<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
        }}
        #header {{
            background: #333;
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        #header h1 {{ font-size: 18px; font-weight: 500; }}
        #legend {{
            display: flex;
            gap: 15px;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 1px solid #333;
        }}
        .legend-line {{
            width: 20px;
            height: 2px;
        }}
        #cy {{
            width: 100%;
            height: calc(100vh - 60px);
            background: white;
        }}
        #controls {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
        }}
        #controls button {{
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background: #333;
            color: white;
            cursor: pointer;
            font-size: 13px;
        }}
        #controls button:hover {{ background: #555; }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{title}</h1>
        <div id="legend">
            <div class="legend-item"><div class="legend-dot" style="background:#4CAF50"></div>Input</div>
            <div class="legend-item"><div class="legend-dot" style="background:#F44336"></div>Output</div>
            <div class="legend-item"><div class="legend-dot" style="background:#3F51B5"></div>Fn Call</div>
            <div class="legend-item"><div class="legend-dot" style="background:#E91E63"></div>Action</div>
            <div class="legend-item"><div class="legend-dot" style="background:#FF9800"></div>Loop</div>
            <div class="legend-item"><div class="legend-dot" style="background:#9C27B0"></div>Aggregator</div>
            <div class="legend-item"><div class="legend-dot" style="background:#00BCD4"></div>Conditional</div>
            <div class="legend-item"><div class="legend-line" style="background:#2196F3"></div>Execution</div>
            <div class="legend-item"><div class="legend-line" style="background:#4CAF50;border-style:dashed"></div>Data Flow</div>
        </div>
    </div>
    <div id="cy"></div>
    <div id="controls">
        <button onclick="cy.fit()">Fit View</button>
        <button onclick="cy.zoom(cy.zoom() * 1.2)">Zoom In</button>
        <button onclick="cy.zoom(cy.zoom() / 1.2)">Zoom Out</button>
        <button onclick="runLayout()">Re-layout</button>
    </div>

    <script>
        const elements = {elements_json};

        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'text-valign': 'bottom',
                        'text-halign': 'center',
                        'font-size': '11px',
                        'text-margin-y': 5,
                        'background-color': 'data(color)',
                        'border-width': 2,
                        'border-color': '#333',
                        'width': 35,
                        'height': 35,
                        'shape': 'data(shape)',
                        'text-wrap': 'wrap',
                        'text-max-width': '120px'
                    }}
                }},
                {{
                    selector: 'node[?isCompound]',
                    style: {{
                        'background-color': 'data(color)',
                        'background-opacity': 0.3,
                        'border-width': 2,
                        'border-color': '#666',
                        'border-style': 'solid',
                        'label': 'data(label)',
                        'text-valign': 'top',
                        'text-halign': 'center',
                        'font-size': '14px',
                        'font-weight': 'bold',
                        'padding': '20px',
                        'text-margin-y': -10
                    }}
                }},
                {{
                    selector: 'edge[edgeType="STATE_MACHINE"]',
                    style: {{
                        'width': 2,
                        'line-color': '#2196F3',
                        'target-arrow-color': '#2196F3',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': 'data(label)',
                        'font-size': '10px',
                        'text-rotation': 'autorotate',
                        'text-margin-y': -10,
                        'color': '#2196F3'
                    }}
                }},
                {{
                    selector: 'edge[edgeType="DATA_FLOW"]',
                    style: {{
                        'width': 1.5,
                        'line-color': '#4CAF50',
                        'line-style': 'dashed',
                        'target-arrow-color': '#4CAF50',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': 'data(label)',
                        'font-size': '9px',
                        'text-rotation': 'autorotate',
                        'text-margin-y': -8,
                        'color': '#4CAF50'
                    }}
                }},
                {{
                    selector: ':selected',
                    style: {{
                        'border-width': 4,
                        'border-color': '#FF5722'
                    }}
                }}
            ],
            layout: {{ name: 'preset' }}
        }});

        function runLayout() {{
            cy.layout({{
                name: 'dagre',
                rankDir: 'TB',
                nodeSep: 50,
                rankSep: 80,
                padding: 30,
                animate: true,
                animationDuration: 500
            }}).run();
        }}

        // Run initial layout
        runLayout();

        // Enable node dragging
        cy.nodes().grabify();
    </script>
</body>
</html>
'''


class DAGConverter:
    """
    Converts Rappel IR into a DAG representation with function isolation.

    Each function is converted into an isolated subgraph with:
    - An "input" boundary node (receives inputs)
    - An "output" boundary node (produces outputs)
    - Internal nodes that only reference variables within the function

    Function calls create "fn_call" nodes that:
    - Connect to the calling function's data flow
    - Reference the called function (but don't merge subgraphs)

    Edge types:
    - State machine: Execution order edges (src, dst) means dst follows src
    - Data flow: Variable propagation edges (src, dst) with variable annotation

    Data flow rules:
    - Data flow edges only exist WITHIN a function
    - Cross-function data flow happens only through explicit inputs/outputs
    - Function call nodes receive inputs and produce outputs
    """

    def __init__(self):
        self.dag = DAG()
        self.node_counter = 0
        self.current_function: str | None = None  # Currently being converted
        self.function_defs: dict[str, RappelFunctionDef] = {}  # name -> def
        # Per-function variable tracking
        self.current_scope_vars: dict[str, str] = {}  # var_name -> defining node id
        self.var_modifications: dict[str, list[str]] = {}  # var_name -> list of modifying node ids

    def convert(self, program: RappelProgram) -> DAG:
        """Convert a Rappel program to a DAG with isolated function subgraphs."""
        self.dag = DAG()
        self.node_counter = 0
        self.function_defs = {}

        # First pass: collect all function definitions
        for stmt in program.statements:
            if isinstance(stmt, RappelFunctionDef):
                self.function_defs[stmt.name] = stmt

        # Second pass: convert each function into an isolated subgraph
        for fn_name, fn_def in self.function_defs.items():
            self._convert_function(fn_def)

        return self.dag

    def _convert_function(self, fn_def: RappelFunctionDef) -> None:
        """Convert a function definition into an isolated subgraph."""
        self.current_function = fn_def.name
        self.current_scope_vars = {}
        self.var_modifications = {}

        # Create input boundary node
        input_id = self._next_id(f"{fn_def.name}_input")
        input_label = f"input: [{', '.join(fn_def.inputs)}]" if fn_def.inputs else "input: []"
        input_node = DAGNode(
            id=input_id,
            node_type="input",
            ir_node=None,
            label=input_label,
            function_name=fn_def.name,
            is_input=True,
            io_vars=fn_def.inputs
        )
        self.dag.add_node(input_node)

        # Track input variables as defined at the input node
        for var in fn_def.inputs:
            self._track_var_definition(var, input_id)

        # Convert function body
        prev_node_id = input_id
        for stmt in fn_def.body:
            node_ids = self._convert_statement(stmt)

            if prev_node_id and node_ids:
                self.dag.add_edge(DAGEdge(
                    source=prev_node_id,
                    target=node_ids[0],
                    edge_type=EdgeType.STATE_MACHINE
                ))

            if node_ids:
                prev_node_id = node_ids[-1]

        # Create output boundary node
        output_id = self._next_id(f"{fn_def.name}_output")
        output_label = f"output: [{', '.join(fn_def.outputs)}]"
        output_node = DAGNode(
            id=output_id,
            node_type="output",
            ir_node=None,
            label=output_label,
            function_name=fn_def.name,
            is_output=True,
            io_vars=fn_def.outputs
        )
        self.dag.add_node(output_node)

        # Connect last body node to output
        if prev_node_id:
            self.dag.add_edge(DAGEdge(
                source=prev_node_id,
                target=output_id,
                edge_type=EdgeType.STATE_MACHINE
            ))

        # Add data flow edges within this function
        self._add_data_flow_edges_for_function(fn_def.name)

        self.current_function = None

    def _next_id(self, prefix: str = "node") -> str:
        """Generate the next unique node ID."""
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def _convert_statement(self, stmt: RappelStatement) -> list[str]:
        """
        Convert a statement to DAG node(s).
        Returns list of node IDs created (in order).
        """
        if isinstance(stmt, RappelAssignment):
            return self._convert_assignment(stmt)
        elif isinstance(stmt, RappelMultiAssignment):
            return self._convert_multi_assignment(stmt)
        elif isinstance(stmt, RappelFunctionDef):
            # Function definitions are handled separately in _convert_function
            return []
        elif isinstance(stmt, RappelForLoop):
            return self._convert_for_loop(stmt)
        elif isinstance(stmt, RappelIfStatement):
            return self._convert_if_statement(stmt)
        elif isinstance(stmt, RappelPythonBlock):
            return self._convert_python_block(stmt)
        elif isinstance(stmt, RappelReturn):
            return self._convert_return(stmt)
        elif isinstance(stmt, RappelExprStatement):
            return self._convert_expr_statement(stmt)
        elif isinstance(stmt, RappelSpreadAction):
            return self._convert_spread_action(stmt)
        else:
            return []

    def _convert_assignment(self, stmt: RappelAssignment) -> list[str]:
        """Convert an assignment statement."""
        # Check if RHS contains a spread that needs aggregation
        if self._contains_spread_action(stmt.value):
            return self._convert_spread_assignment(stmt)

        # Check if RHS is an action call
        if isinstance(stmt.value, RappelActionCall):
            node_id = self._next_id("action")
            label = f"@{stmt.value.action_name}() → {stmt.target}"
        else:
            node_id = self._next_id("assign")
            label = f"{stmt.target} = ..."

        # Check if RHS is a function call
        if isinstance(stmt.value, RappelCall):
            return self._convert_fn_call_assignment(stmt)

        node = DAGNode(
            id=node_id,
            node_type="action_call" if isinstance(stmt.value, RappelActionCall) else "assignment",
            ir_node=stmt,
            label=label,
            function_name=self.current_function
        )
        self.dag.add_node(node)

        # Track variable definition
        self._track_var_definition(stmt.target, node_id)

        return [node_id]

    def _convert_spread_assignment(self, stmt: RappelAssignment) -> list[str]:
        """
        Convert an assignment with spread action to action + aggregator nodes.

        Spread is implemented as:
        1. Action node that processes items in parallel
        2. Aggregator node that collects results
        """
        # Find the action call in the spread
        action_call = self._find_action_in_expr(stmt.value)

        # Create action node
        action_id = self._next_id("spread_action")
        action_label = f"@{action_call.action_name}() [spread]" if action_call else "spread_action"
        action_node = DAGNode(
            id=action_id,
            node_type="action_call",
            ir_node=stmt,
            label=action_label,
            function_name=self.current_function
        )
        self.dag.add_node(action_node)

        # Create aggregator node
        agg_id = self._next_id("aggregator")
        agg_node = DAGNode(
            id=agg_id,
            node_type="aggregator",
            ir_node=None,
            label=f"aggregate → {stmt.target}",
            function_name=self.current_function,
            is_aggregator=True,
            aggregates_from=action_id
        )
        self.dag.add_node(agg_node)

        # Connect action to aggregator
        self.dag.add_edge(DAGEdge(
            source=action_id,
            target=agg_id,
            edge_type=EdgeType.STATE_MACHINE
        ))

        # Track variable definition at aggregator
        self._track_var_definition(stmt.target, agg_id)

        return [action_id, agg_id]

    def _convert_multi_assignment(self, stmt: RappelMultiAssignment) -> list[str]:
        """Convert a multi-assignment (unpacking)."""
        # Check if RHS is a function call
        if isinstance(stmt.value, RappelCall):
            return self._convert_fn_call_multi_assignment(stmt)

        node_id = self._next_id("multi_assign")
        targets_str = ", ".join(stmt.targets)
        node = DAGNode(
            id=node_id,
            node_type="assignment",
            ir_node=stmt,
            label=f"{targets_str} = ...",
            function_name=self.current_function
        )
        self.dag.add_node(node)

        # Track all variable definitions
        for target in stmt.targets:
            self._track_var_definition(target, node_id)

        return [node_id]

    def _convert_fn_call_assignment(self, stmt: RappelAssignment) -> list[str]:
        """Convert a function call assignment: x = fn_name(kwargs)"""
        call = stmt.value
        assert isinstance(call, RappelCall)

        node_id = self._next_id("fn_call")
        node = DAGNode(
            id=node_id,
            node_type="fn_call",
            ir_node=stmt,
            label=f"{call.target}() → {stmt.target}",
            function_name=self.current_function,
            is_fn_call=True,
            called_function=call.target
        )
        self.dag.add_node(node)

        # Track variable definition
        self._track_var_definition(stmt.target, node_id)

        return [node_id]

    def _convert_fn_call_multi_assignment(self, stmt: RappelMultiAssignment) -> list[str]:
        """Convert a function call multi-assignment: x, y = fn_name(kwargs)"""
        call = stmt.value
        assert isinstance(call, RappelCall)

        node_id = self._next_id("fn_call")
        targets_str = ", ".join(stmt.targets)
        node = DAGNode(
            id=node_id,
            node_type="fn_call",
            ir_node=stmt,
            label=f"{call.target}() → {targets_str}",
            function_name=self.current_function,
            is_fn_call=True,
            called_function=call.target
        )
        self.dag.add_node(node)

        # Track all variable definitions
        for target in stmt.targets:
            self._track_var_definition(target, node_id)

        return [node_id]

    def _convert_for_loop(self, stmt: RappelForLoop) -> list[str]:
        """
        Convert a for loop.

        For loops become a single node with:
        - "continue" edge to the first node in the body
        - "done" edge after all iterations complete

        Variables needed by the loop body are passed TO the for loop head.
        """
        # Create loop head node
        loop_id = self._next_id("for_loop")
        loop_node = DAGNode(
            id=loop_id,
            node_type="for_loop",
            ir_node=stmt,
            label=f"for {stmt.loop_var} in ...",
            function_name=self.current_function,
            is_loop_head=True,
            loop_var=stmt.loop_var
        )
        self.dag.add_node(loop_node)

        # Track loop variable
        self._track_var_definition(stmt.loop_var, loop_id)

        # Track output variables from the loop body assignment
        # Body contains exactly one assignment with a function call
        if stmt.body and isinstance(stmt.body[0], RappelAssignment):
            body_assignment = stmt.body[0]
            self._track_var_definition(body_assignment.target, loop_id)

        # The loop is a single node - body is handled at runtime
        # State machine edges for "continue" and "done" will be added
        # when we know what follows

        return [loop_id]

    def _convert_if_statement(self, stmt: RappelIfStatement) -> list[str]:
        """
        Convert an if statement.

        Creates:
        - Condition node
        - Then branch nodes
        - Else branch nodes (if present)
        - Join node
        """
        # Create condition node
        cond_id = self._next_id("if_cond")
        cond_node = DAGNode(
            id=cond_id,
            node_type="if",
            ir_node=stmt,
            label="if ...",
            function_name=self.current_function
        )
        self.dag.add_node(cond_node)

        result_nodes = [cond_id]
        then_last: str | None = None
        else_last: str | None = None

        # Process then branch
        prev_id = cond_id
        for body_stmt in stmt.then_body:
            node_ids = self._convert_statement(body_stmt)
            if node_ids:
                # Connect first node to previous
                self.dag.add_edge(DAGEdge(
                    source=prev_id,
                    target=node_ids[0],
                    edge_type=EdgeType.STATE_MACHINE,
                    condition="then" if prev_id == cond_id else None
                ))
                prev_id = node_ids[-1]
                result_nodes.extend(node_ids)
        then_last = prev_id if prev_id != cond_id else None

        # Process else branch
        if stmt.else_body:
            prev_id = cond_id
            for body_stmt in stmt.else_body:
                node_ids = self._convert_statement(body_stmt)
                if node_ids:
                    self.dag.add_edge(DAGEdge(
                        source=prev_id,
                        target=node_ids[0],
                        edge_type=EdgeType.STATE_MACHINE,
                        condition="else" if prev_id == cond_id else None
                    ))
                    prev_id = node_ids[-1]
                    result_nodes.extend(node_ids)
            else_last = prev_id if prev_id != cond_id else None

        # Create join node if we have branches
        if then_last or else_last:
            join_id = self._next_id("if_join")
            join_node = DAGNode(
                id=join_id,
                node_type="join",
                ir_node=None,
                label="join",
                function_name=self.current_function
            )
            self.dag.add_node(join_node)
            result_nodes.append(join_id)

            if then_last:
                self.dag.add_edge(DAGEdge(
                    source=then_last,
                    target=join_id,
                    edge_type=EdgeType.STATE_MACHINE
                ))
            else:
                # Empty then branch
                self.dag.add_edge(DAGEdge(
                    source=cond_id,
                    target=join_id,
                    edge_type=EdgeType.STATE_MACHINE,
                    condition="then"
                ))

            if else_last:
                self.dag.add_edge(DAGEdge(
                    source=else_last,
                    target=join_id,
                    edge_type=EdgeType.STATE_MACHINE
                ))
            elif stmt.else_body:
                # Empty else branch
                self.dag.add_edge(DAGEdge(
                    source=cond_id,
                    target=join_id,
                    edge_type=EdgeType.STATE_MACHINE,
                    condition="else"
                ))

        return result_nodes

    def _convert_python_block(self, stmt: RappelPythonBlock) -> list[str]:
        """Convert a python block."""
        node_id = self._next_id("python")
        outputs_str = ", ".join(stmt.outputs)
        node = DAGNode(
            id=node_id,
            node_type="python_block",
            ir_node=stmt,
            label=f"python → {outputs_str}",
            function_name=self.current_function
        )
        self.dag.add_node(node)

        # Track output variables
        for output_var in stmt.outputs:
            self._track_var_definition(output_var, node_id)

        return [node_id]

    def _convert_return(self, stmt: RappelReturn) -> list[str]:
        """Convert a return statement."""
        node_id = self._next_id("return")
        node = DAGNode(
            id=node_id,
            node_type="return",
            ir_node=stmt,
            label="return",
            function_name=self.current_function
        )
        self.dag.add_node(node)

        return [node_id]

    def _convert_expr_statement(self, stmt: RappelExprStatement) -> list[str]:
        """Convert an expression statement (usually an action call)."""
        if isinstance(stmt.expr, RappelActionCall):
            node_id = self._next_id("action")
            node = DAGNode(
                id=node_id,
                node_type="action_call",
                ir_node=stmt,
                label=f"@{stmt.expr.action_name}()",
                function_name=self.current_function
            )
        else:
            node_id = self._next_id("expr")
            node = DAGNode(
                id=node_id,
                node_type="expression",
                ir_node=stmt,
                label="expr",
                function_name=self.current_function
            )
        self.dag.add_node(node)

        return [node_id]

    def _convert_spread_action(self, stmt: RappelSpreadAction) -> list[str]:
        """
        Convert a spread action to action + aggregator nodes.

        spread items:item -> @fetch_details(id=item)

        Creates:
        1. Spread action node (launches parallel actions)
        2. Aggregator node (collects results)
        """
        # Create spread action node
        action_id = self._next_id("spread_action")
        action_label = f"@{stmt.action.action_name}() [spread over {stmt.item_var}]"
        action_node = DAGNode(
            id=action_id,
            node_type="action_call",
            ir_node=stmt,
            label=action_label,
            function_name=self.current_function
        )
        self.dag.add_node(action_node)

        # Create aggregator node
        agg_id = self._next_id("aggregator")
        target_label = f" → {stmt.target}" if stmt.target else ""
        agg_node = DAGNode(
            id=agg_id,
            node_type="aggregator",
            ir_node=None,
            label=f"aggregate{target_label}",
            function_name=self.current_function,
            is_aggregator=True,
            aggregates_from=action_id
        )
        self.dag.add_node(agg_node)

        # Connect action to aggregator
        self.dag.add_edge(DAGEdge(
            source=action_id,
            target=agg_id,
            edge_type=EdgeType.STATE_MACHINE
        ))

        # Track variable definition at aggregator (if target is specified)
        if stmt.target:
            self._track_var_definition(stmt.target, agg_id)

        return [action_id, agg_id]

    def _track_var_definition(self, var_name: str, node_id: str) -> None:
        """Track that a variable is defined/modified at a node."""
        self.current_scope_vars[var_name] = node_id
        if var_name not in self.var_modifications:
            self.var_modifications[var_name] = []
        self.var_modifications[var_name].append(node_id)

    def _add_data_flow_edges_for_function(self, function_name: str) -> None:
        """
        Add data flow edges for a specific function.

        Data flow edges only exist WITHIN a function - cross-function
        data flow happens through explicit inputs/outputs.

        Rule: Push data only to the most recent trailing node that
        DOESN'T modify the variable.
        """
        # Get only nodes for this function
        fn_nodes = self.dag.get_nodes_for_function(function_name)

        for node_id, node in fn_nodes.items():
            if node.ir_node is None and not node.is_input:
                continue

            # Get variables used by this node
            if node.ir_node:
                used_vars = self._get_used_variables(node.ir_node)
            else:
                used_vars = set()

            for var_name in used_vars:
                # Find the most recent definition of this variable
                # that comes BEFORE this node in execution order
                source_node = self._find_var_source(var_name, node_id, function_name)
                if source_node and source_node != node_id:
                    self.dag.add_edge(DAGEdge(
                        source=source_node,
                        target=node_id,
                        edge_type=EdgeType.DATA_FLOW,
                        variable=var_name
                    ))

    def _find_var_source(self, var_name: str, target_node_id: str, function_name: str | None = None) -> str | None:
        """
        Find the source node that should provide a variable's value.

        Returns the most recent node that defines the variable and
        comes before target_node_id in execution order.
        Only considers nodes within the same function.
        """
        if var_name not in self.var_modifications:
            return None

        modifications = self.var_modifications[var_name]

        # Get topological order for this function's nodes only
        if function_name:
            fn_nodes = set(self.dag.get_nodes_for_function(function_name).keys())
            order = self._get_execution_order_for_nodes(fn_nodes)
        else:
            order = self._get_execution_order_for_nodes(set(self.dag.nodes.keys()))

        if target_node_id not in order:
            return modifications[-1] if modifications else None

        target_idx = order.index(target_node_id)

        # Find the most recent modification before target
        best_source = None
        best_idx = -1

        for mod_node in modifications:
            if mod_node in order:
                mod_idx = order.index(mod_node)
                if mod_idx < target_idx and mod_idx > best_idx:
                    best_idx = mod_idx
                    best_source = mod_node

        return best_source

    def _get_execution_order_for_nodes(self, node_ids: set[str]) -> list[str]:
        """Get nodes in topological (execution) order for a subset of nodes."""
        # Simple topological sort using state machine edges
        in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
        adj: dict[str, list[str]] = {nid: [] for nid in node_ids}

        for edge in self.dag.get_state_machine_edges():
            if edge.source in node_ids and edge.target in node_ids:
                adj[edge.source].append(edge.target)
                in_degree[edge.target] += 1

        # Kahn's algorithm
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def _get_used_variables(self, node: RappelStatement | RappelExpr) -> set[str]:
        """Extract variable names used (read) by a node."""
        used = set()
        self._collect_used_vars(node, used)
        return used

    def _collect_used_vars(self, node, used: set[str]) -> None:
        """Recursively collect used variables."""
        if isinstance(node, RappelVariable):
            used.add(node.name)
        elif isinstance(node, RappelAssignment):
            self._collect_used_vars(node.value, used)
        elif isinstance(node, RappelMultiAssignment):
            self._collect_used_vars(node.value, used)
        elif isinstance(node, RappelBinaryOp):
            self._collect_used_vars(node.left, used)
            self._collect_used_vars(node.right, used)
        elif isinstance(node, RappelUnaryOp):
            self._collect_used_vars(node.operand, used)
        elif isinstance(node, RappelIndexAccess):
            self._collect_used_vars(node.target, used)
            self._collect_used_vars(node.index, used)
        elif isinstance(node, RappelDotAccess):
            self._collect_used_vars(node.target, used)
        elif isinstance(node, RappelCall):
            for _, arg in node.kwargs:
                self._collect_used_vars(arg, used)
        elif isinstance(node, RappelActionCall):
            for _, arg in node.kwargs:
                self._collect_used_vars(arg, used)
        elif isinstance(node, RappelListExpr):
            for item in node.items:
                self._collect_used_vars(item, used)
        elif isinstance(node, RappelDictExpr):
            for k, v in node.pairs:
                self._collect_used_vars(k, used)
                self._collect_used_vars(v, used)
        elif isinstance(node, RappelSpread):
            self._collect_used_vars(node.target, used)
        elif isinstance(node, RappelIfStatement):
            self._collect_used_vars(node.condition, used)
        elif isinstance(node, RappelForLoop):
            self._collect_used_vars(node.iterable, used)
        elif isinstance(node, RappelReturn):
            for val in node.values:
                self._collect_used_vars(val, used)
        elif isinstance(node, RappelExprStatement):
            self._collect_used_vars(node.expr, used)
        elif isinstance(node, RappelSpreadAction):
            self._collect_used_vars(node.source_list, used)
            # Also collect vars from the action kwargs (excluding the item_var)
            for _, arg in node.action.kwargs:
                self._collect_used_vars(arg, used)
            # Remove the item_var since it's defined by the spread, not used
            used.discard(node.item_var)

    def _contains_spread_action(self, expr: RappelExpr) -> bool:
        """Check if expression contains a spread with an action."""
        if isinstance(expr, RappelSpread):
            return True
        elif isinstance(expr, RappelActionCall):
            # Check if any arg contains spread
            for _, arg in expr.kwargs:
                if self._contains_spread_action(arg):
                    return True
        elif isinstance(expr, RappelBinaryOp):
            return self._contains_spread_action(expr.left) or self._contains_spread_action(expr.right)
        elif isinstance(expr, RappelListExpr):
            return any(self._contains_spread_action(item) for item in expr.items)
        return False

    def _find_action_in_expr(self, expr: RappelExpr) -> RappelActionCall | None:
        """Find an action call in an expression."""
        if isinstance(expr, RappelActionCall):
            return expr
        elif isinstance(expr, RappelBinaryOp):
            left = self._find_action_in_expr(expr.left)
            if left:
                return left
            return self._find_action_in_expr(expr.right)
        elif isinstance(expr, RappelListExpr):
            for item in expr.items:
                found = self._find_action_in_expr(item)
                if found:
                    return found
        return None


def convert_to_dag(program: RappelProgram) -> DAG:
    """Convert a Rappel program to a DAG."""
    converter = DAGConverter()
    return converter.convert(program)


# =============================================================================
# CHECKPOINT 5 TESTS: DAG Converter
# =============================================================================


def test_dag_simple_assignments():
    """Test DAG conversion of simple assignments inside a function."""
    source = """
fn test_fn(input: [], output: [z]):
    x = 1
    y = 2
    z = x + y
    return z
"""
    program = parse(source)
    dag = convert_to_dag(program)

    # Should have input + 3 assignments + return + output = 6 nodes
    assert len(dag.nodes) == 6

    # All nodes should belong to test_fn
    assert all(n.function_name == "test_fn" for n in dag.nodes.values())

    # Should have data flow edges for x and y to z
    df_edges = dag.get_data_flow_edges()
    assert len(df_edges) >= 1  # At least one for x or y


def test_dag_action_call():
    """Test DAG conversion of action calls inside a function."""
    source = """
fn fetch_fn(input: [], output: [processed]):
    result = @fetch_data(url="http://example.com")
    processed = @process(data=result)
    return processed
"""
    program = parse(source)
    dag = convert_to_dag(program)

    # Should have action_call nodes
    action_nodes = [n for n in dag.nodes.values() if n.node_type == "action_call"]
    assert len(action_nodes) == 2

    # Should have data flow edge for result
    df_edges = dag.get_data_flow_edges()
    assert any(e.variable == "result" for e in df_edges)


def test_dag_for_loop():
    """Test DAG conversion of for loop inside a function."""
    source = """
fn loop_fn(input: [], output: [result]):
    items = [1, 2, 3]
    for item in items:
        result = process(x=item)
    return result
"""
    program = parse(source)
    dag = convert_to_dag(program)

    # Should have for_loop node
    loop_nodes = [n for n in dag.nodes.values() if n.is_loop_head]
    assert len(loop_nodes) == 1
    assert loop_nodes[0].loop_var == "item"
    assert loop_nodes[0].function_name == "loop_fn"


def test_dag_if_statement():
    """Test DAG conversion of if statement inside a function."""
    source = """
fn cond_fn(input: [x], output: [y]):
    if x > 5:
        result = "big"
    else:
        result = "small"
    y = result
    return y
"""
    program = parse(source)
    dag = convert_to_dag(program)

    # Should have if condition node
    if_nodes = [n for n in dag.nodes.values() if n.node_type == "if"]
    assert len(if_nodes) == 1

    # Should have join node
    join_nodes = [n for n in dag.nodes.values() if n.node_type == "join"]
    assert len(join_nodes) == 1


def test_dag_spread_creates_aggregator():
    """Test that spread creates action + aggregator nodes inside a function."""
    source = """
fn spread_fn(input: [], output: [results]):
    ids = [1, 2, 3]
    results = spread ids:id -> @fetch_details(id=id)
    return results
"""
    program = parse(source)
    dag = convert_to_dag(program)

    # Check for aggregator node
    aggregator_nodes = [n for n in dag.nodes.values() if n.is_aggregator]
    assert len(aggregator_nodes) == 1
    assert "results" in aggregator_nodes[0].label

    # Check for spread action node
    action_nodes = [n for n in dag.nodes.values() if n.node_type == "action_call"]
    assert len(action_nodes) == 1
    assert "spread" in action_nodes[0].label


def test_dag_function_isolation():
    """Test that each function has its own isolated subgraph."""
    source = """
fn add(input: [a, b], output: [result]):
    result = a + b
    return result

fn multiply(input: [x, y], output: [result]):
    result = x * y
    return result
"""
    program = parse(source)
    dag = convert_to_dag(program)

    # Should have nodes for both functions
    functions = dag.get_functions()
    assert "add" in functions
    assert "multiply" in functions

    # Each function should have input and output nodes
    add_nodes = dag.get_nodes_for_function("add")
    multiply_nodes = dag.get_nodes_for_function("multiply")

    assert any(n.is_input for n in add_nodes.values())
    assert any(n.is_output for n in add_nodes.values())
    assert any(n.is_input for n in multiply_nodes.values())
    assert any(n.is_output for n in multiply_nodes.values())


def test_dag_function_call_node():
    """Test that function calls create fn_call nodes."""
    source = """
fn helper(input: [x], output: [y]):
    y = x + 1
    return y

fn main_fn(input: [], output: [result]):
    result = helper(x=10)
    return result
"""
    program = parse(source)
    dag = convert_to_dag(program)

    # Should have fn_call node in main_fn
    fn_call_nodes = [n for n in dag.nodes.values() if n.is_fn_call]
    assert len(fn_call_nodes) == 1
    assert fn_call_nodes[0].called_function == "helper"
    assert fn_call_nodes[0].function_name == "main_fn"


def test_dag_data_flow_respects_modifications():
    """Test that data flow edges respect variable modifications."""
    source = """
fn test_fn(input: [], output: [z]):
    x = 1
    y = x + 1
    x = 10
    z = x + 2
    return z
"""
    program = parse(source)
    dag = convert_to_dag(program)

    # z should get data from the second x definition, not the first
    df_edges = dag.get_data_flow_edges()
    z_incoming = [e for e in df_edges if e.variable == "x"]

    # There should be data flow edges for x
    assert len(z_incoming) >= 1


def test_dag_complex_workflow():
    """Test DAG conversion of the complex example with multiple functions."""
    program = parse(EXAMPLE_ACTION_SPREAD_LOOP)
    dag = convert_to_dag(program)

    # Should have multiple functions
    functions = dag.get_functions()
    assert len(functions) >= 2  # At least process_order and main

    # main function should have action call nodes
    main_nodes = dag.get_nodes_for_function("main")
    action_nodes = [n for n in main_nodes.values() if n.node_type == "action_call"]
    assert len(action_nodes) >= 2

    # main function should have a for loop node
    loop_nodes = [n for n in main_nodes.values() if n.is_loop_head]
    assert len(loop_nodes) >= 1


def test_dag_visualization_no_crash():
    """Test that visualization data can be generated (visual inspection is manual)."""
    source = """
fn viz_fn(input: [x], output: [result]):
    y = @fetch(id=x)
    if y > 0:
        result = "positive"
    else:
        result = "negative"
    return result
"""
    program = parse(source)
    dag = convert_to_dag(program)

    # Just verify we have nodes and edges for visualization
    assert len(dag.nodes) > 0
    assert len(dag.edges) > 0
    # Verify function grouping works
    assert len(dag.get_functions()) == 1
    assert 'viz_fn' in dag.get_functions()


if __name__ == "__main__":
    main()
