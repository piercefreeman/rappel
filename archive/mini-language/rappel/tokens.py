"""Token types and Token class for the Rappel lexer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    """All token types in the Rappel language."""

    # Literals
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()

    # Keywords
    FN = auto()  # function definition
    FOR = auto()  # for loop
    IN = auto()  # for x in list
    IF = auto()  # conditional
    ELSE = auto()  # else branch
    TRY = auto()  # try block
    EXCEPT = auto()  # except handler
    RETURN = auto()  # return statement
    INPUT = auto()  # input declaration
    OUTPUT = auto()  # output declaration
    SPREAD = auto()  # spread keyword
    PARALLEL = auto()  # parallel block
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
