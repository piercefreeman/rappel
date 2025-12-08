"""
Rappel Lexer - Tokenizes source code into tokens.

Handles:
- Keywords: fn, for, in, if, else, try, except, return, input, output, spread, parallel
- Operators: =, +, -, *, /, ==, !=, <, >, <=, >=, and, or, not, ->
- Literals: strings, numbers, booleans
- Identifiers
- Indentation-based blocks
- Spread operator: ...
"""

from __future__ import annotations

from .tokens import Token, TokenType


class RappelLexer:
    """Lexer for the Rappel language."""

    KEYWORDS = {
        "fn": TokenType.FN,
        "for": TokenType.FOR,
        "in": TokenType.IN,
        "if": TokenType.IF,
        "else": TokenType.ELSE,
        "try": TokenType.TRY,
        "except": TokenType.EXCEPT,
        "return": TokenType.RETURN,
        "input": TokenType.INPUT,
        "output": TokenType.OUTPUT,
        "spread": TokenType.SPREAD,
        "parallel": TokenType.PARALLEL,
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
