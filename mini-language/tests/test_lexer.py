"""Tests for Rappel Lexer."""

from rappel import RappelLexer, TokenType


def test_lexer_simple_assignment():
    """Test lexing a simple assignment."""
    source = "x = 42"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.IDENTIFIER
    assert tokens[0].value == "x"
    assert tokens[1].type == TokenType.ASSIGN
    assert tokens[2].type == TokenType.NUMBER
    assert tokens[2].value == 42


def test_lexer_string_literal():
    """Test lexing string literals."""
    source = 'name = "Alice"'
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[2].type == TokenType.STRING
    assert tokens[2].value == "Alice"


def test_lexer_list_literal():
    """Test lexing list literals."""
    source = "items = [1, 2, 3]"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[2].type == TokenType.LBRACKET
    assert tokens[3].type == TokenType.NUMBER
    assert tokens[3].value == 1
    assert tokens[4].type == TokenType.COMMA


def test_lexer_dict_literal():
    """Test lexing dict literals."""
    source = '{"key": "value"}'
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.LBRACE
    assert tokens[1].type == TokenType.STRING
    assert tokens[1].value == "key"
    assert tokens[2].type == TokenType.COLON


def test_lexer_keywords():
    """Test lexing keywords."""
    source = "fn for in if else return input output"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.FN
    assert tokens[1].type == TokenType.FOR
    assert tokens[2].type == TokenType.IN
    assert tokens[3].type == TokenType.IF
    assert tokens[4].type == TokenType.ELSE
    assert tokens[5].type == TokenType.RETURN
    assert tokens[6].type == TokenType.INPUT
    assert tokens[7].type == TokenType.OUTPUT


def test_lexer_operators():
    """Test lexing operators."""
    source = "a + b - c * d / e == f != g < h > i <= j >= k and l or m"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    token_types = [t.type for t in tokens if t.type not in (TokenType.IDENTIFIER, TokenType.EOF, TokenType.NEWLINE)]
    assert TokenType.PLUS in token_types
    assert TokenType.MINUS in token_types
    assert TokenType.STAR in token_types
    assert TokenType.SLASH in token_types
    assert TokenType.EQ in token_types
    assert TokenType.NEQ in token_types
    assert TokenType.LT in token_types
    assert TokenType.GT in token_types
    assert TokenType.LTE in token_types
    assert TokenType.GTE in token_types
    assert TokenType.AND in token_types
    assert TokenType.OR in token_types


def test_lexer_boolean():
    """Test lexing boolean literals."""
    source = "a = true\nb = false"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    bool_tokens = [t for t in tokens if t.type == TokenType.BOOLEAN]
    assert len(bool_tokens) == 2
    assert bool_tokens[0].value is True
    assert bool_tokens[1].value is False


def test_lexer_indentation():
    """Test lexing with indentation."""
    source = """if x:
    y = 1
z = 2"""
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    token_types = [t.type for t in tokens]
    assert TokenType.INDENT in token_types
    assert TokenType.DEDENT in token_types


def test_lexer_comments():
    """Test that comments are skipped."""
    source = """x = 1  # this is a comment
# another comment
y = 2"""
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    identifiers = [t for t in tokens if t.type == TokenType.IDENTIFIER]
    assert len(identifiers) == 2
    assert identifiers[0].value == "x"
    assert identifiers[1].value == "y"


def test_lexer_function_def():
    """Test lexing a function definition."""
    source = "fn add(input: [a, b], output: [result]):"
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
    source = "items = items + [4, 5]"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert TokenType.PLUS in [t.type for t in tokens]
    assert TokenType.LBRACKET in [t.type for t in tokens]


def test_lexer_dict_concat():
    """Test lexing dict concatenation."""
    source = 'config = config + {"key": "value"}'
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert TokenType.PLUS in [t.type for t in tokens]
    assert TokenType.LBRACE in [t.type for t in tokens]


def test_lexer_ellipsis_operator():
    """Test lexing ellipsis (spread) operator."""
    source = "[...items, 1, 2]"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[1].type == TokenType.ELLIPSIS


def test_lexer_index_access():
    """Test lexing index access."""
    source = "x = items[0]"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert TokenType.LBRACKET in [t.type for t in tokens]
    assert TokenType.RBRACKET in [t.type for t in tokens]


def test_lexer_action_call():
    """Test lexing action call with @ syntax."""
    source = "@fetch_data(url=x)"
    lexer = RappelLexer(source)
    tokens = lexer.tokenize()

    assert tokens[0].type == TokenType.AT
    assert tokens[1].type == TokenType.IDENTIFIER
    assert tokens[1].value == "fetch_data"
