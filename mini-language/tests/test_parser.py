"""Tests for Rappel Parser."""

import pytest

from rappel import (
    parse,
    RappelAssignment,
    RappelMultiAssignment,
    RappelVariable,
    RappelLiteral,
    RappelListExpr,
    RappelDictExpr,
    RappelBinaryOp,
    RappelUnaryOp,
    RappelIndexAccess,
    RappelDotAccess,
    RappelSpread,
    RappelCall,
    RappelActionCall,
    RappelFunctionDef,
    RappelForLoop,
    RappelIfStatement,
    RappelReturn,
    RappelSpreadAction,
)


def test_parser_simple_assignment():
    """Test parsing a simple assignment."""
    source = "x = 42"
    program = parse(source)

    assert len(program.statements) == 1
    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert stmt.target == "x"
    assert isinstance(stmt.value, RappelLiteral)


def test_parser_string_assignment():
    """Test parsing a string assignment."""
    source = 'name = "Alice"'
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelAssignment)
    assert isinstance(stmt.value, RappelLiteral)


def test_parser_list_literal():
    """Test parsing a list literal."""
    source = "items = [1, 2, 3]"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelListExpr)
    assert len(stmt.value.items) == 3


def test_parser_dict_literal():
    """Test parsing a dict literal."""
    source = '{"key": "value", "num": 42}'
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.expr, RappelDictExpr)
    assert len(stmt.expr.pairs) == 2


def test_parser_list_concat():
    """Test parsing list concatenation."""
    source = "items = items + [4]"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelBinaryOp)
    assert stmt.value.op == "+"


def test_parser_dict_concat():
    """Test parsing dict concatenation."""
    source = 'config = config + {"timeout": 30}'
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelBinaryOp)


def test_parser_index_access():
    """Test parsing index access."""
    source = "first = items[0]"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelIndexAccess)


def test_parser_dict_key_access():
    """Test parsing dict key access."""
    source = 'value = config["key"]'
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelIndexAccess)


def test_parser_dot_access():
    """Test parsing dot access."""
    source = "name = user.name"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelDotAccess)
    assert stmt.value.field == "name"


def test_parser_function_call():
    """Test parsing function call with kwargs."""
    source = "result = calculate(x=10, y=20)"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelCall)
    assert stmt.value.target == "calculate"
    assert len(stmt.value.kwargs) == 2


def test_parser_function_call_colon_syntax():
    """Test parsing function call with colon syntax for kwargs."""
    source = "result = calculate(x: 10, y: 20)"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelCall)
    assert len(stmt.value.kwargs) == 2


def test_parser_function_call_rejects_positional():
    """Test that positional args are rejected."""
    source = "result = calculate(10, 20)"
    with pytest.raises(SyntaxError):
        parse(source)


def test_parser_spread_operator():
    """Test parsing spread operator."""
    source = "combined = [...base, 1, 2]"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelListExpr)
    assert isinstance(stmt.value.items[0], RappelSpread)


def test_parser_multi_assignment():
    """Test parsing multi-assignment (unpacking)."""
    source = "a, b = get_values()"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelMultiAssignment)
    assert stmt.targets == ("a", "b")


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


def test_parser_action_call():
    """Test parsing action call with @ syntax."""
    source = 'response = @fetch_url(url="https://example.com")'
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelActionCall)
    assert stmt.value.action_name == "fetch_url"


def test_parser_spread_action():
    """Test parsing spread action statement."""
    source = "results = spread items:item -> @fetch_details(id=item)"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelSpreadAction)
    assert stmt.target == "results"
    assert stmt.item_var == "item"
    assert stmt.action.action_name == "fetch_details"


def test_parser_spread_action_no_target():
    """Test parsing spread action without assignment target."""
    source = "spread items:item -> @process(id=item)"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelSpreadAction)
    assert stmt.target is None


def test_parser_for_loop():
    """Test parsing for loop with single function call body."""
    source = """for item in items:
    result = process_item(x=item)"""

    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelForLoop)
    assert stmt.loop_vars == ("item",)
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
    result = "non-positive"
"""
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt, RappelIfStatement)
    assert stmt.else_body is not None


def test_parser_return_single():
    """Test parsing single value return."""
    source = """fn test(input: [], output: [x]):
    x = 1
    return x"""
    program = parse(source)

    fn = program.statements[0]
    ret = fn.body[-1]
    assert isinstance(ret, RappelReturn)
    assert len(ret.values) == 1


def test_parser_return_multiple():
    """Test parsing multiple value return."""
    source = """fn test(input: [], output: [a, b]):
    a = 1
    b = 2
    return [a, b]"""
    program = parse(source)

    fn = program.statements[0]
    ret = fn.body[-1]
    assert isinstance(ret, RappelReturn)
    assert len(ret.values) == 2


def test_parser_comparison_operators():
    """Test parsing comparison operators."""
    source = "result = a == b"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelBinaryOp)
    assert stmt.value.op == "=="


def test_parser_arithmetic():
    """Test parsing arithmetic expressions."""
    source = "result = a + b * c - d / e"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelBinaryOp)


def test_parser_unary_not():
    """Test parsing unary not."""
    source = "result = not x"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelUnaryOp)
    assert stmt.value.op == "not"


def test_parser_unary_minus():
    """Test parsing unary minus."""
    source = "result = -x"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelUnaryOp)
    assert stmt.value.op == "-"


def test_parser_parenthesized():
    """Test parsing parenthesized expressions."""
    source = "result = (a + b) * c"
    program = parse(source)

    stmt = program.statements[0]
    assert isinstance(stmt.value, RappelBinaryOp)
    assert stmt.value.op == "*"


def test_parser_rejects_nested_function():
    """Test that nested function definitions are rejected."""
    source = """fn outer(input: [], output: []):
    fn inner(input: [], output: []):
        x = 1"""
    with pytest.raises(SyntaxError):
        parse(source)


def test_parser_action_call_in_function():
    """Test that action calls are allowed inside functions."""
    source = """fn fetch_data(input: [url], output: [data]):
    data = @fetch(url=url)
    return data"""
    program = parse(source)

    fn = program.statements[0]
    assert isinstance(fn.body[0], RappelAssignment)
    assert isinstance(fn.body[0].value, RappelActionCall)


def test_parser_allows_for_loop_in_function():
    """Test that for loops are allowed inside functions."""
    source = """fn process_all(input: [items], output: [results]):
    results = []
    for item in items:
        result = handle(x=item)
    return results"""

    program = parse(source)
    assert len(program.statements) == 1
    fn = program.statements[0]
    assert isinstance(fn, RappelFunctionDef)
    assert fn.name == "process_all"


def test_parser_allows_if_in_function():
    """Test that if statements are allowed inside functions."""
    source = """fn check(input: [x], output: [result]):
    if x > 0:
        result = "positive"
    else:
        result = "non-positive"
    return result"""

    program = parse(source)
    fn = program.statements[0]
    assert isinstance(fn.body[0], RappelIfStatement)
