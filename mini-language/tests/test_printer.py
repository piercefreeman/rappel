"""Tests for Rappel Pretty Printer."""

from rappel import (
    parse,
    RappelPrettyPrinter,
)


def test_printer_simple_assignment():
    """Test printing a simple assignment."""
    source = "x = 42"
    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert "x = 42" in output


def test_printer_string_assignment():
    """Test printing a string assignment."""
    source = 'name = "Alice"'
    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert 'name = "Alice"' in output


def test_printer_list_literal():
    """Test printing a list literal."""
    source = "items = [1, 2, 3]"
    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert "[1, 2, 3]" in output


def test_printer_dict_literal():
    """Test printing a dict literal."""
    source = '{"key": "value", "num": 42}'
    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert '"key"' in output
    assert '"value"' in output


def test_printer_binary_op():
    """Test printing binary operations."""
    source = "result = a + b * c"
    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert "a + b * c" in output or "a + (b * c)" in output


def test_printer_function_def():
    """Test printing function definition."""
    source = """fn add(input: [a, b], output: [result]):
    result = a + b
    return result"""
    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert "fn add" in output
    assert "input:" in output
    assert "output:" in output
    assert "return" in output


def test_printer_action_call():
    """Test printing action call."""
    source = 'response = @fetch_url(url="https://example.com")'
    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert "@fetch_url" in output
    assert "url=" in output


def test_printer_for_loop():
    """Test printing for loop."""
    source = """for item in items:
    result = process(x=item)"""
    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert "for item in items" in output


def test_printer_if_statement():
    """Test printing if statement."""
    source = """if x > 0:
    result = "positive"
else:
    result = "negative"
"""
    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert "if" in output
    assert "else" in output


def test_printer_spread_operator():
    """Test printing spread operator."""
    source = "combined = [...base, 1, 2]"
    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert "..." in output


def test_printer_spread_action():
    """Test printing spread action."""
    source = "results = spread items:item -> @fetch(id=item)"
    program = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program)

    assert "spread" in output
    assert "@fetch" in output


def test_printer_roundtrip():
    """Test that parsed code can be printed and re-parsed."""
    source = """fn test(input: [x], output: [y]):
    y = x + 1
    return y"""

    program1 = parse(source)
    printer = RappelPrettyPrinter()
    output = printer.print(program1)

    # Should be able to parse the printed output
    program2 = parse(output)

    assert len(program1.statements) == len(program2.statements)
