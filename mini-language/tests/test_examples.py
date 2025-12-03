"""Tests for Rappel example programs."""

from rappel import (
    parse,
    convert_to_dag,
    RappelPrettyPrinter,
    EXAMPLE_IMMUTABLE_VARS,
    EXAMPLE_LIST_OPERATIONS,
    EXAMPLE_DICT_OPERATIONS,
    EXAMPLE_FUNCTION_DEF,
    EXAMPLE_ACTION_CALL,
    EXAMPLE_FOR_LOOP,
    EXAMPLE_SPREAD_OPERATOR,
    EXAMPLE_CONDITIONALS,
    EXAMPLE_COMPLEX_WORKFLOW,
    EXAMPLE_ACTION_SPREAD_LOOP,
)


def test_example_immutable_vars():
    """Test parsing immutable vars example."""
    program = parse(EXAMPLE_IMMUTABLE_VARS)
    assert len(program.statements) > 0


def test_example_list_operations():
    """Test parsing list operations example."""
    program = parse(EXAMPLE_LIST_OPERATIONS)
    assert len(program.statements) > 0


def test_example_dict_operations():
    """Test parsing dict operations example."""
    program = parse(EXAMPLE_DICT_OPERATIONS)
    assert len(program.statements) > 0


def test_example_function_def():
    """Test parsing function def example."""
    program = parse(EXAMPLE_FUNCTION_DEF)
    assert len(program.statements) > 0


def test_example_action_call():
    """Test parsing action call example."""
    program = parse(EXAMPLE_ACTION_CALL)
    assert len(program.statements) > 0


def test_example_for_loop():
    """Test parsing for loop example."""
    program = parse(EXAMPLE_FOR_LOOP)
    assert len(program.statements) > 0


def test_example_spread_operator():
    """Test parsing spread operator example."""
    program = parse(EXAMPLE_SPREAD_OPERATOR)
    assert len(program.statements) > 0


def test_example_conditionals():
    """Test parsing conditionals example."""
    program = parse(EXAMPLE_CONDITIONALS)
    assert len(program.statements) > 0


def test_example_complex_workflow():
    """Test parsing complex workflow example."""
    program = parse(EXAMPLE_COMPLEX_WORKFLOW)
    assert len(program.statements) > 0


def test_example_action_spread_loop():
    """Test parsing action spread loop example."""
    program = parse(EXAMPLE_ACTION_SPREAD_LOOP)
    assert len(program.statements) > 0


def test_all_examples_convert_to_dag():
    """Test that all examples can be converted to DAG without error."""
    examples = [
        EXAMPLE_IMMUTABLE_VARS,
        EXAMPLE_LIST_OPERATIONS,
        EXAMPLE_DICT_OPERATIONS,
        EXAMPLE_FUNCTION_DEF,
        EXAMPLE_ACTION_CALL,
        EXAMPLE_FOR_LOOP,
        EXAMPLE_SPREAD_OPERATOR,
        EXAMPLE_CONDITIONALS,
        EXAMPLE_COMPLEX_WORKFLOW,
        EXAMPLE_ACTION_SPREAD_LOOP,
    ]

    for example in examples:
        program = parse(example)
        # Should not raise any exception
        dag = convert_to_dag(program)
        # DAG is valid (may have 0 nodes if no functions)
        assert dag is not None


def test_examples_with_functions_have_dag_nodes():
    """Test that examples with functions produce non-empty DAGs."""
    # These examples contain function definitions
    examples_with_functions = [
        EXAMPLE_FUNCTION_DEF,
        EXAMPLE_FOR_LOOP,
        EXAMPLE_CONDITIONALS,
        EXAMPLE_COMPLEX_WORKFLOW,
        EXAMPLE_ACTION_SPREAD_LOOP,
    ]

    for example in examples_with_functions:
        program = parse(example)
        dag = convert_to_dag(program)
        assert len(dag.nodes) > 0, f"Expected nodes for example with functions"


def test_all_examples_pretty_print():
    """Test that all examples can be pretty printed."""
    examples = [
        EXAMPLE_IMMUTABLE_VARS,
        EXAMPLE_LIST_OPERATIONS,
        EXAMPLE_DICT_OPERATIONS,
        EXAMPLE_FUNCTION_DEF,
        EXAMPLE_ACTION_CALL,
        EXAMPLE_FOR_LOOP,
        EXAMPLE_SPREAD_OPERATOR,
        EXAMPLE_CONDITIONALS,
        EXAMPLE_COMPLEX_WORKFLOW,
        EXAMPLE_ACTION_SPREAD_LOOP,
    ]

    printer = RappelPrettyPrinter()

    for example in examples:
        program = parse(example)
        output = printer.print(program)
        assert len(output) > 0
