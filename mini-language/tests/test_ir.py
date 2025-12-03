"""Tests for Rappel IR nodes."""

from rappel import (
    RappelString,
    RappelNumber,
    RappelBoolean,
    RappelLiteral,
    RappelVariable,
    RappelListExpr,
    RappelDictExpr,
    RappelBinaryOp,
    RappelIndexAccess,
    RappelSpread,
    RappelAssignment,
    RappelMultiAssignment,
    RappelReturn,
    RappelFunctionDef,
    RappelActionCall,
    RappelCall,
    RappelForLoop,
    RappelIfStatement,
)


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
    items = (
        RappelLiteral(RappelNumber(1)),
        RappelLiteral(RappelNumber(2)),
        RappelLiteral(RappelNumber(3)),
    )
    lst = RappelListExpr(items=items)
    assert len(lst.items) == 3

    pairs = (
        (RappelLiteral(RappelString("a")), RappelLiteral(RappelNumber(1))),
        (RappelLiteral(RappelString("b")), RappelLiteral(RappelNumber(2))),
    )
    dct = RappelDictExpr(pairs=pairs)
    assert len(dct.pairs) == 2


def test_ir_expressions():
    """Test creating various expressions."""
    var = RappelVariable(name="x")
    assert var.name == "x"

    add = RappelBinaryOp(
        op="+",
        left=RappelVariable(name="a"),
        right=RappelVariable(name="b"),
    )
    assert add.op == "+"

    idx = RappelIndexAccess(
        target=RappelVariable(name="my_list"),
        index=RappelLiteral(RappelNumber(0)),
    )
    assert isinstance(idx.target, RappelVariable)

    spread = RappelSpread(target=RappelVariable(name="items"))
    assert isinstance(spread.target, RappelVariable)


def test_ir_statements():
    """Test creating statements."""
    assign = RappelAssignment(
        target="x",
        value=RappelLiteral(RappelNumber(42)),
    )
    assert assign.target == "x"

    multi = RappelMultiAssignment(
        targets=("a", "b", "c"),
        value=RappelVariable(name="result"),
    )
    assert len(multi.targets) == 3


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
        loop_vars=("item",),
        iterable=RappelVariable(name="items"),
        body=(body_stmt,),
    )
    assert for_loop.loop_vars == ("item",)
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

    try:
        assign.target = "y"  # type: ignore
        assert False, "Should have raised an error"
    except AttributeError:
        pass  # Expected - frozen dataclass
