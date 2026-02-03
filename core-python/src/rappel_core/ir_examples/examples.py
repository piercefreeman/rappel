"""Collection of complex IR example programs."""

from __future__ import annotations

import textwrap
from typing import Callable

from proto import ast_pb2 as ir

from ..ir_parser import IRParser


def _parse_source(source: str) -> ir.Program:
    parser = IRParser()
    return parser.parse_program(textwrap.dedent(source).strip())


CONTROL_FLOW_SOURCE = """
fn main(input: [base], output: [summary]):
    payload = {"items": [1, 2, 3, 4], "limit": base}
    items = payload.items
    first_item = items[0]
    limit = payload.limit
    results = []
    for idx, item in enumerate(items):
        if item % 2 == 0:
            doubled = @double(value=item)
            results = results + [doubled]
            continue
        elif item > limit:
            break
        else:
            results = results + [item]
    count = len(results)
    summary = {"count": count, "first": first_item, "results": results}
    return summary
"""


def build_control_flow_program() -> ir.Program:
    """Program with for/if/break/continue, action calls, and dict/index usage."""
    return _parse_source(CONTROL_FLOW_SOURCE)


PARALLEL_SPREAD_SOURCE = """
fn main(input: [base], output: [final]):
    values = range(1, base + 1)
    doubles = spread values:item -> @double(value=item)
    a, b = parallel:
        @double(value=base)
        @double(value=base + 1)
    pair_sum = a + b
    total = @sum(values=doubles)
    final = pair_sum + total
    return final
"""


def build_parallel_spread_program() -> ir.Program:
    """Program with spread and parallel expressions feeding action calls."""
    return _parse_source(PARALLEL_SPREAD_SOURCE)


TRY_EXCEPT_SOURCE = """
fn risky(input: [numerator, denominator], output: [result]):
    try:
        result = numerator / denominator
    except ZeroDivisionError as err:
        result = 0
    return result

fn main(input: [values], output: [total]):
    total = 0
    for item in values:
        denom = item - 2
        part = risky(numerator=item, denominator=denom)
        total = total + part
    return total
"""


def build_try_except_program() -> ir.Program:
    """Program with try/except and a user-defined function call."""
    return _parse_source(TRY_EXCEPT_SOURCE)


WHILE_LOOP_SOURCE = """
fn main(input: [limit], output: [accum]):
    index = 0
    accum = []
    while index < limit:
        accum = accum + [index]
        if index == 2:
            index = index + 1
            continue
        if index == 4:
            break
        index = index + 1
    return accum
"""


def build_while_loop_program() -> ir.Program:
    """Program with while-loop control flow and incremental updates."""
    return _parse_source(WHILE_LOOP_SOURCE)


EXAMPLES: dict[str, Callable[[], ir.Program]] = {
    "control_flow": build_control_flow_program,
    "parallel_spread": build_parallel_spread_program,
    "try_except": build_try_except_program,
    "while_loop": build_while_loop_program,
}


def list_examples() -> list[str]:
    """Return the available example program names."""
    return sorted(EXAMPLES.keys())


def get_example(name: str) -> ir.Program:
    """Fetch an example IR program by name."""
    builder = EXAMPLES.get(name)
    if builder is None:
        raise KeyError(f"unknown example: {name}")
    return builder()
