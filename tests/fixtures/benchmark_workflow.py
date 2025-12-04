"""
Benchmark workflow for stress testing the Rappel runtime.

This workflow is designed to saturate the host CPU and test throughput:
1. Fan-out: Spawns many parallel action calls
2. CPU-intensive: Each action performs non-trivial computation
3. Fan-in: Aggregates results from all parallel branches
4. Nested parallelism: Multiple layers of fan-out/fan-in

Configurable parameters:
- fan_out_width: Number of parallel branches at each level
- computation_depth: CPU work per action (iterations)
- nesting_levels: Depth of nested fan-out/fan-in patterns
"""

import asyncio
import hashlib
from typing import Any

from rappel import action, workflow
from rappel.workflow import Workflow


# =============================================================================
# CPU-Intensive Actions
# =============================================================================


@action
async def hash_chain(seed: str, iterations: int) -> str:
    """
    Perform a chain of SHA-256 hashes.

    This is CPU-intensive and cannot be optimized away.
    Each iteration depends on the previous, preventing parallelization.
    """
    result = seed.encode()
    for _ in range(iterations):
        result = hashlib.sha256(result).digest()
    return result.hex()


@action
async def prime_sieve(limit: int) -> int:
    """
    Count primes up to limit using Sieve of Eratosthenes.

    CPU-intensive with memory pressure.
    """
    if limit < 2:
        return 0

    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False

    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False

    return sum(sieve)


@action
async def matrix_multiply(size: int, seed: int) -> int:
    """
    Perform matrix multiplication on randomly seeded matrices.

    O(n^3) complexity - very CPU intensive for larger sizes.
    Returns a checksum of the result matrix.
    """
    # Simple PRNG for reproducible "random" matrices
    def lcg(x: int) -> int:
        return (1103515245 * x + 12345) & 0x7FFFFFFF

    # Generate matrices
    state = seed
    a: list[list[int]] = []
    b: list[list[int]] = []

    for i in range(size):
        row_a: list[int] = []
        row_b: list[int] = []
        for j in range(size):
            state = lcg(state)
            row_a.append(state % 100)
            state = lcg(state)
            row_b.append(state % 100)
        a.append(row_a)
        b.append(row_b)

    # Multiply
    c: list[list[int]] = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                c[i][j] += a[i][k] * b[k][j]

    # Checksum
    checksum = 0
    for row in c:
        for val in row:
            checksum ^= val
    return checksum


@action
async def fibonacci_memo(n: int) -> int:
    """
    Compute nth Fibonacci number with memoization.

    Tests memory allocation patterns with large dict.
    """
    memo: dict[int, int] = {0: 0, 1: 1}

    def fib(x: int) -> int:
        if x in memo:
            return memo[x]
        memo[x] = fib(x - 1) + fib(x - 2)
        return memo[x]

    return fib(n)


@action
async def string_processing(text: str, rounds: int) -> str:
    """
    Perform multiple rounds of string transformations.

    Tests string allocation and manipulation performance.
    """
    result = text
    for _ in range(rounds):
        # Reverse
        result = result[::-1]
        # Uppercase/lowercase alternating
        chars = []
        for i, c in enumerate(result):
            chars.append(c.upper() if i % 2 == 0 else c.lower())
        result = "".join(chars)
        # Hash and append
        h = hashlib.md5(result.encode()).hexdigest()[:8]
        result = result + h
    return result[-64:]  # Return last 64 chars to bound output size


# =============================================================================
# Aggregation Actions
# =============================================================================


@action
async def aggregate_hashes(hashes: list[str]) -> str:
    """Combine multiple hashes into a single result."""
    combined = "".join(hashes)
    return hashlib.sha256(combined.encode()).hexdigest()


@action
async def aggregate_counts(counts: list[int]) -> dict[str, Any]:
    """Aggregate numeric results with statistics."""
    if not counts:
        return {"sum": 0, "count": 0, "min": 0, "max": 0, "avg": 0.0}

    return {
        "sum": sum(counts),
        "count": len(counts),
        "min": min(counts),
        "max": max(counts),
        "avg": sum(counts) / len(counts),
    }


@action
async def final_summary(
    hash_result: str,
    prime_stats: dict[str, Any],
    matrix_checksum: int,
    fib_value: int,
    string_result: str,
) -> dict[str, Any]:
    """Create final benchmark summary."""
    return {
        "hash_aggregate": hash_result,
        "prime_statistics": prime_stats,
        "matrix_checksum": matrix_checksum,
        "fibonacci_result": fib_value,
        "string_sample": string_result,
        "benchmark_complete": True,
    }


# =============================================================================
# Benchmark Workflows
# =============================================================================


@workflow
class BenchmarkFanOutWorkflow(Workflow):
    """
    Simple fan-out/fan-in benchmark.

    Spawns parallel hash chains based on provided seeds, then aggregates results.
    Uses list comprehension pattern for parallel execution.

    Args:
        seeds: List of seed strings for hash chains (e.g., ["bench_0", "bench_1", ...])
        hash_iterations: Number of hash iterations per chain
    """

    async def run(
        self,
        seeds: list,
        hash_iterations: int = 1000,
    ) -> str:
        # Fan-out: spawn parallel hash computations using list comprehension
        results = await asyncio.gather(*[
            hash_chain(seed=seed, iterations=hash_iterations)
            for seed in seeds
        ])

        # Fan-in: aggregate results
        final = await aggregate_hashes(list(results))
        return final
