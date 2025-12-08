"""
Benchmark workflow for stress testing the new durable execution engine.

This workflow demonstrates:
1. Fan-out: Spawns many parallel action calls (asyncio.gather)
2. CPU-intensive: Each action performs non-trivial computation
3. Sequential processing with conditionals
4. Fan-in: Aggregates results
"""

import asyncio
import hashlib
from typing import Any

from rappel import action, workflow, Workflow


@action
async def compute_hash(index: int, iterations: int) -> str:
    """Compute a hash chain for a given index."""
    seed = f"benchmark_seed_{index}".encode()
    result = seed
    for _ in range(iterations):
        result = hashlib.sha256(result).digest()
    return result.hex()


@action
async def analyze_hash(hash_value: str) -> dict[str, Any]:
    """Analyze a hash and return statistics."""
    leading_zeros = 0
    for c in hash_value:
        if c == "0":
            leading_zeros += 1
        else:
            break

    prefix_value = int(hash_value[:8], 16)

    return {
        "hash": hash_value,
        "leading_zeros": leading_zeros,
        "prefix_value": prefix_value,
        "is_special": leading_zeros >= 2,
    }


@action
async def process_special(analysis: dict[str, Any]) -> str:
    """Process a hash that was marked as special."""
    return "SPECIAL:" + analysis["hash"][:16]


@action
async def process_normal(analysis: dict[str, Any]) -> str:
    """Process a hash that was not special."""
    return "NORMAL:" + analysis["hash"][:16]


@action
async def combine_results(results: list[str]) -> dict[str, Any]:
    """Combine all processed results into a final summary."""
    special_count = sum(1 for r in results if r.startswith("SPECIAL:"))
    normal_count = sum(1 for r in results if r.startswith("NORMAL:"))

    combined = "".join(results)
    final_hash = hashlib.sha256(combined.encode()).hexdigest()

    return {
        "total_processed": len(results),
        "special_count": special_count,
        "normal_count": normal_count,
        "final_hash": final_hash,
    }


@workflow
class BenchmarkWorkflow(Workflow):
    """
    Fan-out/fan-in benchmark with conditional processing.
    """

    async def run(
        self,
        count: int = 16,
        iterations: int = 100,
    ) -> dict[str, Any]:
        # Fan-out: compute hashes in parallel
        hashes = await asyncio.gather(*[
            compute_hash(index=i, iterations=iterations)
            for i in range(count)
        ])

        # Process each hash with conditional logic
        processed = []
        for hash_value in hashes:
            analysis = await analyze_hash(hash_value)

            if analysis["is_special"]:
                result = await process_special(analysis)
            else:
                result = await process_normal(analysis)

            processed.append(result)

        # Fan-in: combine all results
        summary = await combine_results(processed)
        return summary
