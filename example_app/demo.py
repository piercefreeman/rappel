import asyncio
from example_app.workflows import (
    ParallelMathWorkflow,
    SequentialChainWorkflow,
    LoopProcessingWorkflow,
)


async def main():
    result = await ParallelMathWorkflow().run(number=10)
    assert result.factorial == 3628800
    assert result.fibonacci == 55

    result = await SequentialChainWorkflow().run(text="hello")
    assert result.final

    result = await LoopProcessingWorkflow().run(items=["a", "b", "c"])
    assert len(result.processed) == 3

    print("✅ All workflows completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
