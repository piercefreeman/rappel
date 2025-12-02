"""Action benchmark workflow for raw action dispatch throughput testing."""

from rappel import Workflow, action, workflow

from benchmark_common import PayloadResponse, summarize_payload


@action(name="benchmark.echo_payload")
async def echo_payload(payload: str) -> PayloadResponse:
    """Echo the provided payload and return checksum metadata."""
    return summarize_payload(payload)


@workflow
class EchoActionWorkflow(Workflow):
    """Simple workflow that echoes a payload - for action dispatch benchmarking."""

    name = "benchmark.echo_action"
    concurrent = True

    async def run(self, payload: str = "") -> PayloadResponse:
        result = await echo_payload(payload=payload)
        return result
