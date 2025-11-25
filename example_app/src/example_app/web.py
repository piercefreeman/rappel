"""FastAPI surface for the rappel example app."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from example_app.workflows import (
    BranchRequest,
    BranchResult,
    ChainRequest,
    ChainResult,
    ComputationRequest,
    ComputationResult,
    ConditionalBranchWorkflow,
    DurableSleepWorkflow,
    ErrorHandlingWorkflow,
    ErrorRequest,
    ErrorResult,
    LoopProcessingWorkflow,
    LoopRequest,
    LoopResult,
    ParallelMathWorkflow,
    SequentialChainWorkflow,
    SleepRequest,
    SleepResult,
)

app = FastAPI(title="Rappel Example")

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


# =============================================================================
# Parallel Execution (asyncio.gather)
# =============================================================================


@app.post("/api/parallel", response_model=ComputationResult)
async def run_parallel_workflow(payload: ComputationRequest) -> ComputationResult:
    """Run the parallel math workflow demonstrating asyncio.gather."""
    workflow = ParallelMathWorkflow()
    return await workflow.run(number=payload.number)


# =============================================================================
# Sequential Chain
# =============================================================================


@app.post("/api/chain", response_model=ChainResult)
async def run_chain_workflow(payload: ChainRequest) -> ChainResult:
    """Run the sequential chain workflow demonstrating action chaining."""
    workflow = SequentialChainWorkflow()
    return await workflow.run(text=payload.text)


# =============================================================================
# Conditional Branching (if/else)
# =============================================================================


@app.post("/api/branch", response_model=BranchResult)
async def run_branch_workflow(payload: BranchRequest) -> BranchResult:
    """Run the conditional branch workflow demonstrating if/else logic."""
    workflow = ConditionalBranchWorkflow()
    return await workflow.run(value=payload.value)


# =============================================================================
# Loop Processing
# =============================================================================


@app.post("/api/loop", response_model=LoopResult)
async def run_loop_workflow(payload: LoopRequest) -> LoopResult:
    """Run the loop workflow demonstrating iteration."""
    workflow = LoopProcessingWorkflow()
    return await workflow.run(items=payload.items)


# =============================================================================
# Error Handling (try/except)
# =============================================================================


@app.post("/api/error", response_model=ErrorResult)
async def run_error_workflow(payload: ErrorRequest) -> ErrorResult:
    """Run the error handling workflow demonstrating try/except."""
    workflow = ErrorHandlingWorkflow()
    return await workflow.run(should_fail=payload.should_fail)


# =============================================================================
# Durable Sleep
# =============================================================================


@app.post("/api/sleep", response_model=SleepResult)
async def run_sleep_workflow(payload: SleepRequest) -> SleepResult:
    """Run the durable sleep workflow demonstrating asyncio.sleep."""
    workflow = DurableSleepWorkflow()
    return await workflow.run(seconds=payload.seconds)
