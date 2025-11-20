"""FastAPI surface for the rappel example app."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from example_app.workflows import ComputationRequest, ComputationResult, ExampleMathWorkflow

app = FastAPI(title="Rappel Example")

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_number": 5,
        },
    )


@app.post("/api/tasks", response_model=ComputationResult)
async def run_task(payload: ComputationRequest) -> ComputationResult:
    workflow = ExampleMathWorkflow(number=payload.number)
    result = await workflow.run()
    return result
