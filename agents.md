## Coding Conventions

- Never add optional-import fallbacks for core dependencies (e.g., wrapping `pydantic` imports in `try/except`). Import them directly and let the program fail fast if they're missing.
- Always run "make lint" and clear the outstanding linting errors before yielding back. Only on very difficult lints where fixing the lint would corrupt the logic should you yield to me for expert intervention. Never yourself write code that ignores the lints on a per line basis. Linting errors should be respected.
- Any python code that you run should be called with `uv` since this is the environment that will have the python dependencies we need. Also make sure you're in the appropriate directory where our pyproject.toml is defined.

## Unit Tests

- Run python tests with `uv run pytest`
- To run the rust integration tests you'll have to do something like: source .env && cargo test ...
