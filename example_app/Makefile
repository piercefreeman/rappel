VENV := $(PWD)/.venv/bin
DB := postgresql://postgres:pass@localhost:5432/postgres?sslmode=disable
CONTAINER := waymark-pg
USER_MODULE := example_app.workflows

.PHONY: install
install:
	cd ../scripts && uv run python build_wheel.py --out-dir ../target/wheels
	test -d .venv || uv venv
	uv pip install -e .
	uv pip install --force-reinstall $$(ls ../../target/wheels/waymark-*.whl | head -1)
	uv pip install pytest pytest-asyncio

.PHONY: run
run:
	docker rm -f $(CONTAINER) 2>/dev/null || true
	docker run -d --name $(CONTAINER) --rm -e POSTGRES_PASSWORD=pass -p 5432:5432 postgres:17-alpine >/dev/null
	until docker exec $(CONTAINER) pg_isready >/dev/null 2>&1; do sleep 0.5; done
	PATH=$(VENV):$(PATH) PYTHONPATH=$(PWD)/src WAYMARK_DATABASE_URL=$(DB) WAYMARK_USER_MODULE=$(USER_MODULE) $(VENV)/start-workers >/dev/null 2>&1 & echo $$! > /tmp/workers.pid
	sleep 3
	PATH=$(VENV):$(PATH) PYTHONPATH=$(PWD)/src WAYMARK_DATABASE_URL=$(DB) WAYMARK_USER_MODULE=$(USER_MODULE) WAYMARK_RUN_REAL_CLUSTER=1 $(VENV)/pytest tests/ -vvv || true
	kill $$(cat /tmp/workers.pid) 2>/dev/null || true
	docker stop $(CONTAINER) >/dev/null 2>&1

.PHONY: clean
clean:
	rm -rf .venv ../../target/wheels
	docker rm -f $(CONTAINER) 2>/dev/null || true
