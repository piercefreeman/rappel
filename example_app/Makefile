
# Build Rust binaries + install Python packages
.PHONY: install
install:
	cd ../scripts && uv run python build_wheel.py --out-dir ../target/wheels
	test -d .venv || uv venv
	uv pip install -e .
	uv pip install --force-reinstall $$(ls ../../target/wheels/waymark-*.whl | head -1)

VENV := $(PWD)/.venv/bin
DB := postgresql://postgres:pass@localhost:5432/postgres?sslmode=disable
CONTAINER := waymark-pg

# Start postgres, run workers + demo, cleanup
.PHONY: run
run:
	docker rm -f $(CONTAINER) 2>/dev/null || true

	docker run -d --name $(CONTAINER) --rm -e POSTGRES_PASSWORD=pass -p 5432:5432 postgres:17-alpine >/dev/null
	until docker exec $(CONTAINER) pg_isready >/dev/null 2>&1; do sleep 0.5; done

	PATH=$(VENV):$(PATH) PYTHONPATH=$(PWD) WAYMARK_DATABASE_URL=$(DB) WAYMARK_USER_MODULE=example_app.workflows $(VENV)/start-workers >/dev/null 2>&1 & echo $$! > /tmp/workers.pid
	sleep 3

	PYTHONPATH=$(PWD) PATH=$(VENV):$(PATH) WAYMARK_DATABASE_URL=$(DB) WAYMARK_USER_MODULE=example_app.workflows $(VENV)/python demo.py

	kill $$(cat /tmp/workers.pid) 2>/dev/null || true
	docker stop $(CONTAINER) >/dev/null

# Remove venv, wheels, postgres container
.PHONY: clean
clean:
	rm -rf .venv ../../target/wheels
	docker rm -f $(CONTAINER) 2>/dev/null || true
