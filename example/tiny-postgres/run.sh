set -euox pipefail

# Stop any container using port 5433
docker ps --filter "publish=5433" -q | xargs -r docker stop 2>/dev/null || true

# Clean up this project's containers
docker compose down -v 2>/dev/null || true

# Format
uvx isort .
uvx autoflake --remove-all-unused-imports --recursive --in-place .
uvx black --line-length 5000 .

# Start fresh
docker compose up -d --wait
uv run --no-project --with waymark --with pytest --with pytest-asyncio pytest run.py -v --tb=short -x
RET=$?
docker compose down -v
exit $RET
