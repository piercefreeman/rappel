#!/bin/bash
set -euox pipefail

CONTAINER=pg
docker rm -f $CONTAINER 2>/dev/null || true
docker run -d --name $CONTAINER --rm -e POSTGRES_PASSWORD=pass postgres:17-alpine >/dev/null
until docker exec $CONTAINER pg_isready >/dev/null 2>&1; do :; done; sleep 1

# run stuff
docker exec $CONTAINER psql -U postgres -c "CREATE TABLE demo (word TEXT);"
docker exec $CONTAINER psql -U postgres -c "INSERT INTO demo VALUES ('hello'), ('world'); SELECT * FROM demo;"
docker exec $CONTAINER psql -U postgres -c "SELECT * FROM demo;"

docker stop $CONTAINER >/dev/null
