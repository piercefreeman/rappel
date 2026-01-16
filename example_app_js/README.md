## Rappel JS example app

`example_app_js` mirrors the Python example app, but defines workflows + actions in TypeScript
and serves the UI with an Express server. Worker execution is managed by the Rust pool using
the Node runtime.

### Prerequisites

- A running Rappel bridge + Postgres (use the root docker-compose or your local stack).
- `RAPPEL_BRIDGE_GRPC_ADDR` or `RAPPEL_BRIDGE_GRPC_HOST`/`RAPPEL_BRIDGE_GRPC_PORT` set.
- `RAPPEL_DATABASE_URL` set if you want the reset endpoint to work.

### Install + build

```bash
cd js
npm install
npm run build

cd ../example_app_js
npm install
npm run build
```

### Run

Recommended (Rust-managed worker pool with Node runtime):

```bash
cd js
npm run build

cd ../example_app_js
npm run build

RAPPEL_DATABASE_URL=postgresql://rappel:rappel@localhost:5432/rappel_example_js \
RAPPEL_WORKER_RUNTIME=node \
RAPPEL_NODE_WORKER_SCRIPT=$(pwd)/node_modules/@rappel/js/dist/worker-cli.js \
RAPPEL_USER_MODULE=$(pwd)/dist/workflows.js \
cargo run --bin start-workers
```

Start the web server in another terminal:

```bash
cd example_app_js
RAPPEL_BRIDGE_GRPC_ADDR=127.0.0.1:24117 npm run start
```

Then visit http://localhost:8001/.

Notes:
- Actions are defined in `src/workflows.ts` and loaded via `RAPPEL_USER_MODULE`.
- Schedule endpoints use the default schedule name `default`.

### Docker

```bash
cd example_app_js
make up
```

Visit http://localhost:8001/ and tear down with:

```bash
make down
```
