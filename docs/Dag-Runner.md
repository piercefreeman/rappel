## Runnable actions (what the runner actually works on)

- Runnable actions are the only things the worker pool ever sees. They fully scope module, action name, kwargs, retry/timeout, and the DAG node id so the completion can be correlated.
- Delegated nodes (`action_call`) become runnable actions dispatched to Python workers. Inline nodes (assign/return/conditional) execute in the runner and never leave the process.
- Queue state lives in Postgres (`action_queue`). The runner repeatedly dequeues ready items, dispatches delegated work, and processes inline/barrier/for-loop/sleep nodes directly.

## Scope + inbox seeding

- Workflow inputs are written once into both the inline scope **and** downstream inboxes off the input node. This happens via a shared `seed_scope_and_inbox` helper so the same data is available to actions even if no intermediate node re-emits it.
- During completion, the runner builds an `InlineContext` (initial scope from registration + DB-fetched inbox + optional spread index) and passes that to `execute_inline_subgraph`. This keeps inline execution deterministic and avoids re-plumbing multiple maps at every call site.
- Loops persist their accumulators and loop index back into the loop inbox each iteration so break edges and subsequent bodies see up-to-date values.

## Completion path (unified readiness model)

1. Analyze the reachable subgraph from the completed node to find inline nodes vs. frontiers (actions, barriers, for-loops, outputs).
2. Batch-read inboxes for all nodes in that subgraph.
3. Execute inline nodes in memory using the `InlineContext`, collecting inbox writes and readiness increments.
4. Commit the completion plan in one transaction (inbox writes, readiness updates, enqueue new runnable actions, optional instance completion).

Exceptions and retries still flow through the existing handler path; the unified path handles successful completions and durable sleeps/barriers/loops.
