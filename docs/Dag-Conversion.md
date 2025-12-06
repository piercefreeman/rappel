# DAG Conversion

Once we have an IR representation, our job is to make a full DAG. It should accept in a full IR representation and create nodes. This DAG conversion is non trivial, because there are some elements of control flow (like spread-gather operations and for loops) that don't typically fit in standard graphs.

Two edge types:
	— state machine destination
	— data flow

State machine destinations define the possible nodes to travel to after this node is completed, perhaps given some gating criteria to tie-break between them. (src, dst) means that dst should follow src in the execution roder.

Data flow destinatinos determine where to push some variables that were set by this node. (src, dst) means that src is pushing some data in to dst. You’ll want to parameterize these edges by which variable of the output is being pushed.

“for” should be a single node, with output state machine edges for “continue” (first node contained within it) and “done”

Any variables needed by the function within the “for” loop should be passed FROM the for loop. Which means any nodes that need to pass values INTO the for loop should just pass them to the for loop head.

The class you build to do this should have a .visualize() which should popup some matplotlib graph of the dependencies that we have here in some tree like structure. solid lines for the state machine dependencies and dotted lines for the data flow.

"spread" should be implemented as the action of interest, then a node after it acting as an aggregator. reason being is we want to allow each action to finsih in order that it finsihes and just write "action_{i}" to the results payload for the next node without us having to block to modify a list.

NOTE: we should only PUSH data to the most recent trailing node that DOESN’T modify the variable. If a variable name is modified downstream, downstream nodes should rely on the modified node value


## spread-gather

Spread-gather enables parallel execution of the same action across a collection of items. Unlike traditional sequential iteration, all items can be processed concurrently, and results are gathered into a list once all complete.

For more details on the readiness model and frontier nodes, see [specs/unified-readiness-model.md](specs/unified-readiness-model.md).

### Simple Example

Consider this Python code:

```python
@workflow
def process_all(items: list[str]) -> list[str]:
    results = spread items:item -> @process(item)
    return results
```

### DAG Structure

```
┌─────────────────────────────┐
│         input               │
│   (receives: items)         │
└──────────────┬──────────────┘
               │ StateMachine
               ▼
┌─────────────────────────────┐
│     spread_action_1         │
│   is_spread: true           │
│   spread_collection: items  │
│   spread_loop_var: item     │
│   aggregates_to: agg_1      │
└──────────────┬──────────────┘
               │ StateMachine
               ▼
┌─────────────────────────────┐
│       aggregator_1          │
│   is_aggregator: true       │
│   aggregates_from:          │
│     spread_action_1         │
└──────────────┬──────────────┘
               │ StateMachine
               ▼
┌─────────────────────────────┐
│         output              │
│   (returns: results)        │
└─────────────────────────────┘
```

### Step-by-Step Execution (items = ["a", "b", "c"])

**Time T0: Workflow starts**

```
node_readiness:
┌─────────────────┬──────────┬───────────┐
│ node_id         │ required │ completed │
├─────────────────┼──────────┼───────────┤
│ (empty)         │          │           │
└─────────────────┴──────────┴───────────┘

node_inputs (inbox):
┌─────────────────┬───────────┬───────────┬──────────────┐
│ target_node_id  │ variable  │ value     │ spread_index │
├─────────────────┼───────────┼───────────┼──────────────┤
│ input           │ items     │ ["a","b","c"] │ NULL     │
└─────────────────┴───────────┴───────────┴──────────────┘
```

The `input` node executes inline (no worker needed). It pushes `items` to the spread node.

**Time T1: Spread node evaluates**

The spread node is a frontier node (action), so it gets enqueued. When it runs, the runtime:
1. Reads `items` from its inbox
2. Creates N=3 separate action instances: `spread_action_1[0]`, `spread_action_1[1]`, `spread_action_1[2]`
3. Initializes the aggregator's readiness with `required_count = 3`

```
node_readiness:
┌─────────────────┬──────────┬───────────┐
│ node_id         │ required │ completed │
├─────────────────┼──────────┼───────────┤
│ aggregator_1    │ 3        │ 0         │
└─────────────────┴──────────┴───────────┘

action_queue:
┌──────────────────────┬─────────┐
│ node_id              │ status  │
├──────────────────────┼─────────┤
│ spread_action_1[0]   │ queued  │
│ spread_action_1[1]   │ queued  │
│ spread_action_1[2]   │ queued  │
└──────────────────────┴─────────┘
```

**Time T2: First action completes (index 1 finishes first)**

`spread_action_1[1]` returns `"B_processed"`. The completion handler:
1. Writes result to aggregator's inbox with `spread_index = 1`
2. Increments aggregator's `completed_count`

```
node_readiness:
┌─────────────────┬──────────┬───────────┐
│ node_id         │ required │ completed │
├─────────────────┼──────────┼───────────┤
│ aggregator_1    │ 3        │ 1         │
└─────────────────┴──────────┴───────────┘

node_inputs (inbox):
┌─────────────────┬───────────┬────────────────┬──────────────┐
│ target_node_id  │ variable  │ value          │ spread_index │
├─────────────────┼───────────┼────────────────┼──────────────┤
│ aggregator_1    │ result    │ "B_processed"  │ 1            │
└─────────────────┴───────────┴────────────────┴──────────────┘
```

**Time T3: Second action completes (index 0)**

```
node_readiness:
┌─────────────────┬──────────┬───────────┐
│ node_id         │ required │ completed │
├─────────────────┼──────────┼───────────┤
│ aggregator_1    │ 3        │ 2         │
└─────────────────┴──────────┴───────────┘

node_inputs (inbox):
┌─────────────────┬───────────┬────────────────┬──────────────┐
│ target_node_id  │ variable  │ value          │ spread_index │
├─────────────────┼───────────┼────────────────┼──────────────┤
│ aggregator_1    │ result    │ "B_processed"  │ 1            │
│ aggregator_1    │ result    │ "A_processed"  │ 0            │
└─────────────────┴───────────┴────────────────┴──────────────┘
```

**Time T4: Final action completes (index 2) → Aggregator becomes ready**

`completed_count == required_count`, so the aggregator is enqueued as a barrier:

```
node_readiness:
┌─────────────────┬──────────┬───────────┐
│ node_id         │ required │ completed │
├─────────────────┼──────────┼───────────┤
│ aggregator_1    │ 3        │ 3         │  ← READY!
└─────────────────┴──────────┴───────────┘

node_inputs (inbox):
┌─────────────────┬───────────┬────────────────┬──────────────┐
│ target_node_id  │ variable  │ value          │ spread_index │
├─────────────────┼───────────┼────────────────┼──────────────┤
│ aggregator_1    │ result    │ "B_processed"  │ 1            │
│ aggregator_1    │ result    │ "A_processed"  │ 0            │
│ aggregator_1    │ result    │ "C_processed"  │ 2            │
└─────────────────┴───────────┴────────────────┴──────────────┘

action_queue:
┌──────────────────────┬───────────┐
│ node_id              │ status    │
├──────────────────────┼───────────┤
│ aggregator_1         │ queued    │  (barrier type)
└──────────────────────┴───────────┘
```

**Time T5: Aggregator processes**

The polling loop picks up the barrier. It:
1. Reads all inbox entries for `aggregator_1`
2. Sorts by `spread_index` to reconstruct original order
3. Produces `results = ["A_processed", "B_processed", "C_processed"]`
4. Pushes to output node's inbox

```
node_inputs (inbox):
┌─────────────────┬───────────┬──────────────────────────────────────────┬──────────────┐
│ target_node_id  │ variable  │ value                                    │ spread_index │
├─────────────────┼───────────┼──────────────────────────────────────────┼──────────────┤
│ output          │ results   │ ["A_processed","B_processed","C_processed"] │ NULL      │
└─────────────────┴───────────┴──────────────────────────────────────────┴──────────────┘
```

### Key Points

- Results can complete out of order—`spread_index` preserves original ordering
- The aggregator is a "frontier node" that stops inline traversal
- Each spread instance is truly independent—no shared state
- The `required_count` is set at spread expansion time based on collection length

---

## for loops

In a sense I'm playing fast and loose with the acronym "DAG". Our graphs can be cyclic because of for loop support. For loops, in contrast to spread-gather, execute synchronously—each iteration must complete before the next begins, since later iterations may depend on accumulated state.

### Simple Example

Consider this Python code:

```python
@workflow
def sum_all(items: list[int]) -> int:
    total = 0
    for item in items:
        total = @add(total, item)
    return total
```

### DAG Structure

```
┌─────────────────────────────┐
│         input               │
│   (receives: items)         │
└──────────────┬──────────────┘
               │ StateMachine
               ▼
┌─────────────────────────────┐
│     assignment_1            │
│   total = 0                 │
└──────────────┬──────────────┘
               │ StateMachine
               ▼
┌─────────────────────────────┐
│       for_loop_1            │◄─────────────────┐
│   is_loop_head: true        │                  │
│   loop_collection: items    │                  │
│   loop_vars: [item]         │                  │
└──────────────┬──────────────┘                  │
               │ StateMachine                    │
               │ (guard: i < len(items))         │
               ▼                                 │
┌─────────────────────────────┐                  │
│       action_1              │                  │
│   @add(total, item)         │                  │
│   loop_body_of: for_loop_1  │                  │
└──────────────┬──────────────┘                  │
               │ StateMachine                    │
               │ (is_loop_back: true)            │
               └─────────────────────────────────┘

       (from for_loop_1, guard: i >= len(items))
               │
               ▼
┌─────────────────────────────┐
│         output              │
│   (returns: total)          │
└─────────────────────────────┘
```

### Step-by-Step Execution (items = [10, 20, 30])

**Time T0: Workflow starts**

```
node_inputs (inbox):
┌─────────────────┬───────────┬───────────┐
│ target_node_id  │ variable  │ value     │
├─────────────────┼───────────┼───────────┤
│ input           │ items     │ [10,20,30]│
└─────────────────┴───────────┴───────────┘

Scope: { items: [10, 20, 30] }
```

**Time T1: Assignment executes (inline)**

```
node_inputs (inbox):
┌─────────────────┬───────────┬───────────┐
│ target_node_id  │ variable  │ value     │
├─────────────────┼───────────┼───────────┤
│ for_loop_1      │ total     │ 0         │
│ for_loop_1      │ items     │ [10,20,30]│
└─────────────────┴───────────┴───────────┘

Scope: { items: [10, 20, 30], total: 0 }
```

**Time T2: For loop head evaluates (iteration 0)**

The for loop head is a frontier node. When it runs:
1. Reads `items` and `total` from inbox
2. Checks guard: `i=0 < len(items)=3` → true, continue into loop
3. Extracts `item = items[0] = 10`
4. Pushes `item` and `total` to loop body

```
node_inputs (inbox):
┌─────────────────┬───────────┬───────────┐
│ target_node_id  │ variable  │ value     │
├─────────────────┼───────────┼───────────┤
│ action_1        │ total     │ 0         │
│ action_1        │ item      │ 10        │
└─────────────────┴───────────┴───────────┘

for_loop_1 internal state:
  i = 0
  items = [10, 20, 30]
```

**Time T3: Action completes (iteration 0)**

`@add(0, 10)` returns `10`. The loop back edge triggers:
1. Result written back to for loop head's inbox
2. For loop head is re-enqueued

```
node_inputs (inbox):
┌─────────────────┬───────────┬───────────┐
│ target_node_id  │ variable  │ value     │
├─────────────────┼───────────┼───────────┤
│ for_loop_1      │ total     │ 10        │  ← updated!
│ for_loop_1      │ items     │ [10,20,30]│
└─────────────────┴───────────┴───────────┘

for_loop_1 internal state:
  i = 1  (incremented)
```

**Time T4: For loop head evaluates (iteration 1)**

Guard: `i=1 < 3` → true, continue. Extracts `item = items[1] = 20`.

```
node_inputs (inbox):
┌─────────────────┬───────────┬───────────┐
│ target_node_id  │ variable  │ value     │
├─────────────────┼───────────┼───────────┤
│ action_1        │ total     │ 10        │
│ action_1        │ item      │ 20        │
└─────────────────┴───────────┴───────────┘
```

**Time T5: Action completes (iteration 1)**

`@add(10, 20)` returns `30`.

```
node_inputs (inbox):
┌─────────────────┬───────────┬───────────┐
│ target_node_id  │ variable  │ value     │
├─────────────────┼───────────┼───────────┤
│ for_loop_1      │ total     │ 30        │
│ for_loop_1      │ items     │ [10,20,30]│
└─────────────────┴───────────┴───────────┘

for_loop_1 internal state:
  i = 2
```

**Time T6: For loop head evaluates (iteration 2)**

Guard: `i=2 < 3` → true. Extracts `item = items[2] = 30`.

```
node_inputs (inbox):
┌─────────────────┬───────────┬───────────┐
│ target_node_id  │ variable  │ value     │
├─────────────────┼───────────┼───────────┤
│ action_1        │ total     │ 30        │
│ action_1        │ item      │ 30        │
└─────────────────┴───────────┴───────────┘
```

**Time T7: Action completes (iteration 2)**

`@add(30, 30)` returns `60`.

```
node_inputs (inbox):
┌─────────────────┬───────────┬───────────┐
│ target_node_id  │ variable  │ value     │
├─────────────────┼───────────┼───────────┤
│ for_loop_1      │ total     │ 60        │
│ for_loop_1      │ items     │ [10,20,30]│
└─────────────────┴───────────┴───────────┘

for_loop_1 internal state:
  i = 3
```

**Time T8: For loop head evaluates (loop exit)**

Guard: `i=3 < 3` → **false**, exit loop. The "done" edge is followed:

```
node_inputs (inbox):
┌─────────────────┬───────────┬───────────┐
│ target_node_id  │ variable  │ value     │
├─────────────────┼───────────┼───────────┤
│ output          │ total     │ 60        │
└─────────────────┴───────────┴───────────┘
```

**Time T9: Output completes**

Workflow returns `60`.

### Key Points

- The for loop head is a frontier node that gets re-enqueued after each iteration
- Loop state (`i`) is tracked as part of the node's execution context
- The guard expression determines which outgoing edge to follow
- Loop back edges (`is_loop_back: true`) feed results back to the loop head
- Variables modified in the loop body update the inbox for subsequent iterations
- For loops are inherently sequential—no parallelism within a single loop

### Comparison: For Loop vs Spread-Gather

| Aspect | For Loop | Spread-Gather |
|--------|----------|---------------|
| Execution | Sequential | Parallel |
| State | Accumulates across iterations | Each instance independent |
| Use case | When order matters or iterations depend on previous results | When processing is independent |
| DAG shape | Cyclic (loop back edges) | Acyclic (spread → N actions → aggregator) |
| Inbox updates | Loop head inbox updated each iteration | Aggregator inbox grows with each completion |
