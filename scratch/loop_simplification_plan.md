# Loop Simplification Plan

## Variable Initialization & Collection Tracking

### The Problem

When we have code like:
```python
results = []
for order in orders:
    processed = await action(order=order)
    results.append(processed)
```

We need to:
1. **Initialize `results = []`** before the loop starts
2. **Pass `results` to python_block** so it can mutate it
3. **Receive mutated `results` back** from python_block
4. **Pass final `results` to downstream** actions/nodes

### Current Handling

Currently, empty list initializations like `results = []` are detected by `_is_empty_list_init()` in the Python parser and:
- The variable is added to `_known_vars`
- But NO statement is emitted (returns `None`)

This is problematic because `results` needs to exist in `eval_context` for the loop body to mutate it.

### Solution: Lists/Dicts as IR Primitives

Lists and dicts should be first-class primitives in the IR grammar, not hacked through python_block.

**Grammar addition:**
```ebnf
statement       = ...
                | var_init ;

var_init        = IDENT "=" literal ;
literal         = list_literal | dict_literal | scalar ;
list_literal    = "[" (literal ("," literal)*)? "]" ;
dict_literal    = "{" (key_value ("," key_value)*)? "}" ;
key_value       = STRING ":" literal ;
scalar          = INT | FLOAT | STRING | "true" | "false" | "None" ;
```

**IR Proto (proto/ir.proto):**
```protobuf
// Add to Statement oneof:
message Statement {
  oneof kind {
    // ... existing ...
    VarInit var_init = 10;
  }
}

// New messages:
message VarInit {
  string var = 1;
  Value value = 2;
  optional SourceLocation location = 3;
}

message Value {
  oneof kind {
    string string_val = 1;
    int64 int_val = 2;
    double float_val = 3;
    bool bool_val = 4;
    bool null_val = 5;      // true = null
    ListValue list_val = 6;
    DictValue dict_val = 7;
  }
}

message ListValue {
  repeated Value values = 1;
}

message DictValue {
  repeated DictEntry entries = 1;
}

message DictEntry {
  string key = 1;
  Value value = 2;
}
```

**Example transformations:**
```python
# Python source          # IR emitted
results = []            → VarInit { var: "results", value: ListValue { values: [] } }
counts = {}             → VarInit { var: "counts", value: DictValue { entries: [] } }
items = [1, 2, 3]       → VarInit { var: "items", value: ListValue { values: [1, 2, 3] } }
config = {"a": 1}       → VarInit { var: "config", value: DictValue { entries: [{"a": 1}] } }
```

**DAG handling:**
- `VarInit` becomes a node of type `Computed` (or new type `VarInit`)
- Execution: directly write the value to `eval_context` - no worker dispatch needed
- This is a pure server-side operation, O(1)

**Why not python_block?**
- Cleaner IR representation
- No round-trip to worker for simple initialization
- Explicit typing in the IR (we know it's a list/dict)
- Enables future optimizations (e.g., pre-allocate known-size lists)

### eval_context as the Variable Store

The `eval_context` (stored in `instance_eval_context` table) is a JSON object that holds all workflow variables:

```json
{
  "orders": [{"id": 1}, {"id": 2}],
  "results": [],
  "config": {"timeout": 30},
  "order": {"id": 1},
  "processed": {"status": "done"}
}
```

**Key behaviors:**
1. **Actions write results**: When `processed = await action(...)` completes, `processed` is written to `eval_context`
2. **python_block reads/writes**: Block receives subset of context (inputs), returns outputs to merge back
3. **Loop var binding**: Loop head sets `order = orders[idx]` in context
4. **Mutations persist**: `results.append(x)` in python_block mutates the list in place

### Passing Context to Python Workers

When dispatching a python_block to a worker:
```protobuf
message DispatchPayload {
  map<string, bytes> context = 1;  // Serialized values for inputs
  string code = 2;                  // The python code to execute
  repeated string outputs = 3;      // Which variables to return
}
```

Worker executes code and returns:
```protobuf
message CompletionPayload {
  map<string, bytes> results = 1;  // Serialized output values
}
```

These results are merged back into `eval_context`.

### List Mutation Specifics

For `results.append(processed)`:
1. python_block declares `reads: [results, processed]` and `writes: [results]`
2. Worker receives `results = [...]` and `processed = {...}`
3. Worker executes `results.append(processed)`
4. Worker returns `results = [... + processed]`
5. Server merges `results` back to `eval_context`

The list is passed by value (serialized), mutated, and returned. Not true in-place mutation, but semantically equivalent.

### Changes Required for Initialization

**Python parser (ir.py):**
```diff
- def _is_empty_list_init(self, stmt: ast.Assign) -> bool:
-     # Returns True to skip emitting statement
-     ...

+ def _parse_list_init(self, stmt: ast.Assign) -> ir_pb2.Statement:
+     # Emit a python_block that initializes the list
+     var_name = stmt.targets[0].id
+     return ir_pb2.Statement(
+         python_block=ir_pb2.PythonBlock(
+             code=f"{var_name} = []",
+             inputs=[],
+             outputs=[var_name],
+         )
+     )
```

**This ensures:**
- `results = []` becomes a real node in the DAG
- `eval_context` contains `results: []` before loop starts
- Loop body can read and mutate it

---

## Current State (Overly Complex)

### IR Grammar (proto/ir.proto)
```protobuf
message Loop {
  string iterator_expr = 1;
  string loop_var = 2;
  string accumulator = 3;          // Single accumulator variable
  repeated Statement body = 4;      // Body statements
  optional SourceLocation location = 5;
}
```

### DAG Structure (messages.proto)
```protobuf
message LoopHeadMeta {
  string iterator_source = 1;       // Node that produces the iterable
  string loop_var = 2;
  repeated string body_entry = 3;   // First node(s) of body
  string body_tail = 4;             // Last node of body
  string exit_target = 5;           // Node after loop
  repeated AccumulatorSpec accumulators = 6;  // COMPLEX!
  repeated PreambleOp preamble = 7;           // Magic ops
  repeated string body_nodes = 8;
}

message AccumulatorSpec {
  string var = 1;
  optional string source_node = 2;  // Where to get value
  optional string source_expr = 3;  // Expression to evaluate
}

message PreambleOp {
  oneof op {
    SetIteratorValue set_iterator_value = 1;
    SetAccumulatorLen set_accumulator_len = 2;  // UNNECESSARY
  }
}
```

### Execution (db.rs)
- `loop_iteration_state` table tracks:
  - `current_index`: Which iteration we're on
  - `accumulators`: JSON blob of `{ var_name: [values...] }`
- Complex logic to:
  - Detect "python_block accumulator" vs "spread accumulator"
  - Copy accumulators between `eval_context` and `loop_iteration_state`
  - Evaluate `source_expr` to extract values
  - Push accumulator values to body nodes via `node_pending_context`

### Problems
1. Accumulators tracked separately from regular variables
2. Special logic for different accumulator modes
3. `loop_iteration_state.accumulators` duplicates data in `eval_context`
4. Preamble ops are magic that could just be python_blocks
5. Complex data flow between iterations

---

## Target State (Simplified)

### Core Insight
**Everything is just variables in `eval_context`.** The loop is a sub-graph with a back edge. No special accumulator tracking.

### IR Grammar (proto/ir.proto)
```protobuf
message Loop {
  string iterator_expr = 1;         // Expression producing iterable (e.g., "orders")
  string loop_var = 2;              // Loop variable name (e.g., "order")
  repeated Statement body = 3;      // Sub-graph of statements
  optional SourceLocation location = 4;
}
// REMOVED: accumulator field - not needed!
```

### DAG Structure (messages.proto)
```protobuf
message LoopHeadMeta {
  string iterator_source = 1;       // Node/var that produces the iterable
  string loop_var = 2;              // Variable to bind current item to
  repeated string body_entry = 3;   // First node(s) of body
  string body_tail = 4;             // Last node of body (for back edge)
  repeated string body_nodes = 5;   // All nodes in body (for loop_id tagging)
}
// REMOVED: accumulators, preamble, exit_target
// exit_target is implicit - nodes that depend on loop_head but aren't in body
```

### Database (loop_iteration_state)
```sql
CREATE TABLE loop_iteration_state (
    instance_id UUID,
    node_id TEXT,           -- loop_head node ID
    current_index INT,      -- Which iteration (0, 1, 2, ...)
    PRIMARY KEY (instance_id, node_id)
);
-- REMOVED: accumulators column - not needed!
```

### Execution Flow

**Loop Head becomes ready:**
1. Read `iterator_source` variable from `eval_context` → get the list
2. Read `current_index` from `loop_iteration_state` (default 0)
3. If `current_index >= len(list)`:
   - Loop is done → unlock Exit edges (downstream nodes)
4. Else:
   - Set `loop_var = list[current_index]` in `eval_context`
   - Unlock Continue edges (body entry nodes)

**Body node completes (with Back edge to loop_head):**
1. Body has already written its outputs to `eval_context`
2. Increment `current_index` in `loop_iteration_state`
3. Mark `loop_head` as ready again

**Key insight:** Body nodes read/write `eval_context` directly. Mutations to `results.append(x)` happen in a python_block that writes to `eval_context`. No special accumulator handling needed.

### Data Flow Example

```python
results = []
for order in orders:
    processed = await process(order=order)
    results.append(processed)
summary = await summarize(results=results)
```

**eval_context evolution:**
```
Initial:    { orders: [A, B, C], results: [] }

Iteration 0:
  loop_head:     sets order=A
  process:       sets processed=P_A
  python_block:  mutates results=[P_A]

Iteration 1:
  loop_head:     sets order=B
  process:       sets processed=P_B
  python_block:  mutates results=[P_A, P_B]

Iteration 2:
  loop_head:     sets order=C
  process:       sets processed=P_C
  python_block:  mutates results=[P_A, P_B, P_C]

Exit:
  summarize:     reads results=[P_A, P_B, P_C]
```

**No separate accumulator tracking!** Just `eval_context`.

---

## Changes Required

### 1. Proto Changes

**proto/ir.proto:**
```diff
 message Loop {
   string iterator_expr = 1;
   string loop_var = 2;
-  string accumulator = 3;
-  repeated Statement body = 4;
-  optional SourceLocation location = 5;
+  repeated Statement body = 3;
+  optional SourceLocation location = 4;
 }
```

**proto/messages.proto:**
```diff
 message LoopHeadMeta {
   string iterator_source = 1;
   string loop_var = 2;
   repeated string body_entry = 3;
   string body_tail = 4;
-  string exit_target = 5;
-  repeated AccumulatorSpec accumulators = 6;
-  repeated PreambleOp preamble = 7;
-  repeated string body_nodes = 8;
+  repeated string body_nodes = 5;
 }

-message AccumulatorSpec {
-  string var = 1;
-  optional string source_node = 2;
-  optional string source_expr = 3;
-}
-
-message PreambleOp {
-  oneof op {
-    SetIteratorValue set_iterator_value = 1;
-    SetAccumulatorLen set_accumulator_len = 2;
-  }
-}
-
-message SetIteratorValue {
-  string var = 1;
-}
-
-message SetAccumulatorLen {
-  string accumulator_var = 1;
-  string target_var = 2;
-}
```

### 2. Python Parser Changes (python/src/rappel/ir.py)

**_parse_for_loop():**
- Remove accumulator detection logic
- Remove yield statement handling
- Just parse body as `statement+`
- Emit `Loop { iterator_expr, loop_var, body: [Statement...] }`

**_serialize_loop():**
- Remove accumulator serialization
- Just serialize body statements

### 3. Rust DAG Converter Changes (src/ir_to_dag.rs)

**convert_loop():**
- Create `loop_head` node with simplified `LoopHeadMeta`
- Convert body statements to nodes (recursively)
- Add Continue edge: `loop_head` → first body node
- Add Back edge: last body node → `loop_head`
- Exit edges are implicit: nodes depending on `loop_head` that aren't in body

### 4. Database Changes (src/db.rs)

**Schema:**
- Remove `accumulators` column from `loop_iteration_state` table
- Or just ignore it (for backwards compat during migration)

**process_loop_head_ready():**
```rust
// When loop_head becomes ready:
let iterator_val = eval_context.get(iterator_source);
let current_index = get_loop_index(instance_id, loop_head_id);

if current_index >= iterator_val.len() {
    // Exit: unlock downstream nodes (those with Exit edge from loop_head)
    unlock_exit_edges(loop_head_id);
} else {
    // Continue: bind loop_var and unlock body
    eval_context.set(loop_var, iterator_val[current_index]);
    unlock_continue_edges(loop_head_id);
}
```

**process_body_completion():**
```rust
// When a body node with Back edge completes:
// (Body has already written to eval_context via python_block)
increment_loop_index(instance_id, loop_head_id);
mark_ready(loop_head_id);  // Re-run loop_head to check next iteration
```

**REMOVED:**
- All accumulator sync logic
- `uses_python_block_accumulator` detection
- `AccumulatorSpec` handling
- `source_expr` evaluation
- Copying between `loop_iteration_state.accumulators` and `eval_context`

### 5. Spread Handling

Spread can stay as-is (parallel expansion) OR be simplified to also just use `eval_context`:

**Option A: Keep Spread separate**
- Spread expands to N parallel action nodes at IR→DAG time
- Each produces a result
- A sync node collects results into a list variable

**Option B: Spread as syntactic sugar**
- Spread becomes a Loop + implicit `results.append()`
- Loses parallelism but simpler

Recommend **Option A** for now - Spread is genuinely different (parallel vs sequential).

---

## Migration Strategy

1. **Update protos** - add new fields, deprecate old ones
2. **Update Python parser** - emit simplified Loop IR
3. **Update Rust converter** - handle both old and new Loop formats
4. **Update db.rs** - simplified loop execution
5. **Update tests** - remove accumulator-specific test assertions
6. **Clean up** - remove deprecated proto fields and code paths

---

## Test Cases to Verify

1. **Simple loop with append:**
   ```python
   results = []
   for x in items:
       y = await action(x=x)
       results.append(y)
   ```

2. **Loop with multiple mutations:**
   ```python
   results = []
   total = 0
   for x in items:
       y = await action(x=x)
       results.append(y)
       total += y.amount
   ```

3. **Loop with conditional inside:**
   ```python
   results = []
   for x in items:
       if x.valid:
           y = await process(x=x)
       else:
           y = await fallback(x=x)
       results.append(y)
   ```

4. **Loop with multiple actions:**
   ```python
   for order in orders:
       validated = await validate(order=order)
       processed = await process(validated=validated)
       await notify(order_id=processed.id)
   ```

5. **Nested structures (loop after gather):**
   ```python
   (a, b) = await asyncio.gather(fetch_a(), fetch_b())
   results = []
   for x in a:
       y = await process(x=x, config=b)
       results.append(y)
   ```
