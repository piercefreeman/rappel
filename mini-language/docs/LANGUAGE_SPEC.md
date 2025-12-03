# Rappel Language Specification

## Motivation

Going from Python directly to a DAG results in a heavy logical burden on the Python AST parser, plus is hard for us to validate at build time whether our DAG will actually execute. We'd like to catch errors early.

Rappel is an intermediate representation (IR) where by virtue of being built we can assert that our DAG will be able to process it. This also opens up the extension for the `run()` workflow to be written in other languages in the future so long as they can compile down to our IR.

We imagine we won't ever touch this grammar directly (preferring `code -> AST -> IR -> DAG`), but it's useful to sketch out the full range of primitives that we expect to have to support.

## Design Principles

1. **Immutable Variables**: All variables are assigned once and never mutated
2. **Explicit I/O**: Functions declare their inputs and outputs upfront
3. **First-Class Actions**: External calls (`@actions`) are the unit of durable execution
4. **No Closures**: No nested functions or captured state
5. **Serializable State**: All values must be JSON-serializable for distributed execution

---

## Lexical Elements

```
IDENT       = [a-zA-Z_][a-zA-Z0-9_]*
INT         = [0-9]+
FLOAT       = [0-9]+ "." [0-9]+
STRING      = '"' [^"]* '"'
BOOL        = "True" | "False"
COMMENT     = "#" [^\n]*
```

---

## Grammar (EBNF)

### Top-Level

```ebnf
(* Program is a collection of function definitions *)
program         = function_def+ ;

(* Function definition with explicit input/output declarations *)
function_def    = "fn" IDENT "(" io_decl ")" ":" body ;
io_decl         = "input:" "[" ident_list? "]" "," "output:" "[" ident_list? "]" ;
ident_list      = IDENT ("," IDENT)* ;
body            = INDENT statement+ DEDENT ;
```

### Statements

```ebnf
statement       = assignment
                | action_call
                | spread_action
                | for_loop
                | conditional
                | python_block
                | return_stmt
                | expr_stmt ;

(* Assignment - binds a value to a variable *)
assignment      = IDENT "=" expr ;

(* Multi-assignment for tuple unpacking *)
multi_assign    = IDENT ("," IDENT)+ "=" expr ;

(* Expression as statement *)
expr_stmt       = expr ;

(* Return statement *)
return_stmt     = "return" expr? ;
```

### Actions

```ebnf
(* Action call - the fundamental unit of durable execution *)
action_call     = "@" IDENT "(" kwargs? ")" ;
kwargs          = kwarg ("," kwarg)* ;
kwarg           = IDENT "=" expr ;

(* Spread action - parallel execution over a collection *)
spread_action   = "spread" expr ":" IDENT "->" action_call ;
```

### Control Flow

```ebnf
(* For loop - iteration over collection *)
for_loop        = "for" IDENT "in" expr ":" body ;

(* Conditional branching *)
conditional     = if_branch elif_branch* else_branch? ;
if_branch       = "if" expr ":" body ;
elif_branch     = "elif" expr ":" body ;
else_branch     = "else:" body ;
```

### Python Escape Hatch

```ebnf
(* Python block for arbitrary computation *)
python_block    = "python" io_spec "{" CODE "}" ;
io_spec         = "(" io_parts ")" ;
io_parts        = io_part (";" io_part)* ;
io_part         = "reads:" ident_list
                | "writes:" ident_list ;
CODE            = <any valid Python code> ;
```

### Expressions

```ebnf
expr            = literal
                | variable
                | binary_op
                | unary_op
                | list_expr
                | dict_expr
                | index_access
                | dot_access
                | function_call
                | action_call ;

literal         = INT | FLOAT | STRING | BOOL ;
variable        = IDENT ;

(* Binary operations *)
binary_op       = expr operator expr ;
operator        = "+" | "-" | "*" | "/" | "//" | "%"
                | "==" | "!=" | "<" | ">" | "<=" | ">="
                | "and" | "or" ;

(* Unary operations *)
unary_op        = ("not" | "-") expr ;

(* Collection literals *)
list_expr       = "[" (expr ("," expr)*)? "]" ;
dict_expr       = "{" (dict_entry ("," dict_entry)*)? "}" ;
dict_entry      = (STRING | IDENT) ":" expr ;

(* Access operations *)
index_access    = expr "[" expr "]" ;
dot_access      = expr "." IDENT ;

(* Function call (non-action) *)
function_call   = IDENT "(" kwargs? ")" ;
```

---

## Semantic Constraints

### Functions

- Functions must declare all inputs and outputs explicitly
- All function calls must use keyword arguments (no positional args)
- Functions are not first-class values (no passing functions as arguments)
- No recursion allowed (DAG must be acyclic)

### Variables

- Variables are immutable - each name can only be assigned once per scope
- All variables must be defined before use
- Variable names must not shadow input parameters

### Actions

- Actions are marked with `@` prefix to distinguish from regular function calls
- Actions are the unit of durable execution - they run on workers and results are persisted
- Action arguments must be serializable expressions
- Actions may have a target to capture the return value: `result = @action(...)`

### Spread Actions

- `spread collection:item -> @action(...)` executes the action for each item in parallel
- The loop variable (`item`) is scoped to the action call
- Results are collected into a list in original order
- All spread iterations are independent (no dependencies between them)

### Conditionals

- All branches that assign to a variable must assign to the same variable name
- Guards are expressions that evaluate to boolean
- Guards must not contain action calls

### For Loops

- Loop variable is scoped to the loop body
- Loop body can contain any statements including nested loops
- Cannot break or continue (full iteration required for DAG)

### Python Blocks

- Must declare all variables read (`reads:`) and written (`writes:`)
- Cannot contain action calls
- Executed inline (not durably persisted)
- Used for local computation that doesn't need durability

---

## DAG Translation

The IR maps to DAG nodes as follows:

| IR Construct | DAG Representation |
|--------------|-------------------|
| `fn` definition | Subgraph with input/output boundary nodes |
| `x = expr` | Assignment node (inline computation) |
| `@action(...)` | Action node (delegated to worker) |
| `spread c:i -> @a(...)` | Spread node → N action nodes → Aggregator node |
| `if/elif/else` | Condition node → Branch nodes → Merge node |
| `for x in c:` | Iterator → Loop body subgraph → Collector |
| `python {...}` | Inline computation node |
| `return expr` | Output node (function boundary) |

### Edge Types

| Edge Type | Description |
|-----------|-------------|
| `DATA` | Value flows from source to target |
| `CONTROL` | Execution order dependency |
| `SPREAD` | Fan-out from spread source to iterations |
| `AGGREGATE` | Fan-in from iterations to result collector |
| `CONDITION` | Guarded edge (only taken if condition is true) |

---

## Comprehensive Example

```rappel
# Order Processing Workflow
# Demonstrates all language constructs in a realistic scenario

fn process_orders(input: [orders, config], output: [summary]):
    # Step 1: Fetch additional data via action
    inventory = @fetch_inventory(warehouse=config["warehouse"])

    # Step 2: Local computation via Python block
    python(reads: orders, inventory; writes: valid_orders, rejected) {
        valid_orders = []
        rejected = []
        for order in orders:
            if order["sku"] in inventory and inventory[order["sku"]] >= order["qty"]:
                valid_orders.append(order)
            else:
                rejected.append({"order": order, "reason": "out_of_stock"})
    }

    # Step 3: Conditional handling based on validation results
    if len(valid_orders) > 0:
        # Step 4: Parallel processing with spread
        # Each order gets payment processed independently
        payments = spread valid_orders:order -> @process_payment(
            order_id=order["id"],
            amount=order["total"],
            customer=order["customer_id"]
        )

        # Step 5: Parallel shipping quotes for all orders
        shipping = spread valid_orders:order -> @get_shipping_quote(
            destination=order["address"],
            weight=order["weight"]
        )

        # Step 6: Combine results with local computation
        python(reads: payments, shipping, valid_orders; writes: confirmations) {
            confirmations = []
            for i, order in enumerate(valid_orders):
                confirmations.append({
                    "order_id": order["id"],
                    "payment": payments[i],
                    "shipping": shipping[i],
                    "status": "confirmed" if payments[i]["success"] else "failed"
                })
        }
    else:
        confirmations = []

    # Step 7: Send notifications via action
    notification_result = @send_notifications(
        confirmations=confirmations,
        rejected=rejected
    )

    # Step 8: Build final summary
    summary = {
        "processed": len(confirmations),
        "rejected": len(rejected),
        "notification_id": notification_result["id"]
    }

    return summary
```

### DAG Visualization

The above program translates to the following DAG structure:

```
                    ┌─────────────────┐
                    │  INPUT (orders, │
                    │     config)     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ @fetch_inventory│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  python block   │
                    │ (validate)      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   CONDITION     │
                    │ len(valid) > 0  │
                    └───┬─────────┬───┘
                        │         │
              ┌─────────┘         └─────────┐
              │ TRUE                  FALSE │
              ▼                             ▼
     ┌────────────────┐           ┌─────────────────┐
     │  SPREAD NODE   │           │ confirmations=[]│
     │ (payments)     │           └────────┬────────┘
     └───────┬────────┘                    │
             │                             │
    ┌────────┼────────┐                    │
    ▼        ▼        ▼                    │
┌───────┐┌───────┐┌───────┐                │
│@pay(1)││@pay(2)││@pay(3)│                │
└───┬───┘└───┬───┘└───┬───┘                │
    │        │        │                    │
    └────────┼────────┘                    │
             ▼                             │
     ┌───────────────┐                     │
     │  AGGREGATOR   │                     │
     │  (payments)   │                     │
     └───────┬───────┘                     │
             │                             │
             ▼                             │
     ┌────────────────┐                    │
     │  SPREAD NODE   │                    │
     │  (shipping)    │                    │
     └───────┬────────┘                    │
             │                             │
    ┌────────┼────────┐                    │
    ▼        ▼        ▼                    │
┌───────┐┌───────┐┌───────┐                │
│@ship1 ││@ship2 ││@ship3 │                │
└───┬───┘└───┬───┘└───┬───┘                │
    │        │        │                    │
    └────────┼────────┘                    │
             ▼                             │
     ┌───────────────┐                     │
     │  AGGREGATOR   │                     │
     │  (shipping)   │                     │
     └───────┬───────┘                     │
             │                             │
             ▼                             │
     ┌───────────────┐                     │
     │ python block  │                     │
     │ (combine)     │                     │
     └───────┬───────┘                     │
             │                             │
             └──────────┬──────────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │     MERGE       │
               │ (confirmations) │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ @send_notifs    │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  python block   │
               │  (summary)      │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ OUTPUT (summary)│
               └─────────────────┘
```

---

## Execution Model

### Single-Threaded Execution

1. Actions are queued and executed one at a time
2. Results are stored in the database after each action
3. Provides deterministic, debuggable execution

### Multi-Threaded / Distributed Execution

1. Workers poll an action queue (simulates `SELECT ... FOR UPDATE SKIP LOCKED`)
2. Independent actions (e.g., spread iterations) execute in parallel
3. Database serves as the central coordinator
4. Results are persisted for fault tolerance

### Inbox Pattern

For efficient distributed execution, we use an append-only inbox pattern:

1. When Node A completes, it `INSERT`s results into an inbox table for each downstream node
2. When Node B is ready to run, it `SELECT`s all rows where `target_node_id = B`
3. This provides O(1) writes (no locks) and efficient batched reads

---

## Type System

Rappel uses dynamic typing with JSON-serializable values:

| Type | Description | Example |
|------|-------------|---------|
| `int` | Integer numbers | `42` |
| `float` | Floating point | `3.14` |
| `str` | Strings | `"hello"` |
| `bool` | Boolean | `True`, `False` |
| `list` | Ordered collection | `[1, 2, 3]` |
| `dict` | Key-value mapping | `{"a": 1, "b": 2}` |
| `None` | Null value | `None` |

All action inputs and outputs must be one of these types (or nested compositions thereof).

---

## Error Handling

### Compile-Time Errors

- Undefined variable reference
- Duplicate variable assignment (immutability violation)
- Missing required function inputs/outputs
- Action call inside Python block
- Positional arguments in function/action calls

### Runtime Errors

- Action handler not registered
- Action execution failure
- Type mismatch in operations
- Index out of bounds
- Key not found in dict

---

## Future Extensions

Potential additions to the language:

1. **Exception Handling**: `try/except` blocks around action calls
2. **Durable Sleep**: `@sleep(duration)` for time-based delays
3. **Retry Policies**: `@action(...) [retry: 3, backoff: exp(100ms, 2x)]`
4. **Timeouts**: `@action(...) [timeout: 30s]`
5. **Parallel Blocks**: `parallel { @a(), @b(), @c() }` for explicit concurrent execution
6. **Type Annotations**: Optional type hints for better validation
