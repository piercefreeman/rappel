"""
Rappel Language Examples - Demonstration code for all language features.
"""

EXAMPLE_IMMUTABLE_VARS = """
# Immutable variable assignment
x = 42
name = "Alice"
is_active = true

# Variables must be reassigned to update
x = x + 1
name = name + " Smith"
"""

EXAMPLE_LIST_OPERATIONS = """
# List initialization
items = []
items = [1, 2, 3]

# Immutable list update (concatenation)
items = items + [4]
items = items + [5, 6]

# List access
first = items[0]
last = items[5]

# List concatenation
results = []
results = results + [10]
"""

EXAMPLE_DICT_OPERATIONS = """
# Dict initialization
config = {}
config = {"host": "localhost", "port": 8080}

# Immutable dict update
config = config + {"timeout": 30}
config = config + {"retries": 3, "debug": true}

# Dict access
host = config["host"]
port = config["port"]
"""

EXAMPLE_FUNCTION_DEF = """
# Function with explicit input/output
fn calculate_area(input: [width, height], output: [area]):
    area = width * height
    return area

# Function with multiple outputs - returns list for unpacking
fn divide_with_remainder(input: [a, b], output: [quotient, remainder]):
    quotient = a / b
    remainder = a - quotient * b
    return [quotient, remainder]

# Simple transformation function
fn double_value(input: [x], output: [result]):
    result = x * 2
    return result

# Using functions - all calls require kwargs
q, r = divide_with_remainder(a=10, b=3)
doubled = double_value(x=5)
area = calculate_area(width=10, height=20)
"""

EXAMPLE_ACTION_CALL = """
# Actions are external - called with @action_name(kwargs) syntax
# No action definitions in code - they are defined externally

# Call an action to fetch data
response = @fetch_url(url="https://api.example.com/data")

# Call an action to save to database
record_id = @save_to_db(data=response, table="responses")

# Chain action calls
user_data = @fetch_user(id=123)
validated = @validate_user(user=user_data)
result = @update_profile(user_id=123, data=validated)
"""

EXAMPLE_FOR_LOOP = """
# For loop with single function call body
fn double_item(input: [item], output: [doubled]):
    doubled = item * 2
    return [doubled]

fn process_items(input: [], output: [results]):
    items = [1, 2, 3, 4, 5]
    results = []
    for item in items:
        doubled = double_item(item=item)
    return [results]
"""

EXAMPLE_SPREAD_OPERATOR = """
# Spread operator for unpacking in lists
base = [1, 2]
extended = [...base, 3, 4, 5]

# Spread with variables
items = [10, 20, 30]
all_items = [...items, 40, 50]
"""

EXAMPLE_CONDITIONALS = """
# If-else with explicit blocks
if value > 100:
    category = "high"
else:
    category = "low"

# Conditional in function
fn classify(input: [score], output: [grade]):
    if score >= 90:
        grade = "A"
    else:
        if score >= 80:
            grade = "B"
        else:
            if score >= 70:
                grade = "C"
            else:
                grade = "F"
    return grade
"""

EXAMPLE_COMPLEX_WORKFLOW = """
# Complex workflow example combining all features
# Functions defined at top level, actions called with @syntax

# Define validation function
fn validate_item(input: [item], output: [validated]):
    if item > 0:
        validated = item
    else:
        validated = None
    return validated

# Define item processing function
fn process_single_item(input: [item], output: [result, error]):
    validated = validate_item(item=item)
    if validated != None:
        result = validated * 2
        error = None
    else:
        result = None
        error = {"item": item, "reason": "invalid"}
    return [result, error]

# Define batch processing function
fn process_batch(input: [batch], output: [processed, failed]):
    processed = []
    failed = []
    for item in batch:
        result = process_single_item(item=item)
    if result != None:
        processed = processed + [result]
    return [processed, failed]

# Main workflow - initialize state
config = {"api_url": "https://api.example.com", "batch_size": 10}
results = []
errors = []

# Execute workflow - actions called with @syntax
batch = @fetch_batch(url=config["api_url"], offset=0, limit=config["batch_size"])
processed, failed = process_batch(batch=batch)
results = results + processed
errors = errors + failed
"""

EXAMPLE_ACTION_SPREAD_LOOP = """
# Comprehensive example: All code lives in functions
# Functions are isolated - data flows only through explicit inputs/outputs
# main() is the default entrypoint

# Function to process a single order (called per-item in loop)
fn process_order(input: [order], output: [result]):
    if order["total"] > 0:
        payment_result = @process_payment(order_id=order["id"], amount=order["total"])
        if payment_result["status"] == "success":
            update_result = @update_order_status(order_id=order["id"], status="completed")
            result = {"order_id": order["id"], "payment": payment_result, "update": update_result}
        else:
            result = {"order_id": order["id"], "error": payment_result["error"]}
    else:
        result = {"order_id": order["id"], "error": "invalid_total"}
    return [result]

# Main entrypoint - orchestrates the workflow
fn main(input: [], output: [notification]):
    # Step 1: Fetch pending orders
    order_ids = @get_pending_orders(status="pending", limit=100)

    # Step 2: Spread to fetch details in parallel
    order_details = spread order_ids:order_id -> @fetch_order_details(id=order_id)

    # Step 3: Process each order (for loop body has exactly one function call)
    for order in order_details:
        result = process_order(order=order)

    # Step 4: Send summary notification
    notification = @send_summary_notification(order_count=100, channel="slack")
    return [notification]
"""
