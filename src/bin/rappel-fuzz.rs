use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::Parser;
use prost::Message;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile::TempDir;

use rappel::in_memory::InMemoryWorkflowExecutor;
use rappel::messages::{json_to_workflow_argument_value, proto, workflow_argument_value_to_json};
use rappel::workflow_arguments_to_json;

#[derive(Debug, Parser)]
#[command(name = "rappel-fuzz")]
#[command(about = "Fuzz Python vs Rust in-memory execution parity.")]
struct Cli {
    #[arg(long, default_value_t = 25)]
    cases: usize,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 8)]
    max_steps: usize,
    #[arg(long)]
    keep: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArgKind {
    Int,
    ListInt,
}

#[derive(Debug, Clone)]
struct ActionDef {
    name: &'static str,
    args: &'static [(&'static str, ArgKind)],
    output: ArgKind,
}

#[derive(Debug, Clone)]
struct CallSpec {
    action: &'static str,
    args: Vec<(String, String)>,
    output_var: String,
    trace_id: i64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ActionTrace {
    trace_id: String,
    inputs: Value,
    output: Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OracleTrace {
    actions: Vec<ActionTrace>,
    workflow_output: Value,
}

#[derive(Debug, Clone)]
struct FuzzError {
    exc_type: String,
    message: String,
    module: String,
    type_hierarchy: Vec<String>,
}

#[derive(Debug)]
enum ActionOutcome {
    Ok(Value),
    Err(FuzzError),
}

#[derive(Debug)]
struct FuzzCase {
    python_source: String,
    inputs: Value,
}

const ACTIONS: &[ActionDef] = &[
    ActionDef {
        name: "add",
        args: &[("a", ArgKind::Int), ("b", ArgKind::Int)],
        output: ArgKind::Int,
    },
    ActionDef {
        name: "mul",
        args: &[("a", ArgKind::Int), ("b", ArgKind::Int)],
        output: ArgKind::Int,
    },
    ActionDef {
        name: "negate",
        args: &[("value", ArgKind::Int)],
        output: ArgKind::Int,
    },
    ActionDef {
        name: "sum_list",
        args: &[("items", ArgKind::ListInt)],
        output: ArgKind::Int,
    },
    ActionDef {
        name: "len_list",
        args: &[("items", ArgKind::ListInt)],
        output: ArgKind::Int,
    },
    ActionDef {
        name: "make_list",
        args: &[
            ("a", ArgKind::Int),
            ("b", ArgKind::Int),
            ("c", ArgKind::Int),
        ],
        output: ArgKind::ListInt,
    },
    ActionDef {
        name: "append_list",
        args: &[("items", ArgKind::ListInt), ("value", ArgKind::Int)],
        output: ArgKind::ListInt,
    },
    ActionDef {
        name: "scale_list",
        args: &[("items", ArgKind::ListInt), ("factor", ArgKind::Int)],
        output: ArgKind::ListInt,
    },
    ActionDef {
        name: "concat_lists",
        args: &[("left", ArgKind::ListInt), ("right", ArgKind::ListInt)],
        output: ArgKind::ListInt,
    },
    ActionDef {
        name: "take_slice",
        args: &[("items", ArgKind::ListInt), ("end", ArgKind::Int)],
        output: ArgKind::ListInt,
    },
];

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let seed = cli.seed.unwrap_or_else(rand::random);
    let mut rng = StdRng::seed_from_u64(seed);
    println!("fuzz seed: {seed}");

    for case_idx in 0..cli.cases {
        let case = generate_case(&mut rng, cli.max_steps);
        if let Err(err) = run_case(case_idx, &case, cli.keep).await {
            println!("case {case_idx} failed with seed {seed}: {err:#}");
            return Err(err);
        }
    }

    Ok(())
}

fn generate_case(rng: &mut StdRng, max_steps: usize) -> FuzzCase {
    let inputs = generate_inputs(rng);
    let mut int_vars = vec!["seed".to_string(), "factor".to_string()];
    let mut list_vars = vec!["items".to_string()];
    let mut lines: Vec<String> = Vec::new();
    let mut trace_counter: i64 = 1000;
    let mut var_counter = 0usize;

    let steps = rng.gen_range(2..=max_steps.max(4));
    for _ in 0..steps {
        emit_random_action(
            rng,
            &mut lines,
            2,
            &mut int_vars,
            &mut list_vars,
            &mut trace_counter,
            &mut var_counter,
        );
    }

    emit_if_block(
        rng,
        &mut lines,
        &mut int_vars,
        &mut list_vars,
        &mut trace_counter,
        &mut var_counter,
    );

    emit_for_loop(rng, &mut lines, &mut list_vars, &mut trace_counter);

    emit_try_except(
        rng,
        &mut lines,
        &mut int_vars,
        &mut list_vars,
        &mut trace_counter,
        &mut var_counter,
    );

    let tail_steps = rng.gen_range(1..=3);
    for _ in 0..tail_steps {
        emit_random_action(
            rng,
            &mut lines,
            2,
            &mut int_vars,
            &mut list_vars,
            &mut trace_counter,
            &mut var_counter,
        );
    }

    let return_var = choose_return_var(rng, &int_vars, &list_vars);
    push_line(&mut lines, 2, format!("return {return_var}"));

    FuzzCase {
        python_source: render_python_module(&lines),
        inputs,
    }
}

fn generate_inputs(rng: &mut StdRng) -> Value {
    let seed = rng.gen_range(-50..=50);
    let factor = rng.gen_range(1..=6);
    let items_len = rng.gen_range(1..=6);
    let mut items: Vec<i64> = Vec::with_capacity(items_len);
    while items.len() < items_len {
        let candidate = rng.gen_range(-10..=10);
        if !items.contains(&candidate) {
            items.push(candidate);
        }
    }
    serde_json::json!({
        "seed": seed,
        "items": items,
        "factor": factor,
    })
}

fn choose_action(
    rng: &mut StdRng,
    int_vars: &[String],
    list_vars: &[String],
) -> &'static ActionDef {
    let mut eligible: Vec<&ActionDef> = Vec::new();
    for action in ACTIONS {
        if action.args.iter().all(|(_, kind)| match kind {
            ArgKind::Int => !int_vars.is_empty(),
            ArgKind::ListInt => !list_vars.is_empty(),
        }) {
            eligible.push(action);
        }
    }
    let idx = rng.gen_range(0..eligible.len());
    eligible[idx]
}

fn choose_action_with_output(
    rng: &mut StdRng,
    int_vars: &[String],
    list_vars: &[String],
    output: ArgKind,
) -> &'static ActionDef {
    let mut eligible: Vec<&ActionDef> = ACTIONS
        .iter()
        .filter(|action| action.output == output)
        .filter(|action| {
            action.args.iter().all(|(_, kind)| match kind {
                ArgKind::Int => !int_vars.is_empty(),
                ArgKind::ListInt => !list_vars.is_empty(),
            })
        })
        .collect();
    if eligible.is_empty() {
        eligible = ACTIONS
            .iter()
            .filter(|action| action.output == ArgKind::Int)
            .collect();
    }
    let idx = rng.gen_range(0..eligible.len());
    eligible[idx]
}

fn build_call(
    rng: &mut StdRng,
    action: &ActionDef,
    int_vars: &[String],
    list_vars: &[String],
    trace_counter: &mut i64,
    var_counter: &mut usize,
) -> (CallSpec, ArgKind) {
    let trace_id = next_trace_id(trace_counter);
    let mut args: Vec<(String, String)> = Vec::new();
    for (name, kind) in action.args {
        let expr = match kind {
            ArgKind::Int => choose_int_expr(rng, int_vars),
            ArgKind::ListInt => choose_list_var(rng, list_vars),
        };
        args.push((name.to_string(), expr));
    }
    let output_var = next_var(var_counter);
    (
        CallSpec {
            action: action.name,
            args,
            output_var,
            trace_id,
        },
        action.output,
    )
}

fn build_call_with_output(
    rng: &mut StdRng,
    action: &ActionDef,
    int_vars: &[String],
    list_vars: &[String],
    trace_counter: &mut i64,
    output_var: String,
) -> (CallSpec, ArgKind) {
    let trace_id = next_trace_id(trace_counter);
    let mut args: Vec<(String, String)> = Vec::new();
    for (name, kind) in action.args {
        let expr = match kind {
            ArgKind::Int => choose_int_expr(rng, int_vars),
            ArgKind::ListInt => choose_list_var(rng, list_vars),
        };
        args.push((name.to_string(), expr));
    }
    (
        CallSpec {
            action: action.name,
            args,
            output_var,
            trace_id,
        },
        action.output,
    )
}

fn choose_int_expr(rng: &mut StdRng, int_vars: &[String]) -> String {
    if !int_vars.is_empty() && rng.gen_bool(0.7) {
        let idx = rng.gen_range(0..int_vars.len());
        int_vars[idx].clone()
    } else {
        rng.gen_range(-9..=9).to_string()
    }
}

fn choose_list_var(rng: &mut StdRng, list_vars: &[String]) -> String {
    let idx = rng.gen_range(0..list_vars.len());
    list_vars[idx].clone()
}

fn choose_return_var(rng: &mut StdRng, int_vars: &[String], list_vars: &[String]) -> String {
    let total = int_vars.len() + list_vars.len();
    let idx = rng.gen_range(0..total);
    if idx < int_vars.len() {
        int_vars[idx].clone()
    } else {
        list_vars[idx - int_vars.len()].clone()
    }
}

fn next_var(counter: &mut usize) -> String {
    *counter += 1;
    format!("v{counter}")
}

fn next_trace_id(counter: &mut i64) -> i64 {
    let trace_id = *counter;
    *counter += 1;
    trace_id
}

fn reserve_trace_block(counter: &mut i64, size: i64) -> i64 {
    let base = *counter;
    *counter += size;
    base
}

fn add_var(kind: ArgKind, name: &str, int_vars: &mut Vec<String>, list_vars: &mut Vec<String>) {
    match kind {
        ArgKind::Int => int_vars.push(name.to_string()),
        ArgKind::ListInt => list_vars.push(name.to_string()),
    }
}

fn render_trace_call(call: &CallSpec) -> String {
    let args = call
        .args
        .iter()
        .map(|(name, expr)| format!("{name}={expr}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}({args}, trace_id={})", call.action, call.trace_id)
}

fn push_line(lines: &mut Vec<String>, indent: usize, line: impl Into<String>) {
    let line = line.into();
    lines.push(format!("{:indent$}{}", "", line, indent = indent * 4));
}

fn emit_random_action(
    rng: &mut StdRng,
    lines: &mut Vec<String>,
    indent: usize,
    int_vars: &mut Vec<String>,
    list_vars: &mut Vec<String>,
    trace_counter: &mut i64,
    var_counter: &mut usize,
) {
    let roll: u8 = rng.gen_range(0..100);
    if roll < 30 {
        let action_a = choose_action(rng, int_vars, list_vars);
        let action_b = choose_action(rng, int_vars, list_vars);
        let (call_a, kind_a) = build_call(
            rng,
            action_a,
            int_vars,
            list_vars,
            trace_counter,
            var_counter,
        );
        let (call_b, kind_b) = build_call(
            rng,
            action_b,
            int_vars,
            list_vars,
            trace_counter,
            var_counter,
        );
        push_line(
            lines,
            indent,
            format!(
                "{}, {} = await asyncio.gather(",
                call_a.output_var, call_b.output_var
            ),
        );
        push_line(
            lines,
            indent + 1,
            format!("{},", render_trace_call(&call_a)),
        );
        push_line(
            lines,
            indent + 1,
            format!("{},", render_trace_call(&call_b)),
        );
        push_line(lines, indent + 1, "return_exceptions=True,");
        push_line(lines, indent, ")");
        add_var(kind_a, &call_a.output_var, int_vars, list_vars);
        add_var(kind_b, &call_b.output_var, int_vars, list_vars);
        return;
    }

    let action = choose_action(rng, int_vars, list_vars);
    let (call, output_kind) =
        build_call(rng, action, int_vars, list_vars, trace_counter, var_counter);
    push_line(
        lines,
        indent,
        format!("{} = await {}", call.output_var, render_trace_call(&call)),
    );
    add_var(output_kind, &call.output_var, int_vars, list_vars);
}

fn emit_if_block(
    rng: &mut StdRng,
    lines: &mut Vec<String>,
    int_vars: &mut Vec<String>,
    list_vars: &mut Vec<String>,
    trace_counter: &mut i64,
    var_counter: &mut usize,
) {
    let output_var = next_var(var_counter);
    let action_true = choose_action_with_output(rng, int_vars, list_vars, ArgKind::Int);
    let action_mid = choose_action_with_output(rng, int_vars, list_vars, ArgKind::Int);
    let action_false = choose_action_with_output(rng, int_vars, list_vars, ArgKind::Int);

    let (call_true, _) = build_call_with_output(
        rng,
        action_true,
        int_vars,
        list_vars,
        trace_counter,
        output_var.clone(),
    );
    let (call_mid, _) = build_call_with_output(
        rng,
        action_mid,
        int_vars,
        list_vars,
        trace_counter,
        output_var.clone(),
    );
    let (call_false, _) = build_call_with_output(
        rng,
        action_false,
        int_vars,
        list_vars,
        trace_counter,
        output_var.clone(),
    );

    push_line(lines, 2, "if seed > 0:");
    push_line(
        lines,
        3,
        format!("{} = await {}", output_var, render_trace_call(&call_true)),
    );
    push_line(lines, 2, "elif seed == 0:");
    push_line(
        lines,
        3,
        format!("{} = await {}", output_var, render_trace_call(&call_mid)),
    );
    push_line(lines, 2, "else:");
    push_line(
        lines,
        3,
        format!("{} = await {}", output_var, render_trace_call(&call_false)),
    );

    add_var(ArgKind::Int, &output_var, int_vars, list_vars);
}

fn emit_for_loop(
    rng: &mut StdRng,
    lines: &mut Vec<String>,
    list_vars: &mut Vec<String>,
    trace_counter: &mut i64,
) {
    let results_var = "loop_results".to_string();
    let trace_base = reserve_trace_block(trace_counter, 10_000);
    push_line(lines, 2, format!("{results_var} = []"));
    push_line(lines, 2, "for item in items:");
    let nested = rng.gen_bool(0.5);
    if nested {
        push_line(lines, 3, "inner_sum = 0");
        push_line(lines, 3, "for other in items:");
        let trace_expr = format!("{trace_base} + 5000 + item * 100 + other");
        let call = format!("add(a=inner_sum, b=other, trace_id={trace_expr})");
        push_line(lines, 4, format!("inner_sum = await {call}"));
        push_line(lines, 3, format!("{results_var}.append(inner_sum)"));
    } else {
        let trace_expr = format!("{trace_base} + 5000 + item");
        let call = format!("add(a=item, b=factor, trace_id={trace_expr})");
        push_line(lines, 3, format!("processed = await {call}"));
        push_line(lines, 3, format!("{results_var}.append(processed)"));
    }
    if rng.gen_bool(0.6) {
        let trace_raise = format!("{trace_base} + 9000 + item");
        let trace_fallback = format!("{trace_base} + 9500 + item");
        push_line(lines, 3, "try:");
        push_line(
            lines,
            4,
            format!(
                "failed = await self.run_action(raise_error(kind=0, trace_id={trace_raise}), retry=RetryPolicy(attempts=1))"
            ),
        );
        push_line(lines, 3, "except ValueError:");
        push_line(
            lines,
            4,
            format!("failed = await add(a=item, b=factor, trace_id={trace_fallback})"),
        );
        push_line(lines, 3, format!("{results_var}.append(failed)"));
    }
    list_vars.push(results_var);
}

fn emit_try_except(
    rng: &mut StdRng,
    lines: &mut Vec<String>,
    int_vars: &mut Vec<String>,
    list_vars: &mut Vec<String>,
    trace_counter: &mut i64,
    var_counter: &mut usize,
) {
    let output_var = next_var(var_counter);
    let kind_value = if rng.gen_bool(0.5) { 0 } else { 1 };
    let trace_id = next_trace_id(trace_counter);
    let raise_call = format!(
        "self.run_action(raise_error(kind={kind_value}, trace_id={trace_id}), retry=RetryPolicy(attempts=1))"
    );

    let action_value_error = choose_action_with_output(rng, int_vars, list_vars, ArgKind::Int);
    let action_lookup_error = choose_action_with_output(rng, int_vars, list_vars, ArgKind::Int);
    let action_exception = choose_action_with_output(rng, int_vars, list_vars, ArgKind::Int);
    let (call_value, _) = build_call_with_output(
        rng,
        action_value_error,
        int_vars,
        list_vars,
        trace_counter,
        output_var.clone(),
    );
    let (call_lookup, _) = build_call_with_output(
        rng,
        action_lookup_error,
        int_vars,
        list_vars,
        trace_counter,
        output_var.clone(),
    );
    let (call_exception, _) = build_call_with_output(
        rng,
        action_exception,
        int_vars,
        list_vars,
        trace_counter,
        output_var.clone(),
    );

    push_line(lines, 2, "try:");
    push_line(lines, 3, format!("{} = await {}", output_var, raise_call));
    push_line(lines, 2, "except ValueError:");
    push_line(
        lines,
        3,
        format!("{} = await {}", output_var, render_trace_call(&call_value)),
    );
    push_line(lines, 2, "except LookupError:");
    push_line(
        lines,
        3,
        format!("{} = await {}", output_var, render_trace_call(&call_lookup)),
    );
    push_line(lines, 2, "except Exception:");
    push_line(
        lines,
        3,
        format!(
            "{} = await {}",
            output_var,
            render_trace_call(&call_exception)
        ),
    );

    add_var(ArgKind::Int, &output_var, int_vars, list_vars);
}

fn render_python_module(body_lines: &[String]) -> String {
    let mut source = String::new();
    source.push_str("import asyncio\n");
    source.push_str("from rappel import action, workflow\n");
    source.push_str("from rappel.workflow import Workflow, RetryPolicy\n\n");
    source.push_str("TRACE_LOG = []\n\n");
    source.push_str("def record(trace_id: object, inputs: dict, output: object) -> None:\n");
    source.push_str("    normalized = dict(inputs)\n");
    source.push_str("    normalized[\"trace_id\"] = str(trace_id)\n");
    source.push_str(
        "    TRACE_LOG.append({\"trace_id\": str(trace_id), \"inputs\": normalized, \"output\": output})\n",
    );
    source.push('\n');
    source.push_str("def exception_payload(exc: BaseException) -> dict:\n");
    source.push_str("    return {\n");
    source.push_str("        \"__exception__\": {\n");
    source.push_str("            \"type\": exc.__class__.__name__,\n");
    source.push_str("            \"module\": exc.__class__.__module__,\n");
    source.push_str("            \"message\": str(exc),\n");
    source.push_str("            \"traceback\": \"\",\n");
    source.push_str("            \"values\": {},\n");
    source.push_str("        }\n");
    source.push_str("    }\n\n");
    source.push_str("@action\n");
    source.push_str("async def add(a: int, b: int, trace_id: int) -> int:\n");
    source.push_str("    result = a + b\n");
    source.push_str("    record(trace_id, {\"a\": a, \"b\": b, \"trace_id\": trace_id}, result)\n");
    source.push_str("    return result\n\n");
    source.push_str("@action\n");
    source.push_str("async def mul(a: int, b: int, trace_id: int) -> int:\n");
    source.push_str("    result = a * b\n");
    source.push_str("    record(trace_id, {\"a\": a, \"b\": b, \"trace_id\": trace_id}, result)\n");
    source.push_str("    return result\n\n");
    source.push_str("@action\n");
    source.push_str("async def negate(value: int, trace_id: int) -> int:\n");
    source.push_str("    result = -value\n");
    source.push_str("    record(trace_id, {\"value\": value, \"trace_id\": trace_id}, result)\n");
    source.push_str("    return result\n\n");
    source.push_str("@action\n");
    source.push_str("async def sum_list(items: list[int], trace_id: int) -> int:\n");
    source.push_str("    result = sum(items)\n");
    source.push_str("    record(trace_id, {\"items\": items, \"trace_id\": trace_id}, result)\n");
    source.push_str("    return result\n\n");
    source.push_str("@action\n");
    source.push_str("async def len_list(items: list[int], trace_id: int) -> int:\n");
    source.push_str("    result = len(items)\n");
    source.push_str("    record(trace_id, {\"items\": items, \"trace_id\": trace_id}, result)\n");
    source.push_str("    return result\n\n");
    source.push_str("@action\n");
    source.push_str("async def make_list(a: int, b: int, c: int, trace_id: int) -> list[int]:\n");
    source.push_str("    result = [a, b, c]\n");
    source.push_str(
        "    record(trace_id, {\"a\": a, \"b\": b, \"c\": c, \"trace_id\": trace_id}, result)\n",
    );
    source.push_str("    return result\n\n");
    source.push_str("@action\n");
    source.push_str(
        "async def append_list(items: list[int], value: int, trace_id: int) -> list[int]:\n",
    );
    source.push_str("    result = items + [value]\n");
    source.push_str("    record(trace_id, {\"items\": items, \"value\": value, \"trace_id\": trace_id}, result)\n");
    source.push_str("    return result\n\n");
    source.push_str("@action\n");
    source.push_str(
        "async def scale_list(items: list[int], factor: int, trace_id: int) -> list[int]:\n",
    );
    source.push_str("    result = [item * factor for item in items]\n");
    source.push_str("    record(trace_id, {\"items\": items, \"factor\": factor, \"trace_id\": trace_id}, result)\n");
    source.push_str("    return result\n\n");
    source.push_str("@action\n");
    source.push_str(
        "async def concat_lists(left: list[int], right: list[int], trace_id: int) -> list[int]:\n",
    );
    source.push_str("    result = left + right\n");
    source.push_str("    record(trace_id, {\"left\": left, \"right\": right, \"trace_id\": trace_id}, result)\n");
    source.push_str("    return result\n\n");
    source.push_str("@action\n");
    source.push_str(
        "async def take_slice(items: list[int], end: int, trace_id: int) -> list[int]:\n",
    );
    source.push_str("    result = items[:end]\n");
    source.push_str(
        "    record(trace_id, {\"items\": items, \"end\": end, \"trace_id\": trace_id}, result)\n",
    );
    source.push_str("    return result\n\n");
    source.push_str("@action\n");
    source.push_str("async def raise_error(kind: int, trace_id: int) -> int:\n");
    source.push_str("    if kind == 0:\n");
    source.push_str("        exc = ValueError(\"boom\")\n");
    source.push_str("    else:\n");
    source.push_str("        exc = KeyError(\"boom\")\n");
    source.push_str(
        "    record(trace_id, {\"kind\": kind, \"trace_id\": trace_id}, exception_payload(exc))\n",
    );
    source.push_str("    raise exc\n\n");
    source.push_str("@workflow\n");
    source.push_str("class FuzzWorkflow(Workflow):\n");
    source
        .push_str("    async def run(self, seed: int, items: list[int], factor: int) -> object:\n");
    for line in body_lines {
        source.push_str(line);
        source.push('\n');
    }
    source
}

async fn run_case(case_idx: usize, case: &FuzzCase, keep: bool) -> Result<()> {
    let tempdir = TempDir::new().context("create tempdir")?;
    let module_path = tempdir.path().join("case.py");
    let inputs_path = tempdir.path().join("inputs.json");
    let trace_out = tempdir.path().join("trace.json");
    let rust_trace_out = tempdir.path().join("rust-trace.json");
    let registration_out = tempdir.path().join("registration.bin");

    std::fs::write(&module_path, &case.python_source).context("write python module")?;
    std::fs::write(&inputs_path, case.inputs.to_string()).context("write inputs")?;

    run_python_oracle(&module_path, &inputs_path, &trace_out, &registration_out)?;
    let oracle_trace = read_trace(&trace_out)?;
    let registration_bytes = std::fs::read(&registration_out).context("read registration bytes")?;

    let rust_trace = run_in_memory(&registration_bytes).await?;
    if let Err(err) = compare_traces(&oracle_trace, &rust_trace) {
        std::fs::write(
            &rust_trace_out,
            serde_json::to_string_pretty(&rust_trace).context("serialize rust trace")?,
        )
        .context("write rust trace")?;
        let preserved = preserve_tempdir(tempdir, keep, case_idx)?;
        if let Some(path) = preserved {
            println!("case {case_idx} artifacts preserved at {}", path.display());
        }
        return Err(err);
    }

    Ok(())
}

fn run_python_oracle(
    module_path: &Path,
    inputs_path: &Path,
    trace_out: &Path,
    registration_out: &Path,
) -> Result<()> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let python_dir = repo_root.join("python");
    let script_path = python_dir.join("scripts/fuzz_oracle.py");
    let status = Command::new("uv")
        .current_dir(&python_dir)
        .env("PYTHONPATH", python_dir.join("src"))
        .args([
            "run",
            "python",
            script_path.to_str().context("script path")?,
            "--module-path",
            module_path.to_str().context("module path")?,
            "--inputs-path",
            inputs_path.to_str().context("inputs path")?,
            "--trace-out",
            trace_out.to_str().context("trace path")?,
            "--registration-out",
            registration_out.to_str().context("registration path")?,
        ])
        .status()
        .context("run python oracle")?;
    if !status.success() {
        bail!("python oracle failed with status {status}");
    }
    Ok(())
}

fn read_trace(path: &Path) -> Result<OracleTrace> {
    let contents = std::fs::read_to_string(path).context("read trace")?;
    serde_json::from_str(&contents).context("parse trace json")
}

async fn run_in_memory(registration_bytes: &[u8]) -> Result<OracleTrace> {
    let registration =
        proto::WorkflowRegistration::decode(registration_bytes).context("decode registration")?;
    let mut executor = InMemoryWorkflowExecutor::from_registration(registration, true)
        .context("build in-memory executor")?;
    let mut step = executor.start().await.context("start executor")?;
    let mut actions: Vec<ActionTrace> = Vec::new();
    let mut dispatches: VecDeque<proto::ActionDispatch> = VecDeque::new();
    dispatches.extend(step.dispatches);

    while step.completed_payload.is_none() {
        let dispatch = dispatches.pop_front().context("no dispatches available")?;
        let inputs = dispatch
            .kwargs
            .as_ref()
            .map(|kwargs| kwargs.encode_to_vec())
            .and_then(|bytes| workflow_arguments_to_json(&bytes))
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
        let normalized_inputs = normalize_inputs(inputs);
        let trace_id = normalized_inputs
            .get("trace_id")
            .and_then(Value::as_str)
            .context("trace_id missing")?;
        match execute_action(&dispatch.action_name, &normalized_inputs)? {
            ActionOutcome::Ok(output) => {
                actions.push(ActionTrace {
                    trace_id: trace_id.to_string(),
                    inputs: normalized_inputs.clone(),
                    output: output.clone(),
                });
                let payload = build_result_payload(&output);
                let action_result = proto::ActionResult {
                    action_id: dispatch.action_id,
                    success: true,
                    payload: Some(payload),
                    worker_start_ns: 0,
                    worker_end_ns: 0,
                    dispatch_token: None,
                    error_type: None,
                    error_message: None,
                };
                step = executor
                    .handle_action_result(action_result)
                    .await
                    .context("handle action result")?;
                dispatches.extend(step.dispatches);
            }
            ActionOutcome::Err(err) => {
                let output = exception_json(&err);
                actions.push(ActionTrace {
                    trace_id: trace_id.to_string(),
                    inputs: normalized_inputs.clone(),
                    output: output.clone(),
                });
                let payload = build_error_payload(&err);
                let action_result = proto::ActionResult {
                    action_id: dispatch.action_id,
                    success: false,
                    payload: Some(payload),
                    worker_start_ns: 0,
                    worker_end_ns: 0,
                    dispatch_token: None,
                    error_type: Some(err.exc_type.clone()),
                    error_message: Some(err.message.clone()),
                };
                step = executor
                    .handle_action_result(action_result)
                    .await
                    .context("handle action result")?;
                dispatches.extend(step.dispatches);
            }
        }
    }

    let completed_payload = step
        .completed_payload
        .as_ref()
        .context("missing completed payload")?;
    let workflow_json =
        workflow_arguments_to_json(completed_payload).context("decode workflow output")?;
    let workflow_output = workflow_json.get("result").cloned().unwrap_or(Value::Null);

    Ok(OracleTrace {
        actions,
        workflow_output,
    })
}

fn build_result_payload(value: &Value) -> proto::WorkflowArguments {
    proto::WorkflowArguments {
        arguments: vec![proto::WorkflowArgument {
            key: "result".to_string(),
            value: Some(json_to_workflow_argument_value(value)),
        }],
    }
}

fn build_error_payload(err: &FuzzError) -> proto::WorkflowArguments {
    proto::WorkflowArguments {
        arguments: vec![proto::WorkflowArgument {
            key: "error".to_string(),
            value: Some(build_exception_value(err)),
        }],
    }
}

fn build_exception_value(err: &FuzzError) -> proto::WorkflowArgumentValue {
    proto::WorkflowArgumentValue {
        kind: Some(proto::workflow_argument_value::Kind::Exception(
            proto::WorkflowErrorValue {
                r#type: err.exc_type.clone(),
                module: err.module.clone(),
                message: err.message.clone(),
                traceback: String::new(),
                values: None,
                type_hierarchy: err.type_hierarchy.clone(),
            },
        )),
    }
}

fn exception_json(err: &FuzzError) -> Value {
    workflow_argument_value_to_json(&build_exception_value(err))
}

fn execute_action(action: &str, inputs: &Value) -> Result<ActionOutcome> {
    let map = inputs.as_object().context("inputs must be an object")?;
    match action {
        "add" => {
            let a = get_int(map, "a")?;
            let b = get_int(map, "b")?;
            Ok(ActionOutcome::Ok(Value::from(a + b)))
        }
        "mul" => {
            let a = get_int(map, "a")?;
            let b = get_int(map, "b")?;
            Ok(ActionOutcome::Ok(Value::from(a * b)))
        }
        "negate" => {
            let v = get_int(map, "value")?;
            Ok(ActionOutcome::Ok(Value::from(-v)))
        }
        "sum_list" => {
            let items = get_list(map, "items")?;
            Ok(ActionOutcome::Ok(Value::from(items.iter().sum::<i64>())))
        }
        "len_list" => {
            let items = get_list(map, "items")?;
            Ok(ActionOutcome::Ok(Value::from(items.len() as i64)))
        }
        "make_list" => {
            let a = get_int(map, "a")?;
            let b = get_int(map, "b")?;
            let c = get_int(map, "c")?;
            Ok(ActionOutcome::Ok(Value::Array(vec![
                Value::from(a),
                Value::from(b),
                Value::from(c),
            ])))
        }
        "append_list" => {
            let mut items = get_list(map, "items")?;
            let value = get_int(map, "value")?;
            items.push(value);
            Ok(ActionOutcome::Ok(int_list_value(items)))
        }
        "scale_list" => {
            let items = get_list(map, "items")?;
            let factor = get_int(map, "factor")?;
            Ok(ActionOutcome::Ok(int_list_value(
                items.into_iter().map(|v| v * factor).collect(),
            )))
        }
        "concat_lists" => {
            let left = get_list(map, "left")?;
            let right = get_list(map, "right")?;
            let mut combined = left;
            combined.extend(right);
            Ok(ActionOutcome::Ok(int_list_value(combined)))
        }
        "take_slice" => {
            let items = get_list(map, "items")?;
            let end = get_int(map, "end")?;
            let len = items.len() as i64;
            let mut end = if end >= 0 { end } else { len + end };
            if end < 0 {
                end = 0;
            }
            if end > len {
                end = len;
            }
            let slice = items.into_iter().take(end as usize).collect::<Vec<_>>();
            Ok(ActionOutcome::Ok(int_list_value(slice)))
        }
        "raise_error" => {
            let kind = get_int(map, "kind")?;
            Ok(ActionOutcome::Err(build_fuzz_error(kind)))
        }
        other => bail!("unknown action {other}"),
    }
}

fn build_fuzz_error(kind: i64) -> FuzzError {
    if kind == 0 {
        FuzzError {
            exc_type: "ValueError".to_string(),
            message: "boom".to_string(),
            module: "builtins".to_string(),
            type_hierarchy: vec![
                "ValueError".to_string(),
                "Exception".to_string(),
                "BaseException".to_string(),
            ],
        }
    } else {
        FuzzError {
            exc_type: "KeyError".to_string(),
            message: "'boom'".to_string(),
            module: "builtins".to_string(),
            type_hierarchy: vec![
                "KeyError".to_string(),
                "LookupError".to_string(),
                "Exception".to_string(),
                "BaseException".to_string(),
            ],
        }
    }
}

fn get_int(map: &serde_json::Map<String, Value>, key: &str) -> Result<i64> {
    let value = map.get(key).context("missing int key")?;
    value
        .as_i64()
        .context("expected integer value")
        .with_context(|| format!("key {key}"))
}

fn get_list(map: &serde_json::Map<String, Value>, key: &str) -> Result<Vec<i64>> {
    let value = map.get(key).context("missing list key")?;
    let array = value.as_array().context("expected list value")?;
    array
        .iter()
        .map(|item| item.as_i64().context("expected int in list"))
        .collect::<Result<Vec<_>>>()
        .with_context(|| format!("key {key}"))
}

fn int_list_value(items: Vec<i64>) -> Value {
    Value::Array(items.into_iter().map(Value::from).collect())
}

fn normalize_inputs(mut inputs: Value) -> Value {
    if let Value::Object(ref mut map) = inputs
        && let Some(trace_value) = map.get("trace_id").cloned()
    {
        let trace_str = match trace_value {
            Value::String(s) => s,
            Value::Number(n) => n.to_string(),
            other => other.to_string(),
        };
        map.insert("trace_id".to_string(), Value::String(trace_str));
    }
    inputs
}

fn compare_traces(python: &OracleTrace, rust: &OracleTrace) -> Result<()> {
    if python.workflow_output != rust.workflow_output {
        bail!(
            "workflow output mismatch: python={} rust={}",
            python.workflow_output,
            rust.workflow_output
        );
    }

    let python_map = map_actions(&python.actions)?;
    let rust_map = map_actions(&rust.actions)?;

    if python_map.len() != rust_map.len() {
        bail!(
            "action count mismatch: python={} rust={}",
            python_map.len(),
            rust_map.len()
        );
    }

    for (trace_id, rust_action) in rust_map {
        let python_action = python_map.get(&trace_id).context("missing python trace")?;
        if python_action.inputs != rust_action.inputs {
            bail!(
                "inputs mismatch for {trace_id}: python={} rust={}",
                python_action.inputs,
                rust_action.inputs
            );
        }
        if python_action.output != rust_action.output {
            bail!(
                "output mismatch for {trace_id}: python={} rust={}",
                python_action.output,
                rust_action.output
            );
        }
    }

    Ok(())
}

fn map_actions(actions: &[ActionTrace]) -> Result<HashMap<String, ActionTrace>> {
    let mut map = HashMap::new();
    for action in actions {
        if map
            .insert(action.trace_id.clone(), action.clone())
            .is_some()
        {
            bail!("duplicate trace_id {}", action.trace_id);
        }
    }
    Ok(map)
}

fn preserve_tempdir(tempdir: TempDir, keep: bool, case_idx: usize) -> Result<Option<PathBuf>> {
    if keep {
        return Ok(Some(tempdir.keep()));
    }
    let path = tempdir.keep();
    let preserved = path
        .parent()
        .map(|parent| parent.join(format!("rappel-fuzz-case-{case_idx}")));
    if let Some(dest) = preserved {
        if dest.exists() {
            std::fs::remove_dir_all(&dest).context("remove existing preserve dir")?;
        }
        std::fs::rename(&path, &dest).context("preserve tempdir")?;
        Ok(Some(dest))
    } else {
        Ok(None)
    }
}
