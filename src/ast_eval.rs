use std::{collections::HashMap, convert::TryFrom};

use anyhow::{Context, Result, anyhow};
use serde_json::{Number, Value};

use crate::messages::proto;

pub type EvalContext = HashMap<String, Value>;

pub fn eval_expr(expr: &proto::Expr, ctx: &EvalContext) -> Result<Value> {
    match expr.kind.as_ref().context("expr missing kind")? {
        proto::expr::Kind::Name(name) => ctx
            .get(&name.id)
            .cloned()
            .with_context(|| format!("name '{}' not found", name.id)),
        proto::expr::Kind::Constant(c) => Ok(constant_to_value(c)),
        proto::expr::Kind::Attribute(attr) => {
            let base = eval_expr(attr.value.as_ref().context("attr missing value")?, ctx)?;
            attr_lookup(&base, &attr.attr)
        }
        proto::expr::Kind::Subscript(sub) => {
            let base = eval_expr(sub.value.as_ref().context("subscript missing value")?, ctx)?;
            let idx = eval_expr(sub.slice.as_ref().context("subscript missing slice")?, ctx)?;
            subscript_lookup(&base, &idx)
        }
        proto::expr::Kind::BinOp(bin) => {
            let left = eval_expr(bin.left.as_ref().context("binop missing left")?, ctx)?;
            let right = eval_expr(bin.right.as_ref().context("binop missing right")?, ctx)?;
            eval_bin_op(&left, &right, bin.op())
        }
        proto::expr::Kind::BoolOp(bop) => {
            let op = bop.op();
            let mut last = Value::Bool(false);
            match op {
                proto::BoolOpKind::And => {
                    for v in &bop.values {
                        last = eval_expr(v, ctx)?;
                        if !is_truthy(&last) {
                            return Ok(last);
                        }
                    }
                    Ok(last)
                }
                proto::BoolOpKind::Or => {
                    for v in &bop.values {
                        last = eval_expr(v, ctx)?;
                        if is_truthy(&last) {
                            return Ok(last);
                        }
                    }
                    Ok(last)
                }
                _ => Err(anyhow!("unsupported bool op")),
            }
        }
        proto::expr::Kind::Compare(cmp) => eval_compare(cmp, ctx),
        proto::expr::Kind::Call(call) => eval_call(call, ctx),
        proto::expr::Kind::List(list) => {
            let mut items = Vec::with_capacity(list.elts.len());
            for elt in &list.elts {
                items.push(eval_expr(elt, ctx)?);
            }
            Ok(Value::Array(items))
        }
        proto::expr::Kind::Tuple(tuple) => {
            let mut items = Vec::with_capacity(tuple.elts.len());
            for elt in &tuple.elts {
                items.push(eval_expr(elt, ctx)?);
            }
            Ok(Value::Array(items))
        }
        proto::expr::Kind::Dict(dict) => {
            let mut map = serde_json::Map::new();
            for (k, v) in dict.keys.iter().zip(dict.values.iter()) {
                let key_val = eval_expr(k, ctx)?;
                let Value::String(key) = key_val else {
                    return Err(anyhow!("dict key must be string"));
                };
                let value = eval_expr(v, ctx)?;
                map.insert(key, value);
            }
            Ok(Value::Object(map))
        }
        proto::expr::Kind::UnaryOp(op) => {
            let operand = eval_expr(op.operand.as_ref().context("unary missing operand")?, ctx)?;
            eval_unary_op(&operand, op.op())
        }
    }
}

pub fn eval_stmt(stmt: &proto::Stmt, ctx: &mut EvalContext) -> Result<()> {
    match stmt.kind.as_ref().context("stmt missing kind")? {
        proto::stmt::Kind::Assign(assign) => {
            let value = eval_expr(assign.value.as_ref().context("assign missing value")?, ctx)?;
            for target in &assign.targets {
                assign_target(target, value.clone(), ctx)?;
            }
            Ok(())
        }
        proto::stmt::Kind::Expr(expr) => {
            let _ = eval_expr(expr, ctx)?;
            Ok(())
        }
    }
}

fn assign_target(target: &proto::Expr, value: Value, ctx: &mut EvalContext) -> Result<()> {
    match target.kind.as_ref().context("assign target missing kind")? {
        proto::expr::Kind::Name(name) => {
            ctx.insert(name.id.clone(), value);
            Ok(())
        }
        proto::expr::Kind::Subscript(sub) => {
            let base = eval_expr(sub.value.as_ref().context("subscript missing value")?, ctx)?;
            let idx = eval_expr(sub.slice.as_ref().context("subscript missing slice")?, ctx)?;
            set_subscript(base, idx, value, ctx, sub.value.as_ref().unwrap())
        }
        proto::expr::Kind::Attribute(attr) => {
            let base = eval_expr(attr.value.as_ref().context("attr missing value")?, ctx)?;
            set_attribute(base, &attr.attr, value, ctx, attr.value.as_ref().unwrap())
        }
        _ => Err(anyhow!("unsupported assignment target")),
    }
}

fn set_subscript(
    base: Value,
    idx: Value,
    value: Value,
    ctx: &mut EvalContext,
    base_expr: &proto::Expr,
) -> Result<()> {
    match (base, idx) {
        (Value::Array(mut arr), Value::Number(n)) => {
            let i = n
                .as_i64()
                .ok_or_else(|| anyhow!("subscript index must be int"))?;
            if i < 0 {
                return Err(anyhow!("subscript index out of range"));
            }
            let idx_usize = i as usize;
            if idx_usize >= arr.len() {
                return Err(anyhow!("subscript index out of range"));
            }
            arr[idx_usize] = value;
            let name = base_name(base_expr)?;
            ctx.insert(name, Value::Array(arr));
            Ok(())
        }
        (Value::Object(mut map), Value::String(key)) => {
            map.insert(key.clone(), value);
            let name = base_name(base_expr)?;
            ctx.insert(name, Value::Object(map));
            Ok(())
        }
        _ => Err(anyhow!("unsupported subscript assignment target")),
    }
}

fn set_attribute(
    base: Value,
    attr: &str,
    value: Value,
    ctx: &mut EvalContext,
    base_expr: &proto::Expr,
) -> Result<()> {
    if let Value::Object(mut map) = base {
        map.insert(attr.to_string(), value);
        let name = base_name(base_expr)?;
        ctx.insert(name, Value::Object(map));
        Ok(())
    } else {
        Err(anyhow!("unsupported attribute assignment target"))
    }
}

fn base_name(expr: &proto::Expr) -> Result<String> {
    match expr.kind.as_ref().context("base expr missing kind")? {
        proto::expr::Kind::Name(name) => Ok(name.id.clone()),
        _ => Err(anyhow!("nested assignment targets not supported")),
    }
}

fn constant_to_value(c: &proto::Constant) -> Value {
    match c.value.as_ref() {
        Some(proto::constant::Value::StringValue(s)) => Value::String(s.clone()),
        Some(proto::constant::Value::FloatValue(f)) => {
            Value::Number(Number::from_f64(*f).unwrap_or(Number::from(0)))
        }
        Some(proto::constant::Value::IntValue(i)) => Value::Number(Number::from(*i)),
        Some(proto::constant::Value::BoolValue(b)) => Value::Bool(*b),
        Some(proto::constant::Value::IsNone(_)) => Value::Null,
        None => Value::Null,
    }
}

fn attr_lookup(base: &Value, attr: &str) -> Result<Value> {
    match base {
        Value::Object(map) => map
            .get(attr)
            .cloned()
            .with_context(|| format!("attribute '{}' not found", attr)),
        _ => Err(anyhow!("attribute access requires object")),
    }
}

fn subscript_lookup(base: &Value, idx: &Value) -> Result<Value> {
    match (base, idx) {
        (Value::Array(arr), Value::Number(n)) => {
            let i = n
                .as_i64()
                .ok_or_else(|| anyhow!("subscript index must be int"))?;
            if i < 0 {
                return Err(anyhow!("subscript index out of range"));
            }
            let idx_usize = i as usize;
            arr.get(idx_usize)
                .cloned()
                .with_context(|| format!("index {} out of range", idx_usize))
        }
        (Value::Object(map), Value::String(key)) => map
            .get(key)
            .cloned()
            .with_context(|| format!("key '{}' not found", key)),
        (Value::String(s), Value::Number(n)) => {
            let i = n
                .as_i64()
                .ok_or_else(|| anyhow!("subscript index must be int"))?;
            let idx_usize = i
                .try_into()
                .map_err(|_| anyhow!("subscript index out of range"))?;
            s.chars()
                .nth(idx_usize)
                .map(|c| Value::String(c.to_string()))
                .with_context(|| format!("index {} out of range", idx_usize))
        }
        _ => Err(anyhow!("unsupported subscript lookup")),
    }
}

fn eval_bin_op(left: &Value, right: &Value, op: proto::BinOpKind) -> Result<Value> {
    match op {
        proto::BinOpKind::Add => add_values(left, right),
        proto::BinOpKind::Sub => numeric_op(left, right, |a, b| a - b),
        proto::BinOpKind::Mult => numeric_op(left, right, |a, b| a * b),
        proto::BinOpKind::Div => numeric_op(left, right, |a, b| a / b),
        proto::BinOpKind::Mod => numeric_op(left, right, |a, b| a % b),
        proto::BinOpKind::Floordiv => numeric_op(left, right, |a, b| (a / b).floor()),
        proto::BinOpKind::Pow => numeric_op(left, right, |a, b| a.powf(b)),
        _ => Err(anyhow!("unsupported bin op")),
    }
}

fn add_values(left: &Value, right: &Value) -> Result<Value> {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => {
            if let (Some(ai), Some(bi)) = (a.as_i64(), b.as_i64()) {
                return Ok(Value::Number(Number::from(ai + bi)));
            }
            if let (Some(au), Some(bu)) = (a.as_u64(), b.as_u64()) {
                return Ok(Value::Number(Number::from(au + bu)));
            }
            let res = to_f64(&Value::Number(a.clone()))? + to_f64(&Value::Number(b.clone()))?;
            Ok(Value::Number(
                Number::from_f64(res).unwrap_or(Number::from(0)),
            ))
        }
        (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{a}{b}"))),
        (Value::Array(a), Value::Array(b)) => {
            let mut merged = a.clone();
            merged.extend(b.clone());
            Ok(Value::Array(merged))
        }
        _ => Err(anyhow!("unsupported add operands")),
    }
}

fn numeric_op<F>(left: &Value, right: &Value, f: F) -> Result<Value>
where
    F: Fn(f64, f64) -> f64,
{
    let a = to_f64(left)?;
    let b = to_f64(right)?;
    let res = f(a, b);
    Ok(Value::Number(
        Number::from_f64(res).unwrap_or(Number::from(0)),
    ))
}

fn to_f64(v: &Value) -> Result<f64> {
    match v {
        Value::Number(n) => n
            .as_f64()
            .or_else(|| n.as_i64().map(|i| i as f64))
            .context("number expected"),
        _ => Err(anyhow!("number expected")),
    }
}

fn eval_unary_op(operand: &Value, op: proto::UnaryOpKind) -> Result<Value> {
    match op {
        proto::UnaryOpKind::Usub => {
            let val = -to_f64(operand)?;
            Ok(Value::Number(
                Number::from_f64(val).unwrap_or(Number::from(0)),
            ))
        }
        proto::UnaryOpKind::Uadd => {
            let val = to_f64(operand)?;
            Ok(Value::Number(
                Number::from_f64(val).unwrap_or(Number::from(0)),
            ))
        }
        proto::UnaryOpKind::Not => Ok(Value::Bool(!is_truthy(operand))),
        _ => Err(anyhow!("unsupported unary op")),
    }
}

fn eval_compare(cmp: &proto::Compare, ctx: &EvalContext) -> Result<Value> {
    let mut left = eval_expr(cmp.left.as_ref().context("compare missing left")?, ctx)?;
    for (op, comp) in cmp.ops.iter().zip(cmp.comparators.iter()) {
        let right = eval_expr(comp, ctx)?;
        let op_kind = proto::CmpOpKind::try_from(*op).unwrap_or(proto::CmpOpKind::Unspecified);
        let ok = match op_kind {
            proto::CmpOpKind::Eq => left == right,
            proto::CmpOpKind::NotEq => left != right,
            proto::CmpOpKind::Lt => to_f64(&left)? < to_f64(&right)?,
            proto::CmpOpKind::LtE => to_f64(&left)? <= to_f64(&right)?,
            proto::CmpOpKind::Gt => to_f64(&left)? > to_f64(&right)?,
            proto::CmpOpKind::GtE => to_f64(&left)? >= to_f64(&right)?,
            proto::CmpOpKind::In => contains(&right, &left)?,
            proto::CmpOpKind::NotIn => !contains(&right, &left)?,
            proto::CmpOpKind::Is => is_identical(&left, &right),
            proto::CmpOpKind::IsNot => !is_identical(&left, &right),
            _ => return Err(anyhow!("unsupported compare op")),
        };
        if !ok {
            return Ok(Value::Bool(false));
        }
        left = right;
    }
    Ok(Value::Bool(true))
}

fn contains(container: &Value, item: &Value) -> Result<bool> {
    match container {
        Value::Array(arr) => Ok(arr.contains(item)),
        Value::String(s) => {
            let Value::String(needle) = item else {
                return Ok(false);
            };
            Ok(s.contains(needle.as_str()))
        }
        Value::Object(map) => {
            let Value::String(key) = item else {
                return Ok(false);
            };
            Ok(map.contains_key(key))
        }
        _ => Ok(false),
    }
}

fn is_identical(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        _ => a == b,
    }
}

fn eval_call(call: &proto::Call, ctx: &EvalContext) -> Result<Value> {
    let func_expr = call.func.as_ref().context("call missing func")?;
    let mut args = Vec::with_capacity(call.args.len());
    for arg in &call.args {
        args.push(eval_expr(arg, ctx)?);
    }
    match func_expr.kind.as_ref().context("call func missing kind")? {
        proto::expr::Kind::Name(name) => match name.id.as_str() {
            "len" => {
                if args.len() != 1 {
                    return Err(anyhow!("len expects 1 argument"));
                }
                let len = match &args[0] {
                    Value::Array(a) => a.len(),
                    Value::Object(o) => o.len(),
                    Value::String(s) => s.chars().count(),
                    _ => return Err(anyhow!("len unsupported type")),
                };
                Ok(Value::Number(Number::from(len as u64)))
            }
            "str" => {
                if args.len() != 1 {
                    return Err(anyhow!("str expects 1 argument"));
                }
                let text = match &args[0] {
                    Value::String(s) => s.clone(),
                    Value::Number(n) => n.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::Null => "None".to_string(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                };
                Ok(Value::String(text))
            }
            other => Err(anyhow!("unsupported call '{}'", other)),
        },
        proto::expr::Kind::Attribute(attr) => {
            let base = eval_expr(
                attr.value.as_ref().context("attribute call missing base")?,
                ctx,
            )?;
            match attr.attr.as_str() {
                "get" => {
                    if args.is_empty() || args.len() > 2 {
                        return Err(anyhow!("dict.get expects 1 or 2 arguments"));
                    }
                    let default = args.get(1).cloned().unwrap_or(Value::Null);
                    let key = match &args[0] {
                        Value::String(s) => s.clone(),
                        _ => return Ok(default),
                    };
                    match base {
                        Value::Object(map) => Ok(map.get(&key).cloned().unwrap_or(default)),
                        _ => Err(anyhow!("get requires object base")),
                    }
                }
                other => Err(anyhow!("unsupported attribute call '{}'", other)),
            }
        }
        _ => Err(anyhow!("only simple function names supported")),
    }
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i != 0
            } else if let Some(f) = n.as_f64() {
                f != 0.0
            } else {
                true
            }
        }
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(id: &str) -> proto::Expr {
        proto::Expr {
            kind: Some(proto::expr::Kind::Name(proto::Name { id: id.to_string() })),
        }
    }

    fn int(i: i64) -> proto::Expr {
        proto::Expr {
            kind: Some(proto::expr::Kind::Constant(proto::Constant {
                value: Some(proto::constant::Value::IntValue(i)),
            })),
        }
    }

    #[test]
    fn evals_binop_add_numbers() {
        let expr = proto::Expr {
            kind: Some(proto::expr::Kind::BinOp(Box::new(proto::BinOp {
                left: Some(Box::new(int(2))),
                right: Some(Box::new(int(3))),
                op: proto::BinOpKind::Add as i32,
            }))),
        };
        let ctx = EvalContext::new();
        let result = eval_expr(&expr, &ctx).unwrap();
        assert_eq!(result, Value::Number(Number::from(5)));
    }

    #[test]
    fn evals_compare_chain() {
        let expr = proto::Expr {
            kind: Some(proto::expr::Kind::Compare(Box::new(proto::Compare {
                left: Some(Box::new(int(1))),
                ops: vec![proto::CmpOpKind::Lt as i32, proto::CmpOpKind::Lt as i32],
                comparators: vec![int(2), int(3)],
            }))),
        };
        let ctx = EvalContext::new();
        let result = eval_expr(&expr, &ctx).unwrap();
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn supports_len_call() {
        let expr = proto::Expr {
            kind: Some(proto::expr::Kind::Call(Box::new(proto::Call {
                func: Some(Box::new(name("len"))),
                args: vec![proto::Expr {
                    kind: Some(proto::expr::Kind::List(proto::List {
                        elts: vec![int(1), int(2)],
                    })),
                }],
                keywords: Vec::new(),
            }))),
        };
        let ctx = EvalContext::new();
        let result = eval_expr(&expr, &ctx).unwrap();
        assert_eq!(result, Value::Number(Number::from(2)));
    }

    #[test]
    fn assigns_to_context() {
        let mut ctx = EvalContext::new();
        let stmt = proto::Stmt {
            kind: Some(proto::stmt::Kind::Assign(proto::Assign {
                targets: vec![name("x")],
                value: Some(int(10)),
            })),
        };
        eval_stmt(&stmt, &mut ctx).unwrap();
        assert_eq!(ctx.get("x"), Some(&Value::Number(Number::from(10))));
    }
}
