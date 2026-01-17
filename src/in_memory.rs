//! In-memory workflow execution for local broker gRPC streaming.
//!
//! This module executes workflow IR in memory and produces ActionDispatch
//! messages plus a final workflow result payload, mirroring the production
//! readiness model without requiring a database.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use anyhow::{Context, Result};
use prost::Message;
use tracing::warn;
use uuid::Uuid;

use crate::completion::{
    CompletionPlan, InlineContext, ReadinessIncrement, analyze_subgraph, execute_inline_subgraph,
};
use crate::dag::{DAG, EdgeType, convert_to_dag};
use crate::dag_state::DAGHelper;
use crate::db::{BackoffKind, WorkflowInstanceId};
use crate::ir_validation::validate_program;
use crate::messages::{ast as ir_ast, proto};
use crate::parser::ast;
use crate::value::WorkflowValue;

#[derive(Debug)]
pub struct ExecutionStep {
    pub dispatches: Vec<proto::ActionDispatch>,
    pub completed_payload: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
struct ActionState {
    node_id: String,
    module_name: String,
    action_name: String,
    dispatch_payload: Vec<u8>,
    timeout_seconds: u32,
    max_retries: u32,
    backoff_kind: BackoffKind,
    backoff_base_delay_ms: u32,
    attempt_number: u32,
}

#[derive(Debug)]
enum ReadyNode {
    Action(ReadinessIncrement),
    Barrier(String),
    Sleep(ReadinessIncrement),
}

pub struct InMemoryWorkflowExecutor {
    dag: DAG,
    instance_id: WorkflowInstanceId,
    initial_scope: HashMap<String, WorkflowValue>,
    inbox: HashMap<String, HashMap<String, WorkflowValue>>,
    spread_inbox: HashMap<String, BTreeMap<i32, WorkflowValue>>,
    readiness_counts: HashMap<String, i32>,
    readiness_required: HashMap<String, i32>,
    action_states: HashMap<String, ActionState>,
    next_sequence: u32,
}

impl InMemoryWorkflowExecutor {
    pub fn from_registration(registration: proto::WorkflowRegistration) -> Result<Self> {
        let program = ir_ast::Program::decode(&registration.ir[..]).context("invalid IR")?;
        validate_program(&program)
            .map_err(|err| anyhow::anyhow!(err))
            .context("invalid IR")?;

        let dag_program =
            ast::Program::decode(&registration.ir[..]).context("invalid IR for DAG conversion")?;
        let dag = convert_to_dag(&dag_program)
            .map_err(|err| anyhow::anyhow!(err))
            .context("failed to convert IR to DAG")?;

        let initial_scope = registration
            .initial_context
            .as_ref()
            .map(workflow_arguments_to_scope)
            .unwrap_or_default();

        Ok(Self {
            dag,
            instance_id: WorkflowInstanceId(Uuid::new_v4()),
            initial_scope,
            inbox: HashMap::new(),
            spread_inbox: HashMap::new(),
            readiness_counts: HashMap::new(),
            readiness_required: HashMap::new(),
            action_states: HashMap::new(),
            next_sequence: 0,
        })
    }

    pub async fn start(&mut self) -> Result<ExecutionStep> {
        let helper = DAGHelper::new(&self.dag);
        let function_names = helper.get_function_names();

        if function_names.is_empty() {
            return Ok(ExecutionStep {
                dispatches: Vec::new(),
                completed_payload: None,
            });
        }

        let entry_fn = function_names
            .iter()
            .find(|&&name| name == "main")
            .or_else(|| function_names.iter().find(|&&name| !name.starts_with("__")))
            .or(function_names.first())
            .copied()
            .context("no valid entry function found")?;

        let input_node_id = helper
            .find_input_node(entry_fn)
            .context("input node not found")?
            .id
            .clone();
        drop(helper);

        let (_scope, inbox_writes) = seed_scope_and_inbox(
            &self.initial_scope,
            &self.dag,
            &input_node_id,
            self.instance_id,
        );
        self.apply_inbox_writes(inbox_writes);

        let plan = self.build_completion_plan(&input_node_id, WorkflowValue::Null, None)?;

        self.apply_plan(plan).await
    }

    pub async fn handle_action_result(
        &mut self,
        result: proto::ActionResult,
    ) -> Result<ExecutionStep> {
        let action_id = result.action_id.clone();
        let state = match self.action_states.remove(&action_id) {
            Some(state) => state,
            None => {
                warn!(action_id = %action_id, "received action result for unknown action");
                return Ok(ExecutionStep {
                    dispatches: Vec::new(),
                    completed_payload: None,
                });
            }
        };

        if result.success {
            let result_value = parse_action_payload(&result.payload, "result");
            let (base_node_id, mut spread_index) = parse_spread_node_id(&state.node_id);
            if spread_index.is_none() {
                spread_index = parallel_list_index(base_node_id, &self.dag);
            }
            self.handle_completed_node(base_node_id, result_value, spread_index)
                .await
        } else {
            if state.attempt_number < state.max_retries {
                let mut next_state = state.clone();
                next_state.attempt_number += 1;
                self.maybe_backoff(&next_state).await;
                let dispatch = self.dispatch_action(&action_id, &next_state)?;
                self.action_states.insert(action_id, next_state);
                return Ok(ExecutionStep {
                    dispatches: vec![dispatch],
                    completed_payload: None,
                });
            }

            let payload = result.payload.map(|p| p.encode_to_vec());
            let fallback = proto::WorkflowArguments {
                arguments: Vec::new(),
            }
            .encode_to_vec();
            Ok(ExecutionStep {
                dispatches: Vec::new(),
                completed_payload: Some(payload.unwrap_or(fallback)),
            })
        }
    }

    async fn handle_completed_node(
        &mut self,
        node_id: &str,
        result: WorkflowValue,
        spread_index: Option<usize>,
    ) -> Result<ExecutionStep> {
        let plan = self.build_completion_plan(node_id, result, spread_index)?;
        self.apply_plan(plan).await
    }

    fn build_completion_plan(
        &self,
        node_id: &str,
        result: WorkflowValue,
        spread_index: Option<usize>,
    ) -> Result<CompletionPlan> {
        let helper = DAGHelper::new(&self.dag);
        let subgraph = analyze_subgraph(node_id, &self.dag, &helper);
        let existing_inbox = self.collect_existing_inbox(&subgraph.all_node_ids);
        let ctx = InlineContext {
            initial_scope: &self.initial_scope,
            existing_inbox: &existing_inbox,
            spread_index,
        };
        execute_inline_subgraph(node_id, result, ctx, &subgraph, &self.dag, self.instance_id)
            .context("failed to execute inline subgraph")
    }

    async fn apply_plan(&mut self, plan: CompletionPlan) -> Result<ExecutionStep> {
        let mut dispatches: Vec<proto::ActionDispatch> = Vec::new();
        let mut plan_queue: VecDeque<CompletionPlan> = VecDeque::new();
        plan_queue.push_back(plan);

        while let Some(plan) = plan_queue.pop_front() {
            self.apply_inbox_writes(plan.inbox_writes);

            for node_id in plan.readiness_resets {
                self.readiness_counts.insert(node_id, 0);
            }
            for init in plan.readiness_inits {
                self.readiness_required
                    .insert(init.node_id.clone(), init.required_count);
                self.readiness_counts.entry(init.node_id).or_insert(0);
            }

            if let Some(completion) = plan.instance_completion {
                return Ok(ExecutionStep {
                    dispatches: Vec::new(),
                    completed_payload: Some(completion.result_payload),
                });
            }

            let mut ready_nodes: Vec<ReadyNode> = Vec::new();

            for increment in plan.readiness_increments {
                self.readiness_required
                    .insert(increment.node_id.clone(), increment.required_count);

                if increment.required_count == 1 && !increment.is_aggregator {
                    ready_nodes.push(Self::ready_from_increment(increment));
                    continue;
                }

                let counter = self
                    .readiness_counts
                    .entry(increment.node_id.clone())
                    .or_insert(0);
                *counter += 1;
                if *counter >= increment.required_count {
                    ready_nodes.push(Self::ready_from_increment(increment));
                }
            }

            for barrier_id in plan.barrier_enqueues {
                ready_nodes.push(ReadyNode::Barrier(barrier_id));
            }

            for node in ready_nodes {
                match node {
                    ReadyNode::Action(increment) => {
                        let action_id = Uuid::new_v4().to_string();
                        let state = ActionState {
                            node_id: increment.node_id.clone(),
                            module_name: increment.module_name.unwrap_or_default(),
                            action_name: increment.action_name.unwrap_or_default(),
                            dispatch_payload: increment.dispatch_payload.unwrap_or_default(),
                            timeout_seconds: increment.timeout_seconds as u32,
                            max_retries: increment.max_retries as u32,
                            backoff_kind: increment.backoff_kind,
                            backoff_base_delay_ms: increment.backoff_base_delay_ms as u32,
                            attempt_number: 0,
                        };
                        let dispatch = self.dispatch_action(&action_id, &state)?;
                        self.action_states.insert(action_id, state);
                        dispatches.push(dispatch);
                    }
                    ReadyNode::Barrier(node_id) => {
                        let next_plan = self.build_barrier_plan(&node_id)?;
                        plan_queue.push_back(next_plan);
                    }
                    ReadyNode::Sleep(increment) => {
                        self.sleep_from_increment(&increment).await;
                        let next_plan = self.build_completion_plan(
                            &increment.node_id,
                            WorkflowValue::Null,
                            None,
                        )?;
                        plan_queue.push_back(next_plan);
                    }
                }
            }
        }

        Ok(ExecutionStep {
            dispatches,
            completed_payload: None,
        })
    }

    fn build_barrier_plan(&self, node_id: &str) -> Result<CompletionPlan> {
        let aggregated = self.aggregate_spread_results(node_id);
        self.build_completion_plan(node_id, aggregated, None)
    }

    async fn sleep_from_increment(&self, increment: &ReadinessIncrement) {
        let duration_secs = increment
            .dispatch_payload
            .as_ref()
            .and_then(|payload| serde_json::from_slice::<serde_json::Value>(payload).ok())
            .and_then(|value| value.get("duration").cloned())
            .and_then(|value| value.as_f64().or_else(|| value.as_i64().map(|v| v as f64)))
            .unwrap_or(0.0);

        if duration_secs > 0.0 {
            let duration_ms = (duration_secs * 1000.0) as u64;
            tokio::time::sleep(std::time::Duration::from_millis(duration_ms)).await;
        }
    }

    async fn maybe_backoff(&self, state: &ActionState) {
        let delay_ms = match state.backoff_kind {
            BackoffKind::None => 0,
            BackoffKind::Linear => state.backoff_base_delay_ms * (state.attempt_number + 1),
            BackoffKind::Exponential => state
                .backoff_base_delay_ms
                .saturating_mul(2_u32.pow(state.attempt_number)),
        };

        if delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms as u64)).await;
        }
    }

    fn dispatch_action(
        &mut self,
        action_id: &str,
        state: &ActionState,
    ) -> Result<proto::ActionDispatch> {
        let kwargs = json_bytes_to_workflow_args(&state.dispatch_payload);
        let dispatch = proto::ActionDispatch {
            action_id: action_id.to_string(),
            instance_id: self.instance_id.0.to_string(),
            sequence: self.next_sequence,
            action_name: state.action_name.clone(),
            module_name: state.module_name.clone(),
            kwargs: Some(kwargs),
            timeout_seconds: Some(state.timeout_seconds),
            max_retries: Some(state.max_retries),
            attempt_number: Some(state.attempt_number),
            dispatch_token: None,
        };
        self.next_sequence = self.next_sequence.saturating_add(1);
        Ok(dispatch)
    }

    fn apply_inbox_writes(&mut self, inbox_writes: Vec<crate::completion::InboxWrite>) {
        for write in inbox_writes {
            if let Some(spread_index) = write.spread_index {
                let entry = self
                    .spread_inbox
                    .entry(write.target_node_id.clone())
                    .or_default();
                entry.insert(spread_index, write.value.clone());
                continue;
            }

            let entry = self.inbox.entry(write.target_node_id.clone()).or_default();
            entry.insert(write.variable_name.clone(), write.value.clone());
        }
    }

    fn collect_existing_inbox(
        &self,
        node_ids: &HashSet<String>,
    ) -> HashMap<String, HashMap<String, WorkflowValue>> {
        node_ids
            .iter()
            .filter_map(|node_id| {
                self.inbox
                    .get(node_id)
                    .map(|vars| (node_id.clone(), vars.clone()))
            })
            .collect()
    }

    fn aggregate_spread_results(&self, node_id: &str) -> WorkflowValue {
        let aggregated = self
            .spread_inbox
            .get(node_id)
            .map(|entries| entries.values().cloned().collect::<Vec<WorkflowValue>>())
            .unwrap_or_default();

        WorkflowValue::List(aggregated)
    }

    fn ready_from_increment(increment: ReadinessIncrement) -> ReadyNode {
        match increment.node_type {
            crate::completion::NodeType::Action => ReadyNode::Action(increment),
            crate::completion::NodeType::Barrier => ReadyNode::Barrier(increment.node_id),
            crate::completion::NodeType::Sleep => ReadyNode::Sleep(increment),
        }
    }
}

fn workflow_arguments_to_scope(args: &proto::WorkflowArguments) -> HashMap<String, WorkflowValue> {
    args.arguments
        .iter()
        .filter_map(|arg| {
            arg.value
                .as_ref()
                .map(|value| (arg.key.clone(), WorkflowValue::from_proto(value)))
        })
        .collect()
}

fn parse_action_payload(payload: &Option<proto::WorkflowArguments>, key: &str) -> WorkflowValue {
    payload
        .as_ref()
        .and_then(|args| {
            args.arguments
                .iter()
                .find(|arg| arg.key == key)
                .and_then(|arg| arg.value.as_ref())
        })
        .map(WorkflowValue::from_proto)
        .unwrap_or(WorkflowValue::Null)
}

fn json_bytes_to_workflow_args(payload: &[u8]) -> proto::WorkflowArguments {
    if payload.is_empty() {
        return proto::WorkflowArguments { arguments: vec![] };
    }

    let json: serde_json::Value = match serde_json::from_slice(payload) {
        Ok(v) => v,
        Err(e) => {
            warn!("Failed to parse dispatch payload as JSON: {}", e);
            return proto::WorkflowArguments { arguments: vec![] };
        }
    };

    match json {
        serde_json::Value::Object(obj) => {
            let arguments: Vec<proto::WorkflowArgument> = obj
                .iter()
                .map(|(k, v)| proto::WorkflowArgument {
                    key: k.clone(),
                    value: Some(WorkflowValue::from_json(v).to_proto()),
                })
                .collect();
            proto::WorkflowArguments { arguments }
        }
        _ => {
            warn!("dispatch_payload is not a JSON object, expected kwargs");
            proto::WorkflowArguments { arguments: vec![] }
        }
    }
}

fn parse_spread_node_id(node_id: &str) -> (&str, Option<usize>) {
    if let Some(bracket_pos) = node_id.rfind('[')
        && node_id.ends_with(']')
    {
        let base = &node_id[..bracket_pos];
        let idx_str = &node_id[bracket_pos + 1..node_id.len() - 1];
        if let Ok(idx) = idx_str.parse::<usize>() {
            return (base, Some(idx));
        }
    }
    (node_id, None)
}

fn parallel_list_index(node_id: &str, dag: &DAG) -> Option<usize> {
    let node = dag.nodes.get(node_id)?;
    let agg_id = node.aggregates_to.as_ref()?;
    let agg_node = dag.nodes.get(agg_id)?;
    let target_count = agg_node.targets.as_ref().map(|t| t.len()).unwrap_or(0);
    if target_count != 1 {
        return None;
    }

    dag.edges
        .iter()
        .filter(|edge| edge.edge_type == EdgeType::StateMachine)
        .find_map(|edge| {
            if edge.target != node_id {
                return None;
            }
            let condition = edge.condition.as_deref()?;
            let idx_str = condition.strip_prefix("parallel:")?;
            idx_str.parse::<usize>().ok()
        })
}

fn collect_inbox_writes_for_node_with_spread(
    source_node_id: &str,
    variable_name: &str,
    value: &WorkflowValue,
    dag: &DAG,
    instance_id: WorkflowInstanceId,
    spread_index: Option<usize>,
    inbox_writes: &mut Vec<crate::completion::InboxWrite>,
) {
    for edge in dag.edges.iter() {
        if edge.source == source_node_id
            && edge.edge_type == EdgeType::DataFlow
            && edge.variable.as_deref() == Some(variable_name)
        {
            inbox_writes.push(crate::completion::InboxWrite {
                instance_id,
                target_node_id: edge.target.clone(),
                variable_name: variable_name.to_string(),
                value: value.clone(),
                source_node_id: source_node_id.to_string(),
                spread_index: spread_index.map(|i| i as i32),
            });
        }
    }
}

fn seed_scope_and_inbox(
    initial_inputs: &HashMap<String, WorkflowValue>,
    dag: &DAG,
    source_node_id: &str,
    instance_id: WorkflowInstanceId,
) -> (
    HashMap<String, WorkflowValue>,
    Vec<crate::completion::InboxWrite>,
) {
    let mut inbox_writes = Vec::new();
    for (var_name, value) in initial_inputs {
        collect_inbox_writes_for_node_with_spread(
            source_node_id,
            var_name,
            value,
            dag,
            instance_id,
            None,
            &mut inbox_writes,
        );
    }
    (initial_inputs.clone(), inbox_writes)
}
