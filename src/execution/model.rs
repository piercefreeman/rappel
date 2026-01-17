use crate::completion::CompletionPlan;

#[derive(Debug)]
pub struct EnginePlan {
    pub plan: CompletionPlan,
    pub subgraph_us: u64,
    pub inbox_us: u64,
    pub inline_us: u64,
}

#[derive(Debug)]
pub struct EngineStartPlan {
    pub plan: CompletionPlan,
    pub seed_inbox_writes: Vec<crate::completion::InboxWrite>,
    pub initial_scope: std::collections::HashMap<String, crate::value::WorkflowValue>,
}

#[derive(Debug)]
pub enum ExceptionHandlingOutcome {
    Retry,
    Handled {
        plan: Box<CompletionPlan>,
        handler_node_id: String,
    },
    Unhandled,
}
