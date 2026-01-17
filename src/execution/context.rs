use std::collections::{HashMap, HashSet};

use anyhow::Result;

use crate::dag::DAG;
use crate::db::WorkflowInstanceId;
use crate::value::WorkflowValue;

#[tonic::async_trait]
pub trait ExecutionContext {
    fn dag(&self) -> &DAG;
    fn instance_id(&self) -> WorkflowInstanceId;

    async fn initial_scope(&self) -> Result<HashMap<String, WorkflowValue>>;

    async fn load_inbox(
        &self,
        node_ids: &HashSet<String>,
    ) -> Result<HashMap<String, HashMap<String, WorkflowValue>>>;

    async fn read_inbox(&self, node_id: &str) -> Result<HashMap<String, WorkflowValue>>;

    async fn read_spread_inbox(&self, node_id: &str) -> Result<Vec<(i32, WorkflowValue)>>;
}
