use base64::Engine;
use prost::Message;
use rappel::{
    EdgeType, WorkflowInstanceId,
    ast::Program,
    completion::{InlineContext, analyze_subgraph, execute_inline_subgraph},
    convert_to_dag,
    dag_state::DAGHelper,
};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use uuid::Uuid;

fn main() {
    // Base64-encoded Program proto for benchmark workflow (captured from DB)
    let b64 = "Co8BCg1fX2lmX3RoZW5fMV9fEgAacgpwCmQKBnJlc3VsdBJaUk4KFHByb2Nlc3Nfc3BlY2lhbF9oYXNoEiIKCGFuYWx5c2lzEhYSCgoIYW5hbHlzaXNaCAgUEDAYFCA4IhJiZW5jaG1hcmtfd29ya2Zsb3daCAgUEAwYFCA5UggIFBAMGBQgOSIICBMQCBgWIDgKjgEKDV9faWZfZWxzZV8yX18SABpxCm8KYwoGcmVzdWx0EllSTQoTcHJvY2Vzc19ub3JtYWxfaGFzaBIiCghhbmFseXNpcxIWEgoKCGFuYWx5c2lzWggIFhAvGBYgNyISYmVuY2htYXJrX3dvcmtmbG93WggIFhAMGBYgOFIICBYQDBgWIDgiCAgWEAwYFiA4CuQDCg5fX2Zvcl9ib2R5XzNfXxIiCgpoYXNoX3ZhbHVlCglwcm9jZXNzZWQSCXByb2Nlc3NlZBqjAwpuCmIKCGFuYWx5c2lzElZSSgoMYW5hbHl6ZV9oYXNoEiYKCmhhc2hfdmFsdWUSGBIMCgpoYXNoX3ZhbHVlWggIEBAmGBAgMCISYmVuY2htYXJrX3dvcmtmbG93WggIEBAIGBAgMVIICBAQCBgQIDEKowEylgEKaQo+OjIKFhIKCghhbmFseXNpc1oICBMQCxgTIBMSGAoMGgppc19zcGVjaWFsWggIExAUGBMgIFoICBMQCxgTICESHRIREg8KDV9faWZfdGhlbl8xX18iCAgTEAgYFiA4GggIExAIGBYgOBopCh0SERIPCg1fX2lmX2Vsc2VfMl9fIggIFhAMGBYgOBIICBYQDBgWIDhSCAgTEAgYFiA4CmQKWAoJcHJvY2Vzc2VkEksaPwoXEgsKCXByb2Nlc3NlZFoICBgQCBgYICAQARoiKhYKFBIICgZyZXN1bHRaCAgYEBkYGCAfWggIGBAIGBggIFoICBgQCBgYICBSCAgYEAgYGCAgCiVCGQoXEgsKCXByb2Nlc3NlZFoICA4QBBgYICBSCAgOEAQYGCAgIggIDhAEGBggIAqqBAoDcnVuEhUKB2luZGljZXMKCml0ZXJhdGlvbnMagwQKrwEKogEKBmhhc2hlcxKXAVoICAcQBBgKIAZqigEKFRIJCgdpbmRpY2VzWggICRARGAkgGBIBaRpuChZjb21wdXRlX2hhc2hfZm9yX2luZGV4EhgKBWluZGV4Eg8SAwoBaVoICAgQJRgIICYSJgoKaXRlcmF0aW9ucxIYEgwKCml0ZXJhdGlvbnNaCAgIEDMYCCA9IhJiZW5jaG1hcmtfd29ya2Zsb3dSCAgHEAQYCiAGCiUKGQoJcHJvY2Vzc2VkEgwqAFoICA0QEBgNIBJSCAgNEAQYDSASCpQBKocBCgpoYXNoX3ZhbHVlEhQSCAoGaGFzaGVzWggIDhAWGA4gHBpjCglwcm9jZXNzZWQSTBJKCg5fX2Zvcl9ib2R5XzNfXxocCgpoYXNoX3ZhbHVlEg4SDAoKaGFzaF92YWx1ZRoaCglwcm9jZXNzZWQSDRILCglwcm9jZXNzZWQiCAgOEAQYGCAgUggIDhAEGBggIApsCmAKB3N1bW1hcnkSVVJJCg9jb21iaW5lX3Jlc3VsdHMSIgoHcmVzdWx0cxIXEgsKCXByb2Nlc3NlZFoICBsQJBgbIC0iEmJlbmNobWFya193b3JrZmxvd1oICBsQBBgbIC5SCAgbEAQYGyAuCiNCFwoVEgkKB3N1bW1hcnlaCAgcEAsYHCASUggIHBAEGBwgEiIGCAEYHCAS";

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .expect("failed to decode base64");
    let program = Program::decode(bytes.as_slice()).expect("decode program");
    let dag = convert_to_dag(&program);

    println!("=== Nodes ===");
    let nodes: BTreeMap<_, _> = dag.nodes.iter().collect();
    for (id, node) in nodes.iter() {
        println!(
            "{}: type={} targets={:?} loop_body_of={:?} loop_collection={:?} kwargs={:?}",
            id, node.node_type, node.targets, node.loop_body_of, node.loop_collection, node.kwargs
        );
    }
    if let Some(assign) = dag
        .nodes
        .get("for_body_call_21:assign_13")
        .and_then(|n| n.assign_expr.as_ref())
    {
        println!(
            "assign_13 expr = {}",
            rappel::ast_printer::print_expr(assign)
        );
    }

    println!("\n=== State Machine Edges ===");
    for edge in dag
        .edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::StateMachine)
    {
        println!(
            "{} -> {} (loop_back={} guard={:?})",
            edge.source, edge.target, edge.is_loop_back, edge.guard_string
        );
    }

    println!("\n=== Data Flow Edges ===");
    for edge in dag
        .edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::DataFlow)
    {
        println!(
            "{} -[{:?}]-> {}",
            edge.source,
            edge.variable.as_deref().unwrap_or("?"),
            edge.target
        );
    }

    let helper = DAGHelper::new(&dag);
    let subgraph = analyze_subgraph("aggregator_18", &dag, &helper);
    println!(
        "\nFrontiers from aggregator_18: {:?}",
        subgraph.frontier_nodes
    );

    let ctx = InlineContext {
        initial_scope: &HashMap::new(),
        existing_inbox: &HashMap::new(),
        spread_index: None,
    };
    let plan = execute_inline_subgraph(
        "aggregator_18",
        json!(["dummy"]),
        ctx,
        &subgraph,
        &dag,
        WorkflowInstanceId(Uuid::nil()),
    )
    .expect("execute_inline_subgraph");
    println!(
        "Readiness increments targets: {:?}",
        plan.readiness_increments
            .iter()
            .map(|r| r.node_id.as_str())
            .collect::<Vec<_>>()
    );
}
