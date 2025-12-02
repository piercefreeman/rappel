//! Integration tests using the new Store API.

mod harness;

use harness::{WorkflowHarness, WorkflowHarnessConfig};

const INTEGRATION_MODULE: &str = include_str!("fixtures/integration_module.py");

const INTEGRATION_MODULE_ENTRYPOINT: &str = r#"
import sys
import os
import asyncio

# Add the module path
sys.path.insert(0, os.path.dirname(__file__))

from integration_module import IntegrationWorkflow

if __name__ == "__main__":
    # Running the workflow will register it with the server
    asyncio.run(IntegrationWorkflow().run())
"#;

/// Test: Simple workflow with one action
#[tokio::test]
async fn test_integration_module() {
    let config = WorkflowHarnessConfig {
        files: &[("integration_module.py", INTEGRATION_MODULE)],
        entrypoint: INTEGRATION_MODULE_ENTRYPOINT,
        workflow_name: "integrationworkflow",
        user_module: "integration_module",
        inputs: &[],
    };

    let harness = match WorkflowHarness::new(config).await {
        Ok(Some(h)) => h,
        Ok(None) => {
            eprintln!("Skipping test: database not available");
            return;
        }
        Err(e) => panic!("Failed to create harness: {}", e),
    };

    // Dispatch all actions
    let metrics = harness.dispatch_all().await.expect("dispatch_all failed");

    // Should have executed expected actions
    assert_eq!(
        metrics.len(),
        harness.expected_actions(),
        "Expected {} actions, got {}",
        harness.expected_actions(),
        metrics.len()
    );

    // All should succeed
    for m in &metrics {
        assert!(m.success, "Action {} failed", m.action_id);
    }

    // Check final result
    let result = harness.stored_result().await.expect("get result failed");
    assert!(result.is_some(), "Expected workflow to have result");

    let result_value = result.unwrap();
    assert_eq!(
        result_value,
        serde_json::json!("hello world"),
        "Expected 'hello world', got {:?}",
        result_value
    );

    harness.shutdown().await.expect("shutdown failed");
}
