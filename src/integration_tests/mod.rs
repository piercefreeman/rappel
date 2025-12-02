//! Integration tests using the new Store API.
//!
//! These tests require a PostgreSQL database and must run sequentially
//! to avoid database state conflicts.

mod harness;

use harness::{WorkflowHarness, WorkflowHarnessConfig};
use serial_test::serial;

const INTEGRATION_MODULE: &str = include_str!("fixtures/integration_module.py");
const INTEGRATION_COMPLEX: &str = include_str!("fixtures/integration_complex.py");
const INTEGRATION_LOOP: &str = include_str!("fixtures/integration_loop.py");
const INTEGRATION_LOOP_ACCUM: &str = include_str!("fixtures/integration_loop_accum.py");
const INTEGRATION_CONDITIONAL: &str = include_str!("fixtures/integration_conditional.py");
const INTEGRATION_EXCEPTION: &str = include_str!("fixtures/integration_exception.py");

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
#[serial]
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

const INTEGRATION_COMPLEX_ENTRYPOINT: &str = r#"
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(__file__))

from integration_complex import ComplexWorkflow

if __name__ == "__main__":
    asyncio.run(ComplexWorkflow().run())
"#;

/// Test: Complex workflow with gather, loops (from async list comprehensions), and conditionals
#[tokio::test]
#[serial]
async fn test_integration_complex() {
    let config = WorkflowHarnessConfig {
        files: &[("integration_complex.py", INTEGRATION_COMPLEX)],
        entrypoint: INTEGRATION_COMPLEX_ENTRYPOINT,
        workflow_name: "complexworkflow",
        user_module: "integration_complex",
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

    // All should succeed
    for m in &metrics {
        assert!(m.success, "Action {} failed", m.action_id);
    }

    // Check final result
    // Expected: fetch_left=1, fetch_right=3 -> double: 2, 6 -> sum=8 > 6 -> "big"
    // computed: [3, 7] -> "big:3,7"
    let result = harness.stored_result().await.expect("get result failed");
    assert!(result.is_some(), "Expected workflow to have result");

    let result_value = result.unwrap();
    assert_eq!(
        result_value,
        serde_json::json!("big:3,7"),
        "Expected 'big:3,7', got {:?}",
        result_value
    );

    harness.shutdown().await.expect("shutdown failed");
}

const INTEGRATION_LOOP_ENTRYPOINT: &str = r#"
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(__file__))

from integration_loop import LoopWorkflow

if __name__ == "__main__":
    asyncio.run(LoopWorkflow().run())
"#;

/// Test: Loop workflow with preamble (f-string computation before action)
/// seeds = ["alpha", "beta"]
/// Loop: local_value = f"{seed}-local" -> decorate_item -> "seed-local-decorated"
/// outputs = ["alpha-local-decorated", "beta-local-decorated"]
/// result = "alpha-local-decorated,beta-local-decorated"
#[tokio::test]
#[serial]
async fn test_integration_loop() {
    let config = WorkflowHarnessConfig {
        files: &[("integration_loop.py", INTEGRATION_LOOP)],
        entrypoint: INTEGRATION_LOOP_ENTRYPOINT,
        workflow_name: "loopworkflow",
        user_module: "integration_loop",
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
        serde_json::json!("alpha-local-decorated,beta-local-decorated"),
        "Expected 'alpha-local-decorated,beta-local-decorated', got {:?}",
        result_value
    );

    harness.shutdown().await.expect("shutdown failed");
}

const INTEGRATION_LOOP_ACCUM_ENTRYPOINT: &str = r#"
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(__file__))

from integration_loop_accum import LoopAccumWorkflow

if __name__ == "__main__":
    asyncio.run(LoopAccumWorkflow().run())
"#;

/// Test: Loop workflow that reads from accumulator in preamble
/// This tests that the accumulator variable (outputs) is accessible in loop preamble
/// seeds = ["alpha", "beta"]
/// Loop iteration 1: index = len([]) = 0, local_value = "alpha-local-0" -> "alpha-local-0-decorated"
/// Loop iteration 2: index = len([...]) = 1, local_value = "beta-local-1" -> "beta-local-1-decorated"
/// result = "alpha-local-0-decorated,beta-local-1-decorated"
#[tokio::test]
#[serial]
async fn test_integration_loop_accum() {
    let config = WorkflowHarnessConfig {
        files: &[("integration_loop_accum.py", INTEGRATION_LOOP_ACCUM)],
        entrypoint: INTEGRATION_LOOP_ACCUM_ENTRYPOINT,
        workflow_name: "loopaccumworkflow",
        user_module: "integration_loop_accum",
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
        serde_json::json!("alpha-local-0-decorated,beta-local-1-decorated"),
        "Expected 'alpha-local-0-decorated,beta-local-1-decorated', got {:?}",
        result_value
    );

    harness.shutdown().await.expect("shutdown failed");
}

const INTEGRATION_CONDITIONAL_HIGH_ENTRYPOINT: &str = r#"
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(__file__))

from integration_conditional import ConditionalWorkflow

if __name__ == "__main__":
    asyncio.run(ConditionalWorkflow().run(tier="high"))
"#;

/// Test: Conditional workflow taking the "high" branch
/// tier="high" -> get_value returns 100 -> 100 >= 75 -> evaluate_high(100) -> "high:100"
/// Result: "high:high:100"
#[tokio::test]
#[serial]
async fn test_integration_conditional_high() {
    let config = WorkflowHarnessConfig {
        files: &[("integration_conditional.py", INTEGRATION_CONDITIONAL)],
        entrypoint: INTEGRATION_CONDITIONAL_HIGH_ENTRYPOINT,
        workflow_name: "conditionalworkflow",
        user_module: "integration_conditional",
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
        serde_json::json!("high:high:100"),
        "Expected 'high:high:100', got {:?}",
        result_value
    );

    harness.shutdown().await.expect("shutdown failed");
}

const INTEGRATION_EXCEPTION_ENTRYPOINT: &str = r#"
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(__file__))

from integration_exception import ExceptionWorkflow

if __name__ == "__main__":
    asyncio.run(ExceptionWorkflow().run())
"#;

/// Test: Exception handling workflow
/// try:
///     number = provide_value() -> 10
///     explode(10) -> raises ValueError("boom:10")
/// except ValueError:
///     result = cleanup("fallback") -> "handled:fallback"
/// return result -> "handled:fallback"
///
/// NOTE: This test is currently ignored because exception handling in the
/// scheduler is not yet implemented. The DAG structure for try/except is
/// created correctly, but the scheduler's handle_failure() method doesn't
/// yet route exceptions to handlers.
#[tokio::test]
#[serial]
#[ignore = "Exception handling not yet implemented in scheduler"]
async fn test_integration_exception() {
    let config = WorkflowHarnessConfig {
        files: &[("integration_exception.py", INTEGRATION_EXCEPTION)],
        entrypoint: INTEGRATION_EXCEPTION_ENTRYPOINT,
        workflow_name: "exceptionworkflow",
        user_module: "integration_exception",
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

    // Some actions may fail (explode), but handler should succeed
    // We expect: provide_value (success), explode (fail), cleanup (success)
    let successes: Vec<_> = metrics.iter().filter(|m| m.success).collect();
    let failures: Vec<_> = metrics.iter().filter(|m| !m.success).collect();

    assert!(
        successes.len() >= 2,
        "Expected at least 2 successful actions (provide_value, cleanup), got {}",
        successes.len()
    );
    assert!(
        failures.len() >= 1,
        "Expected at least 1 failed action (explode), got {}",
        failures.len()
    );

    // Check final result
    let result = harness.stored_result().await.expect("get result failed");
    assert!(result.is_some(), "Expected workflow to have result");

    let result_value = result.unwrap();
    assert_eq!(
        result_value,
        serde_json::json!("handled:fallback"),
        "Expected 'handled:fallback', got {:?}",
        result_value
    );

    harness.shutdown().await.expect("shutdown failed");
}
