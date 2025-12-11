//! Tests for action timeout and retry handling.
//!
//! These tests verify the process_timed_out_actions and requeue_failed_actions
//! database operations work correctly.

use std::env;

use anyhow::Result;
use chrono::{Duration, Utc};
use serial_test::serial;
use sqlx::Row;
use uuid::Uuid;

use rappel::{BackoffKind, Database, NewAction, WorkflowInstanceId, WorkflowVersionId};

/// Helper to create a test database connection.
async fn setup_db() -> Option<Database> {
    let database_url = match env::var("DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            eprintln!("skipping test: DATABASE_URL not set");
            return None;
        }
    };

    let db = Database::connect(&database_url).await.ok()?;
    cleanup_database(&db).await.ok()?;
    Some(db)
}

/// Clean up all tables before each test.
async fn cleanup_database(db: &Database) -> Result<()> {
    sqlx::query(
        "TRUNCATE action_queue, instance_context, loop_state, node_inputs, node_readiness, workflow_instances, workflow_versions CASCADE",
    )
    .execute(db.pool())
    .await?;
    Ok(())
}

/// Helper to create a workflow version and instance for testing.
async fn create_test_instance(db: &Database) -> Result<(WorkflowVersionId, WorkflowInstanceId)> {
    let version_id = db
        .upsert_workflow_version("test_workflow", "hash123", b"proto", false)
        .await?;

    let instance_id = db
        .create_instance("test_workflow", version_id, None)
        .await?;

    Ok((version_id, instance_id))
}

/// Helper to insert an action directly with specific parameters for testing.
async fn insert_test_action(
    db: &Database,
    instance_id: WorkflowInstanceId,
    status: &str,
    attempt_number: i32,
    timeout_retry_limit: i32,
    deadline_at: Option<chrono::DateTime<Utc>>,
    backoff_kind: &str,
    backoff_base_delay_ms: i32,
) -> Result<Uuid> {
    let row = sqlx::query(
        r#"
        INSERT INTO action_queue (
            instance_id, action_seq, module_name, action_name,
            dispatch_payload, status, attempt_number, timeout_retry_limit,
            deadline_at, backoff_kind, backoff_base_delay_ms,
            delivery_token, dispatched_at
        )
        VALUES ($1, 0, 'test_module', 'test_action', $2, $3, $4, $5, $6, $7, $8, $9, NOW())
        RETURNING id
        "#,
    )
    .bind(instance_id.0)
    .bind(Vec::<u8>::new())
    .bind(status)
    .bind(attempt_number)
    .bind(timeout_retry_limit)
    .bind(deadline_at)
    .bind(backoff_kind)
    .bind(backoff_base_delay_ms)
    .bind(Uuid::new_v4()) // delivery_token
    .fetch_one(db.pool())
    .await?;

    Ok(row.get("id"))
}

/// Helper to get action status and attempt_number.
async fn get_action_state(db: &Database, action_id: Uuid) -> Result<(String, i32)> {
    let row = sqlx::query("SELECT status, attempt_number FROM action_queue WHERE id = $1")
        .bind(action_id)
        .fetch_one(db.pool())
        .await?;

    Ok((row.get("status"), row.get("attempt_number")))
}

/// Helper to get action scheduled_at.
async fn get_action_scheduled_at(
    db: &Database,
    action_id: Uuid,
) -> Result<chrono::DateTime<Utc>> {
    let row = sqlx::query("SELECT scheduled_at FROM action_queue WHERE id = $1")
        .bind(action_id)
        .fetch_one(db.pool())
        .await?;

    Ok(row.get("scheduled_at"))
}

// =============================================================================
// process_timed_out_actions Tests
// =============================================================================

#[tokio::test]
#[serial]
async fn test_process_timed_out_actions_requeues_with_retries_remaining() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    // Create a dispatched action with deadline in the past and retries remaining
    let past_deadline = Utc::now() - Duration::seconds(60);
    let action_id = insert_test_action(
        &db,
        instance_id,
        "dispatched",
        0,               // attempt_number = 0
        3,               // timeout_retry_limit = 3 (so 0 < 3, retries remaining)
        Some(past_deadline),
        "exponential",
        1000,
    )
    .await?;

    // Process timed out actions
    let (requeued, permanently_failed) = db.process_timed_out_actions(100).await?;

    assert_eq!(requeued, 1, "should have requeued 1 action");
    assert_eq!(permanently_failed, 0, "should have 0 permanently failed");

    // Verify action state
    let (status, attempt_number) = get_action_state(&db, action_id).await?;
    assert_eq!(status, "queued", "action should be requeued");
    assert_eq!(attempt_number, 1, "attempt_number should be incremented");

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_process_timed_out_actions_permanently_fails_when_retries_exhausted() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    // Create a dispatched action with deadline in the past and NO retries remaining
    let past_deadline = Utc::now() - Duration::seconds(60);
    let action_id = insert_test_action(
        &db,
        instance_id,
        "dispatched",
        3,               // attempt_number = 3
        3,               // timeout_retry_limit = 3 (so 3 >= 3, no retries remaining)
        Some(past_deadline),
        "exponential",
        1000,
    )
    .await?;

    // Process timed out actions
    let (requeued, permanently_failed) = db.process_timed_out_actions(100).await?;

    assert_eq!(requeued, 0, "should have 0 requeued");
    assert_eq!(permanently_failed, 1, "should have 1 permanently failed");

    // Verify action state
    let (status, attempt_number) = get_action_state(&db, action_id).await?;
    assert_eq!(status, "timed_out", "action should be permanently timed_out");
    assert_eq!(attempt_number, 3, "attempt_number should NOT be incremented");

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_process_timed_out_actions_ignores_non_overdue_actions() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    // Create a dispatched action with deadline in the FUTURE
    let future_deadline = Utc::now() + Duration::seconds(300);
    let action_id = insert_test_action(
        &db,
        instance_id,
        "dispatched",
        0,
        3,
        Some(future_deadline),
        "exponential",
        1000,
    )
    .await?;

    // Process timed out actions
    let (requeued, permanently_failed) = db.process_timed_out_actions(100).await?;

    assert_eq!(requeued, 0, "should have 0 requeued");
    assert_eq!(permanently_failed, 0, "should have 0 permanently failed");

    // Verify action state unchanged
    let (status, attempt_number) = get_action_state(&db, action_id).await?;
    assert_eq!(status, "dispatched", "action should still be dispatched");
    assert_eq!(attempt_number, 0, "attempt_number should be unchanged");

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_process_timed_out_actions_ignores_non_dispatched_actions() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    // Create a queued action (not dispatched) with past deadline
    let past_deadline = Utc::now() - Duration::seconds(60);
    let action_id = insert_test_action(
        &db,
        instance_id,
        "queued", // Not dispatched!
        0,
        3,
        Some(past_deadline),
        "exponential",
        1000,
    )
    .await?;

    // Process timed out actions
    let (requeued, permanently_failed) = db.process_timed_out_actions(100).await?;

    assert_eq!(requeued, 0, "should have 0 requeued");
    assert_eq!(permanently_failed, 0, "should have 0 permanently failed");

    // Verify action state unchanged
    let (status, _) = get_action_state(&db, action_id).await?;
    assert_eq!(status, "queued", "action should still be queued");

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_process_timed_out_actions_applies_exponential_backoff() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    // Create a dispatched action with exponential backoff
    let past_deadline = Utc::now() - Duration::seconds(60);
    let action_id = insert_test_action(
        &db,
        instance_id,
        "dispatched",
        1,               // attempt_number = 1 (second attempt)
        3,               // timeout_retry_limit = 3
        Some(past_deadline),
        "exponential",
        1000,            // base delay = 1000ms
    )
    .await?;

    // Process timed out actions
    let (requeued, _) = db.process_timed_out_actions(100).await?;
    assert_eq!(requeued, 1);

    let after_scheduled = get_action_scheduled_at(&db, action_id).await?;

    // With exponential backoff: base_delay * 2^attempt_number = 1000 * 2^1 = 2000ms
    // The scheduled_at should be at least 2000ms in the future from NOW()
    let actual_delay = after_scheduled - Utc::now();

    // Allow some tolerance (should be close to 2000ms, but check it's at least 1500ms)
    assert!(
        actual_delay > Duration::milliseconds(1500),
        "backoff delay should be at least 1500ms, got {:?}",
        actual_delay
    );

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_process_timed_out_actions_applies_linear_backoff() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    // Create a dispatched action with linear backoff
    let past_deadline = Utc::now() - Duration::seconds(60);
    let action_id = insert_test_action(
        &db,
        instance_id,
        "dispatched",
        1,               // attempt_number = 1
        3,               // timeout_retry_limit = 3
        Some(past_deadline),
        "linear",
        1000,            // base delay = 1000ms
    )
    .await?;

    // Process timed out actions
    let (requeued, _) = db.process_timed_out_actions(100).await?;
    assert_eq!(requeued, 1);

    let after_scheduled = get_action_scheduled_at(&db, action_id).await?;

    // With linear backoff: base_delay * (attempt_number + 1) = 1000 * (1 + 1) = 2000ms
    let actual_delay = after_scheduled - Utc::now();

    // Allow some tolerance
    assert!(
        actual_delay > Duration::milliseconds(1500),
        "backoff delay should be at least 1500ms, got {:?}",
        actual_delay
    );

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_process_timed_out_actions_handles_no_backoff() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    // Create a dispatched action with no backoff
    let past_deadline = Utc::now() - Duration::seconds(60);
    let action_id = insert_test_action(
        &db,
        instance_id,
        "dispatched",
        0,
        3,
        Some(past_deadline),
        "none",          // No backoff
        1000,
    )
    .await?;

    // Process timed out actions
    let (requeued, _) = db.process_timed_out_actions(100).await?;
    assert_eq!(requeued, 1);

    let after_scheduled = get_action_scheduled_at(&db, action_id).await?;

    // With no backoff, scheduled_at should be very close to NOW()
    let actual_delay = after_scheduled - Utc::now();

    assert!(
        actual_delay < Duration::seconds(1),
        "with no backoff, delay should be minimal, got {:?}",
        actual_delay
    );

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_process_timed_out_actions_clears_delivery_token() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    let past_deadline = Utc::now() - Duration::seconds(60);
    let action_id = insert_test_action(
        &db,
        instance_id,
        "dispatched",
        0,
        3,
        Some(past_deadline),
        "exponential",
        1000,
    )
    .await?;

    // Verify delivery_token was set initially
    let row = sqlx::query("SELECT delivery_token FROM action_queue WHERE id = $1")
        .bind(action_id)
        .fetch_one(db.pool())
        .await?;
    let initial_token: Option<Uuid> = row.get("delivery_token");
    assert!(initial_token.is_some(), "delivery_token should be set initially");

    // Process timed out actions
    db.process_timed_out_actions(100).await?;

    // Verify delivery_token is cleared
    let row = sqlx::query("SELECT delivery_token, deadline_at FROM action_queue WHERE id = $1")
        .bind(action_id)
        .fetch_one(db.pool())
        .await?;
    let cleared_token: Option<Uuid> = row.get("delivery_token");
    let deadline: Option<chrono::DateTime<Utc>> = row.get("deadline_at");

    assert!(cleared_token.is_none(), "delivery_token should be cleared");
    assert!(deadline.is_none(), "deadline_at should be cleared");

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_process_timed_out_actions_mixed_results() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;
    let past_deadline = Utc::now() - Duration::seconds(60);

    // Action 1: Should be requeued (retries remaining)
    let action1_id = insert_test_action(
        &db,
        instance_id,
        "dispatched",
        0,  // attempt_number
        3,  // timeout_retry_limit
        Some(past_deadline),
        "exponential",
        1000,
    )
    .await?;

    // We need to insert another action with a different action_seq
    sqlx::query(
        r#"
        INSERT INTO action_queue (
            instance_id, action_seq, module_name, action_name,
            dispatch_payload, status, attempt_number, timeout_retry_limit,
            deadline_at, backoff_kind, backoff_base_delay_ms,
            delivery_token, dispatched_at
        )
        VALUES ($1, 1, 'test_module', 'test_action', $2, $3, $4, $5, $6, $7, $8, $9, NOW())
        RETURNING id
        "#,
    )
    .bind(instance_id.0)
    .bind(Vec::<u8>::new())
    .bind("dispatched")
    .bind(3i32)  // attempt_number = 3 (exhausted)
    .bind(3i32)  // timeout_retry_limit = 3
    .bind(Some(past_deadline))
    .bind("exponential")
    .bind(1000i32)
    .bind(Uuid::new_v4())
    .fetch_one(db.pool())
    .await?;

    // Process timed out actions
    let (requeued, permanently_failed) = db.process_timed_out_actions(100).await?;

    assert_eq!(requeued, 1, "should have 1 requeued");
    assert_eq!(permanently_failed, 1, "should have 1 permanently failed");

    // Verify action 1 was requeued
    let (status1, _) = get_action_state(&db, action1_id).await?;
    assert_eq!(status1, "queued");

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_process_timed_out_actions_respects_limit() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;
    let past_deadline = Utc::now() - Duration::seconds(60);

    // Create 5 overdue actions
    for i in 0..5 {
        sqlx::query(
            r#"
            INSERT INTO action_queue (
                instance_id, action_seq, module_name, action_name,
                dispatch_payload, status, attempt_number, timeout_retry_limit,
                deadline_at, backoff_kind, backoff_base_delay_ms,
                delivery_token, dispatched_at
            )
            VALUES ($1, $2, 'test_module', 'test_action', $3, 'dispatched', 0, 3, $4, 'none', 0, $5, NOW())
            "#,
        )
        .bind(instance_id.0)
        .bind(i)
        .bind(Vec::<u8>::new())
        .bind(Some(past_deadline))
        .bind(Uuid::new_v4())
        .execute(db.pool())
        .await?;
    }

    // Process with limit of 2
    let (requeued, permanently_failed) = db.process_timed_out_actions(2).await?;

    assert_eq!(requeued + permanently_failed, 2, "should only process 2 actions");

    // Count remaining dispatched actions
    let row = sqlx::query("SELECT COUNT(*) as count FROM action_queue WHERE status = 'dispatched'")
        .fetch_one(db.pool())
        .await?;
    let remaining: i64 = row.get("count");
    assert_eq!(remaining, 3, "should have 3 actions still dispatched");

    Ok(())
}

// =============================================================================
// requeue_failed_actions Tests
// =============================================================================

#[tokio::test]
#[serial]
async fn test_requeue_failed_actions_requeues_with_retries_remaining() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    // Create a failed action with retries remaining
    sqlx::query(
        r#"
        INSERT INTO action_queue (
            instance_id, action_seq, module_name, action_name,
            dispatch_payload, status, retry_kind, attempt_number, max_retries,
            backoff_kind, backoff_base_delay_ms
        )
        VALUES ($1, 0, 'test_module', 'test_action', $2, 'failed', 'failure', 0, 3, 'exponential', 1000)
        RETURNING id
        "#,
    )
    .bind(instance_id.0)
    .bind(Vec::<u8>::new())
    .fetch_one(db.pool())
    .await?;

    // Requeue failed actions
    let requeued = db.requeue_failed_actions(100).await?;

    assert_eq!(requeued, 1, "should have requeued 1 action");

    // Verify it was requeued
    let row = sqlx::query("SELECT status, attempt_number FROM action_queue WHERE instance_id = $1")
        .bind(instance_id.0)
        .fetch_one(db.pool())
        .await?;
    let status: String = row.get("status");
    let attempt: i32 = row.get("attempt_number");

    assert_eq!(status, "queued");
    assert_eq!(attempt, 1);

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_requeue_failed_actions_ignores_exhausted_retries() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    // Create a failed action with NO retries remaining
    sqlx::query(
        r#"
        INSERT INTO action_queue (
            instance_id, action_seq, module_name, action_name,
            dispatch_payload, status, retry_kind, attempt_number, max_retries,
            backoff_kind, backoff_base_delay_ms
        )
        VALUES ($1, 0, 'test_module', 'test_action', $2, 'failed', 'failure', 3, 3, 'exponential', 1000)
        "#,
    )
    .bind(instance_id.0)
    .bind(Vec::<u8>::new())
    .execute(db.pool())
    .await?;

    // Requeue failed actions
    let requeued = db.requeue_failed_actions(100).await?;

    assert_eq!(requeued, 0, "should not have requeued any actions");

    // Verify it's still failed
    let row = sqlx::query("SELECT status FROM action_queue WHERE instance_id = $1")
        .bind(instance_id.0)
        .fetch_one(db.pool())
        .await?;
    let status: String = row.get("status");
    assert_eq!(status, "failed");

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_requeue_failed_actions_ignores_timeout_retry_kind() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    // Create a failed action with timeout retry_kind (should be ignored by requeue_failed_actions)
    sqlx::query(
        r#"
        INSERT INTO action_queue (
            instance_id, action_seq, module_name, action_name,
            dispatch_payload, status, retry_kind, attempt_number, max_retries,
            backoff_kind, backoff_base_delay_ms
        )
        VALUES ($1, 0, 'test_module', 'test_action', $2, 'failed', 'timeout', 0, 3, 'exponential', 1000)
        "#,
    )
    .bind(instance_id.0)
    .bind(Vec::<u8>::new())
    .execute(db.pool())
    .await?;

    // Requeue failed actions
    let requeued = db.requeue_failed_actions(100).await?;

    assert_eq!(requeued, 0, "should not requeue timeout-type failures");

    Ok(())
}

// =============================================================================
// Integration-like tests
// =============================================================================

#[tokio::test]
#[serial]
async fn test_full_timeout_retry_cycle() -> Result<()> {
    let Some(db) = setup_db().await else {
        return Ok(());
    };

    let (_, instance_id) = create_test_instance(&db).await?;

    // Enqueue an action
    let action = NewAction {
        instance_id,
        module_name: "test_module".to_string(),
        action_name: "test_action".to_string(),
        dispatch_payload: vec![],
        timeout_seconds: 1, // 1 second timeout
        max_retries: 3,
        backoff_kind: BackoffKind::None,
        backoff_base_delay_ms: 0,
        node_id: Some("test_node".to_string()),
        node_type: Some("action".to_string()),
    };
    db.enqueue_action(action).await?;

    // Dispatch the action (sets deadline_at)
    let actions = db.dispatch_actions(10).await?;
    assert_eq!(actions.len(), 1);
    let action_id = actions[0].id;

    // Verify it's dispatched with a deadline
    let row = sqlx::query("SELECT status, deadline_at FROM action_queue WHERE id = $1")
        .bind(action_id)
        .fetch_one(db.pool())
        .await?;
    let status: String = row.get("status");
    let deadline: Option<chrono::DateTime<Utc>> = row.get("deadline_at");
    assert_eq!(status, "dispatched");
    assert!(deadline.is_some());

    // Manually set deadline to past to simulate timeout
    sqlx::query("UPDATE action_queue SET deadline_at = NOW() - INTERVAL '1 minute' WHERE id = $1")
        .bind(action_id)
        .execute(db.pool())
        .await?;

    // Process timed out actions - should requeue
    let (requeued, permanently_failed) = db.process_timed_out_actions(100).await?;
    assert_eq!(requeued, 1);
    assert_eq!(permanently_failed, 0);

    // Verify it's requeued with incremented attempt
    let (status, attempt) = get_action_state(&db, action_id).await?;
    assert_eq!(status, "queued");
    assert_eq!(attempt, 1);

    // Dispatch again
    let actions = db.dispatch_actions(10).await?;
    assert_eq!(actions.len(), 1);

    // Simulate timeout again (repeat for all retries)
    for expected_attempt in 2..=3 {
        sqlx::query("UPDATE action_queue SET deadline_at = NOW() - INTERVAL '1 minute' WHERE id = $1")
            .bind(action_id)
            .execute(db.pool())
            .await?;

        let (requeued, permanently_failed) = db.process_timed_out_actions(100).await?;

        if expected_attempt <= 3 {
            // Still has retries
            assert_eq!(requeued, 1);
            assert_eq!(permanently_failed, 0);

            let (status, attempt) = get_action_state(&db, action_id).await?;
            assert_eq!(status, "queued");
            assert_eq!(attempt, expected_attempt);

            // Dispatch again
            db.dispatch_actions(10).await?;
        }
    }

    // Final timeout - should be permanently failed
    sqlx::query("UPDATE action_queue SET deadline_at = NOW() - INTERVAL '1 minute' WHERE id = $1")
        .bind(action_id)
        .execute(db.pool())
        .await?;

    let (requeued, permanently_failed) = db.process_timed_out_actions(100).await?;
    assert_eq!(requeued, 0);
    assert_eq!(permanently_failed, 1);

    // Verify it's permanently timed_out
    let (status, attempt) = get_action_state(&db, action_id).await?;
    assert_eq!(status, "timed_out");
    assert_eq!(attempt, 3); // Not incremented for permanent failure

    Ok(())
}
