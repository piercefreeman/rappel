//! Server implementation for worker bridges.
//!
//! Provides gRPC services for:
//! - ActionWorkerBridge: Bidirectional streaming for action workers
//! - InstanceWorkerBridge: Bidirectional streaming for instance workers
//! - WorkflowService: HTTP endpoints for workflow management

pub mod action_bridge;
pub mod instance_bridge;
pub mod http;

pub use action_bridge::ActionWorkerBridgeService;
pub use instance_bridge::InstanceWorkerBridgeService;
