//! Rappel - worker pool infrastructure plus the core IR/runtime port.

pub mod messages;
pub mod observability;
pub mod rappel_core;
pub mod scheduler;
pub mod server_worker;
pub mod webapp;
pub mod workers;

// Worker infrastructure (preserved from the legacy Rust core).
pub use messages::{MessageError, ast as ir_ast, proto, workflow_argument_value_to_json};
pub use observability::obs;
pub use scheduler::{
    CreateScheduleParams, ScheduleId, ScheduleType, SchedulerConfig, SchedulerDatabase,
    SchedulerTask, WorkflowSchedule, spawn_scheduler,
};
pub use server_worker::{WorkerBridgeChannels, WorkerBridgeServer};
pub use webapp::{WebappConfig, WebappDatabase, WebappServer};
pub use workers::{
    ActionDispatchPayload, PythonWorker, PythonWorkerConfig, PythonWorkerPool, RemoteWorkerPool,
    RoundTripMetrics,
};
