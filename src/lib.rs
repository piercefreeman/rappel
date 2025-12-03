//! Rappel - A workflow execution engine with durable Python workers
//!
//! This crate provides the core infrastructure for executing workflow actions
//! in Python worker processes. The key components are:
//!
//! ## Worker Infrastructure
//!
//! - [`WorkerBridgeServer`]: gRPC server that workers connect to
//! - [`PythonWorkerPool`]: Pool of Python worker processes for action execution
//! - [`PythonWorker`]: Individual worker process management
//!
//! ## IR Language
//!
//! - [`lexer`]: Tokenizer for the Rappel IR language with indentation handling
//! - [`parser`]: Recursive descent parser producing proto-based AST

pub mod lexer;
pub mod messages;
pub mod parser;
pub mod server_worker;
pub mod worker;

// Worker infrastructure
pub use messages::{MessageError, proto};
pub use server_worker::{WorkerBridgeChannels, WorkerBridgeServer};
pub use worker::{
    ActionDispatchPayload, PythonWorker, PythonWorkerConfig, PythonWorkerPool, RoundTripMetrics,
};

// IR language
pub use lexer::{lex, Lexer, LexerError, Span, SpannedToken, Token};
pub use parser::{ast, parse, ParseError, Parser};
