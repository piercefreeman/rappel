//! Protocol buffer message types.
//!
//! Generated from proto/messages.proto by tonic-build.

pub mod rappel {
    pub mod messages {
        tonic::include_proto!("rappel.messages");
    }
}

pub use rappel::messages::*;
