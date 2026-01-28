use tracing::info;

use crate::{ir_printer, messages::ast as ir_ast};

pub(crate) fn log_workflow_ir(workflow_name: &str, ir_hash: &str, program: &ir_ast::Program) {
    let ir_str = ir_printer::print_program(program);
    info!(
        workflow_name = %workflow_name,
        ir_hash = %ir_hash,
        "Registered workflow IR:\n{}",
        ir_str
    );
}
