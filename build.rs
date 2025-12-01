fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/messages.proto");
    println!("cargo:rerun-if-changed=proto/ir.proto");

    // First compile ir.proto to generate rappel.ir types
    tonic_build::configure()
        .build_server(false)
        .build_client(false)
        .compile(&["proto/ir.proto"], &["proto"])?;

    // Then compile messages.proto, using extern_path to reference the ir types
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        // Tell messages.proto where to find rappel.ir types
        .extern_path(".rappel.ir", "crate::ir_parser::proto")
        .compile(&["proto/messages.proto"], &["proto"])?;

    Ok(())
}
