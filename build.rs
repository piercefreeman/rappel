fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/messages.proto");
    println!("cargo:rerun-if-changed=proto/ast.proto");

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(
            &["proto/messages.proto", "proto/ast.proto"],
            &["proto"],
        )?;

    Ok(())
}
