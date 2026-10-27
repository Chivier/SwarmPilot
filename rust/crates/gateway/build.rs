//! Generates the control-channel client from `proto/` at the repo root.
//! protox compiles the .proto files in Rust, so no protoc is needed.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../proto");
    let file = root.join("swarmpilot/gateway/v1/control.proto");
    println!("cargo:rerun-if-changed={}", file.display());
    let fds = protox::compile([&file], [&root])?;
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_fds(fds)?;
    Ok(())
}
