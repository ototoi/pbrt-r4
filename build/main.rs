mod mipmap;
mod sensor;
mod spectrum;
mod utils;

// Entry point for the package build script. All generated data modules are
// reached from here so Cargo's `rerun-if-changed=build` covers their sources.

fn main() {
    // The build script is split across modules under `build/`. Register the
    // directory at the entry point so Cargo notices changes to those modules
    // before it can reuse stale rerun-if-changed output.
    println!("cargo:rerun-if-changed=build");
    spectrum::build();
    sensor::build();
    mipmap::build();
}
