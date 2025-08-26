fn main() {
    let target = std::env::var("TARGET").unwrap();
    // doesnt feel right
    if target.contains("x86_64") && !target.contains("wasm32") {
        // pkg_config::probe_library("openblas").unwrap();
    }
}