use pbrt_r4::gpu::webgpu::shader_composer::{compose, ShaderModuleSpec};

fn module(id: &str, source: &str, dependencies: &[&str]) -> ShaderModuleSpec {
    ShaderModuleSpec {
        id: id.to_string(),
        source: source.to_string(),
        dependencies: dependencies
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
    }
}

#[test]
fn composer_deduplicates_shared_dependencies_in_stable_order() {
    let result = compose(
        vec![
            module("root", "root", &["left", "right"]),
            module("left", "left", &["common"]),
            module("right", "right", &["common"]),
            module("common", "common", &[]),
        ],
        "root",
    )
    .unwrap();
    assert_eq!(result.modules, ["common", "left", "right", "root"]);
    assert_eq!(result.source.matches("// module: common").count(), 1);
}

#[test]
fn composer_rejects_cycles_and_missing_dependencies() {
    let cycle = compose(
        vec![module("a", "a", &["b"]), module("b", "b", &["a"])],
        "a",
    );
    assert!(cycle.is_err());
    let missing = compose(vec![module("a", "a", &["missing"])], "a");
    assert!(missing.is_err());
}
