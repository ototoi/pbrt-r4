const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");
const TRIANGLE_SAMPLING_SHADER: &str = include_str!("shaders/triangle_sampling.wgsl");
const LAYERED_SHADER: &str = include_str!("shaders/layered.wgsl");

use super::shader_composer::{compose, ShaderModuleSpec};

pub fn create_module(device: &wgpu::Device, label: &str, stage_source: &str) -> wgpu::ShaderModule {
    let descriptor = wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(compose_source(stage_source))),
    };
    device.create_shader_module(descriptor)
}

#[doc(hidden)]
pub fn compose_source(stage_source: &str) -> String {
    compose(
        vec![
            ShaderModuleSpec {
                id: "common".to_string(),
                source: COMMON_SHADER.to_string(),
                dependencies: Vec::new(),
            },
            ShaderModuleSpec {
                id: "triangle_sampling".to_string(),
                source: TRIANGLE_SAMPLING_SHADER.to_string(),
                dependencies: vec!["common".to_string()],
            },
            ShaderModuleSpec {
                id: "layered".to_string(),
                source: LAYERED_SHADER.to_string(),
                dependencies: vec!["common".to_string()],
            },
            ShaderModuleSpec {
                id: "stage".to_string(),
                source: stage_source.to_string(),
                dependencies: vec!["triangle_sampling".to_string(), "layered".to_string()],
            },
        ],
        "stage",
    )
    .expect("built-in WebGPU shader module graph is valid")
    .source
}
