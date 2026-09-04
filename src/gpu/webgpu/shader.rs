const COMMON_SHADER: &str = include_str!("shaders/common.wgsl");

pub fn create_module(device: &wgpu::Device, label: &str, stage_source: &str) -> wgpu::ShaderModule {
    let descriptor = wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(compose_source(stage_source))),
    };
    device.create_shader_module(descriptor)
}

#[doc(hidden)]
pub fn compose_source(stage_source: &str) -> String {
    let mut source = String::with_capacity(COMMON_SHADER.len() + stage_source.len() + 1);
    source.push_str(COMMON_SHADER);
    source.push('\n');
    source.push_str(stage_source);
    source
}
