use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::pipeline::Pipeline;
use pbrt_r4::gpu::webgpu::stages::RequiredLimits;

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn texture_material_pipeline_compiles() {
    let _ = env_logger::builder().is_test(true).try_init();
    let required = RequiredLimits {
        storage_buffers_per_shader_stage: 30,
        uniform_buffers_per_shader_stage: 5,
        bind_groups: 2,
    };
    let context = Context::new(required, 1, 1).unwrap();
    Pipeline::new(&context.device, 1, 1).unwrap();
}
