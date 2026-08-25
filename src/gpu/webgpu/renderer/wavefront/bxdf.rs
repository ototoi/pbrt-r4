use super::super::Renderer;
use super::dispatch_stage;
use crate::gpu::webgpu::shader::ShaderStageId;

pub fn dispatch<'a>(renderer: &'a Renderer, pass: &mut wgpu::ComputePass<'a>, workgroups: u32) {
    for stage in [
        ShaderStageId::EvaluateMaterial,
        ShaderStageId::RegisterBxdf,
        ShaderStageId::CountBxdf,
        ShaderStageId::PartitionBxdf,
    ] {
        dispatch_stage(renderer, pass, stage, workgroups);
    }
}
