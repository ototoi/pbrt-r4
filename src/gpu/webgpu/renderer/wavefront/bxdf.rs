use super::super::super::shader::ShaderStageId;
use super::super::Renderer;
use super::dispatch_stage;

pub fn evaluate_material<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(renderer, pass, ShaderStageId::EvaluateMaterial, workgroups);
}

pub fn register_bxdf<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(renderer, pass, ShaderStageId::RegisterBxdf, workgroups);
}

pub fn count_bxdf<'a>(renderer: &'a Renderer, pass: &mut wgpu::ComputePass<'a>, workgroups: u32) {
    dispatch_stage(renderer, pass, ShaderStageId::CountBxdf, workgroups);
}

pub fn partition_bxdf<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(renderer, pass, ShaderStageId::PartitionBxdf, workgroups);
}
