use super::super::Renderer;
use super::dispatch_stage;
use crate::gpu::webgpu::shader::ShaderStageId;

pub fn dispatch<'a>(renderer: &'a Renderer, pass: &mut wgpu::ComputePass<'a>, workgroups: u32) {
    dispatch_stage(renderer, pass, ShaderStageId::IntersectClosest, workgroups);
    dispatch_stage(
        renderer,
        pass,
        ShaderStageId::ClassifyIntersection,
        workgroups,
    );
    dispatch_stage(
        renderer,
        pass,
        ShaderStageId::EvaluateSurfaceInteraction,
        workgroups,
    );
}
