use super::super::super::shader::ShaderStageId;
use super::super::Renderer;
use super::dispatch_stage;

pub fn intersect_closest<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(renderer, pass, ShaderStageId::IntersectClosest, workgroups);
}

pub fn classify_intersection<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(
        renderer,
        pass,
        ShaderStageId::ClassifyIntersection,
        workgroups,
    );
}

pub fn evaluate_surface_interaction<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(
        renderer,
        pass,
        ShaderStageId::EvaluateSurfaceInteraction,
        workgroups,
    );
}
