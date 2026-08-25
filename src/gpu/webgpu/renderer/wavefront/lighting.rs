use super::super::super::shader::ShaderStageId;
use super::super::Renderer;
use super::dispatch_stage;

pub fn sample_direct_lighting<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(
        renderer,
        pass,
        ShaderStageId::SampleDirectLighting,
        workgroups,
    );
}

pub fn generate_indirect_rays<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(
        renderer,
        pass,
        ShaderStageId::GenerateIndirectRays,
        workgroups,
    );
}

pub fn handle_escaped_rays<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(renderer, pass, ShaderStageId::HandleEscapedRays, workgroups);
}

pub fn handle_emissive_intersection<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(
        renderer,
        pass,
        ShaderStageId::HandleEmissiveIntersection,
        workgroups,
    );
}

pub fn intersect_shadow<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(renderer, pass, ShaderStageId::IntersectShadow, workgroups);
}
