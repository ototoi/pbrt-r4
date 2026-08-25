use super::super::super::shader::ShaderStageId;
use super::super::Renderer;
use super::{dispatch_control_stage, dispatch_stage};

pub fn prepare_camera_rays<'a>(renderer: &'a Renderer, pass: &mut wgpu::ComputePass<'a>) {
    dispatch_control_stage(renderer, pass, ShaderStageId::PrepareCameraRays);
}

pub fn generate_camera_rays<'a>(
    renderer: &'a Renderer,
    pass: &mut wgpu::ComputePass<'a>,
    workgroups: u32,
) {
    dispatch_stage(
        renderer,
        pass,
        ShaderStageId::GenerateCameraRays,
        workgroups,
    );
}
