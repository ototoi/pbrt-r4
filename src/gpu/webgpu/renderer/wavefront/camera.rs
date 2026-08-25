use super::super::Renderer;
use super::{dispatch_control_stage, dispatch_stage};
use crate::gpu::webgpu::shader::ShaderStageId;

pub fn dispatch<'a>(renderer: &'a Renderer, pass: &mut wgpu::ComputePass<'a>, workgroups: u32) {
    dispatch_control_stage(renderer, pass, ShaderStageId::PrepareCameraRays);
    dispatch_stage(
        renderer,
        pass,
        ShaderStageId::GenerateCameraRays,
        workgroups,
    );
}
