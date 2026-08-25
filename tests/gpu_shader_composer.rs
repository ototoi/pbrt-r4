#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::webgpu::shader::build_shader_set;
use pbrt_r4::gpu::webgpu::shader::ShaderStageId;
use pbrt_r4::gpu::webgpu::AccelerationMode;

#[test]
fn built_in_shader_sources_are_deterministic_and_mode_specific() {
    let hardware = build_shader_set(AccelerationMode::HardwareRayQuery).unwrap();
    let hardware_again = build_shader_set(AccelerationMode::HardwareRayQuery).unwrap();
    let software = build_shader_set(AccelerationMode::SoftwareBvh).unwrap();

    assert_eq!(hardware, hardware_again);
    assert_ne!(hardware.source, software.source);
    assert!(hardware.source.contains("enable wgpu_ray_query;"));
    assert!(!software.source.contains("enable wgpu_ray_query;"));
    assert!(hardware.source.contains("BEGIN pbrt-r4 shader fragment"));
    assert_eq!(
        hardware
            .stage(ShaderStageId::LegacyRender)
            .unwrap()
            .entry_point,
        "main"
    );
    assert_eq!(
        software
            .stage(ShaderStageId::LegacyRender)
            .unwrap()
            .entry_point,
        "main"
    );
    assert_eq!(
        hardware
            .stage(ShaderStageId::GenerateCamera)
            .unwrap()
            .entry_point,
        "generate_camera"
    );
    assert_eq!(
        hardware
            .stage(ShaderStageId::IntersectClosest)
            .unwrap()
            .entry_point,
        "intersect_closest"
    );
    assert_eq!(
        hardware
            .stage(ShaderStageId::ShadeDiffusePoint)
            .unwrap()
            .entry_point,
        "shade_diffuse_point"
    );
    assert_eq!(
        hardware
            .stage(ShaderStageId::IntersectShadow)
            .unwrap()
            .entry_point,
        "intersect_shadow"
    );
    assert_eq!(
        hardware
            .stage(ShaderStageId::FinishBounce)
            .unwrap()
            .entry_point,
        "finish_bounce"
    );
    assert_eq!(
        hardware
            .stage(ShaderStageId::UpdateFilm)
            .unwrap()
            .entry_point,
        "update_film"
    );
    assert_eq!(
        hardware
            .stage(ShaderStageId::AdvanceSample)
            .unwrap()
            .entry_point,
        "advance_sample"
    );
    assert!(software.stage(ShaderStageId::GenerateCamera).is_none());
}
