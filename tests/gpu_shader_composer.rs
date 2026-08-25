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
}
