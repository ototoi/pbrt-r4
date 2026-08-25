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
            .stage(ShaderStageId::PrepareCameraRays)
            .unwrap()
            .entry_point,
        "prepare_camera_rays"
    );
    assert_eq!(
        hardware
            .stage(ShaderStageId::GenerateCameraRays)
            .unwrap()
            .entry_point,
        "generate_camera_rays"
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
            .stage(ShaderStageId::EvaluateMaterial)
            .unwrap()
            .entry_point,
        "evaluate_material"
    );
    for (stage, entry_point) in [
        (ShaderStageId::ClassifyIntersection, "classify_intersection"),
        (
            ShaderStageId::EvaluateSurfaceInteraction,
            "evaluate_surface_interaction",
        ),
        (ShaderStageId::RegisterBxdf, "register_bxdf"),
        (ShaderStageId::CountBxdf, "count_bxdf"),
        (ShaderStageId::PartitionBxdf, "partition_bxdf"),
        (
            ShaderStageId::SampleDirectLighting,
            "sample_direct_lighting",
        ),
        (
            ShaderStageId::GenerateIndirectRays,
            "generate_indirect_rays",
        ),
        (ShaderStageId::HandleEscapedRays, "handle_escaped_rays"),
        (
            ShaderStageId::HandleEmissiveIntersection,
            "handle_emissive_intersection",
        ),
        (ShaderStageId::PrepareNextBounce, "prepare_next_bounce"),
    ] {
        assert_eq!(hardware.stage(stage).unwrap().entry_point, entry_point);
    }
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
    assert!(software.stage(ShaderStageId::GenerateCameraRays).is_none());
}
