#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::webgpu::shader::{build_wavefront_shader_set, ShaderStageId};

#[test]
fn wavefront_shader_is_composed_from_dedicated_fragments() {
    let shader = build_wavefront_shader_set().unwrap();

    assert!(shader.source.contains("shaders/wavefront/abi.wgsl"));
    assert!(shader.source.contains("shaders/wavefront/camera.wgsl"));
    assert!(shader
        .source
        .contains("shaders/wavefront/intersection.wgsl"));
    assert!(shader.source.contains("shaders/wavefront/escaped.wgsl"));
    assert!(shader.source.contains("shaders/wavefront/indirect.wgsl"));
    assert!(shader.source.contains("shaders/wavefront/surface.wgsl"));
    assert!(shader.source.contains("shaders/wavefront/emissive.wgsl"));
    assert!(shader.source.contains("shaders/wavefront/material.wgsl"));
    assert!(shader
        .source
        .contains("shaders/wavefront/direct_lighting.wgsl"));
    assert!(shader.source.contains("shaders/wavefront/shadow.wgsl"));
    assert!(shader.source.contains("shaders/wavefront/film.wgsl"));
    assert!(!shader.source.contains("shaders/entry/"));
    assert!(shader.stage(ShaderStageId::PrepareCameraRays).is_some());
    assert!(shader.stage(ShaderStageId::GenerateCameraRays).is_some());
    assert!(shader.stage(ShaderStageId::IntersectClosest).is_some());
    assert!(shader
        .stage(ShaderStageId::EvaluateSurfaceInteraction)
        .is_some());
    assert!(shader.stage(ShaderStageId::EvaluateMaterial).is_some());
    assert!(shader.stage(ShaderStageId::SampleDirectLighting).is_some());
    assert!(shader.stage(ShaderStageId::IntersectShadow).is_some());
    assert_eq!(shader.stages.len(), 11);
}
