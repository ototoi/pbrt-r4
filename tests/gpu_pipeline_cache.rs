#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::pipeline::Pipeline;
use pbrt_r4::gpu::webgpu::shader::required_limits_for_sources_with_nanovdb;
use pbrt_r4::gpu::webgpu::stage::COMPUTE_STAGES;
use pbrt_r4::gpu::webgpu::stages::canonical_wavefront_bindings;

fn context() -> Context {
    let canonical_bindings = canonical_wavefront_bindings();
    let stage_sources = COMPUTE_STAGES
        .iter()
        .map(|stage| stage.source)
        .collect::<Vec<_>>();
    let mut required =
        required_limits_for_sources_with_nanovdb(&canonical_bindings, &stage_sources, false)
            .unwrap();
    required.bind_groups = required.bind_groups.max(2);
    Context::new(required, 1, 1).unwrap()
}

fn create_pipeline(
    context: &Context,
    path: Option<&std::path::Path>,
) -> Result<Pipeline, pbrt_r4::util::error::PbrtError> {
    Pipeline::new_with_geometry_page_count(&context.device, 1, 1, 1, false, false, 1, path)
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn pipeline_creation_writes_and_reuses_the_cache() {
    let context = context();
    if !context
        .device
        .features()
        .contains(wgpu::Features::PIPELINE_CACHE)
    {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.bin");
    create_pipeline(&context, Some(&path)).unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
    create_pipeline(&context, Some(&path)).unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn corrupt_cache_file_is_replaced() {
    let context = context();
    if !context
        .device
        .features()
        .contains(wgpu::Features::PIPELINE_CACHE)
    {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cache.bin");
    let mut state = 0x1234_5678u32;
    let corrupt: Vec<u8> = (0..4096)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect();
    std::fs::write(&path, &corrupt).unwrap();
    create_pipeline(&context, Some(&path)).unwrap();
    assert_ne!(std::fs::read(&path).unwrap(), corrupt);
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn unwritable_cache_dir_does_not_fail_creation() {
    let context = context();
    if !context
        .device
        .features()
        .contains(wgpu::Features::PIPELINE_CACHE)
    {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let blocker = dir.path().join("blocker");
    std::fs::write(&blocker, b"file").unwrap();
    let path = blocker.join("cache.bin");
    create_pipeline(&context, Some(&path)).unwrap();
}
