#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::webgpu::context::Context;
use pbrt_r4::gpu::webgpu::pipeline::StagePipeline;
use pbrt_r4::gpu::webgpu::shader::required_limits_for_sources_with_nanovdb;
use pbrt_r4::gpu::webgpu::stage::COMPUTE_STAGES;
use pbrt_r4::gpu::webgpu::stages::canonical_wavefront_bindings;

const SOURCE: &str = "@compute @workgroup_size(1) fn main() {}";

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

fn create_on_worker(
    context: &Context,
    entry_point: &'static str,
) -> Result<StagePipeline, pbrt_r4::util::error::PbrtError> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        assert!(rayon::current_thread_index().is_some());
        StagePipeline::create(
            &context.device,
            "pbrt-r4 broken stage",
            entry_point,
            SOURCE.to_string(),
            &canonical_wavefront_bindings(),
            1,
            1,
            1,
            None,
        )
    })
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn validation_errors_on_a_worker_thread_name_the_stage() {
    let context = context();
    assert!(create_on_worker(&context, "main").is_ok());
    let error = create_on_worker(&context, "missing")
        .err()
        .expect("a missing entry point must fail")
        .to_string();
    assert!(error.contains("pbrt-r4 broken stage"), "{error}");
}
