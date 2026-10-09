pub struct ComputeStageSpec {
    pub id: ComputeStageId,
    pub label: &'static str,
    pub entry_point: &'static str,
    pub source: &'static str,
}

macro_rules! define_compute_stages {
    ($( $id:ident => ($label:literal, $entry_point:literal, $source:expr) ),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub enum ComputeStageId {
            $( $id, )+
        }

        pub const COMPUTE_STAGES: &[ComputeStageSpec] = &[
            $(
                ComputeStageSpec {
                    id: ComputeStageId::$id,
                    label: $label,
                    entry_point: $entry_point,
                    source: $source,
                },
            )+
        ];
    };
}

define_compute_stages! {
    PrepareSubsurfaceExit => ("prepare_subsurface_exit", "prepare_subsurface_exit", include_str!("shaders/prepare_subsurface_exit.wgsl")),
    ScatterSubsurfaceExit => ("scatter_subsurface_exit", "scatter_subsurface_exit", include_str!("shaders/scatter_subsurface_exit.wgsl")),
    PrepareSample => ("pbrt-r4 prepare sample", "prepare_sample", include_str!("shaders/prepare_sample.wgsl")),
    GeneratePrimaryRays => ("pbrt-r4 generate primary rays", "generate_primary_rays", include_str!("shaders/generate_primary_rays.wgsl")),
    ResetShadowQueue => ("pbrt-r4 reset shadow queue", "reset_shadow_queue", include_str!("shaders/reset_shadow_queue.wgsl")),
    ResetClassificationQueues => ("pbrt-r4 reset classification queues", "reset_classification_queues", include_str!("shaders/reset_classification_queues.wgsl")),
    IntersectPrimaryRays => ("pbrt-r4 intersect primary rays", "intersect_primary_rays", include_str!("shaders/intersect_primary_rays.wgsl")),
    SampleMedium => ("pbrt-r4 sample medium", "sample_medium", include_str!("shaders/sample_medium.wgsl")),
    InitializeMediumSegments => ("pbrt-r4 initialize medium segments", "initialize_medium_segments", include_str!("shaders/initialize_medium_segments.wgsl")),
    InitializeShadowSegments => ("pbrt-r4 initialize shadow segments", "initialize_shadow_segments", include_str!("shaders/initialize_shadow_segments.wgsl")),
    HandleEscaped => ("pbrt-r4 handle escaped rays", "handle_escaped", include_str!("shaders/handle_escaped.wgsl")),
    ShadeSurface => ("pbrt-r4 shade surface", "shade_surface", include_str!("shaders/shade_surface.wgsl")),
    HandleEmissive => ("pbrt-r4 handle emissive", "handle_emissive", include_str!("shaders/handle_emissive.wgsl")),
    PrepareQueueDispatch => ("pbrt-r4 prepare queue dispatch", "prepare_queue_dispatch", include_str!("shaders/prepare_queue_dispatch.wgsl")),
    EvaluateTextures => ("pbrt-r4 evaluate textures", "evaluate_textures", include_str!("shaders/evaluate_textures.wgsl")),
    EvaluateAttributes => ("pbrt-r4 evaluate attributes", "evaluate_attributes", include_str!("shaders/evaluate_attributes.wgsl")),
    ClassifySurfaceScatter => ("pbrt-r4 classify surface scatter", "classify_surface_scatter", include_str!("shaders/classify_surface_scatter.wgsl")),
    SampleDirectLight => ("pbrt-r4 sample direct light", "sample_direct_light", include_str!("shaders/sample_direct_light.wgsl")),
    ScatterMedium => ("pbrt-r4 scatter homogeneous medium", "scatter_medium", include_str!("shaders/scatter_medium.wgsl")),
    ScatterDiffuse => ("pbrt-r4 scatter diffuse", "scatter_diffuse", include_str!("shaders/scatter_diffuse.wgsl")),
    ScatterDiffuseTransmission => ("pbrt-r4 scatter diffuse transmission", "scatter_diffuse_transmission", include_str!("shaders/scatter_diffuse_transmission.wgsl")),
    ScatterConductor => ("pbrt-r4 scatter conductor", "scatter_conductor", include_str!("shaders/scatter_conductor.wgsl")),
    ScatterDielectric => ("pbrt-r4 scatter dielectric", "scatter_dielectric", include_str!("shaders/scatter_dielectric.wgsl")),
    ScatterThinDielectric => ("pbrt-r4 scatter thin dielectric", "scatter_thin_dielectric", include_str!("shaders/scatter_thin_dielectric.wgsl")),
    ScatterMeasured => ("pbrt-r4 scatter measured", "scatter_measured", include_str!("shaders/scatter_measured.wgsl")),
    ScatterCoated => ("pbrt-r4 scatter coated", "scatter_coated", include_str!("shaders/scatter_coated.wgsl")),
    IntersectShadow => ("pbrt-r4 intersect shadow", "intersect_shadow", include_str!("shaders/intersect_shadow.wgsl")),
    SwapRayQueues => ("pbrt-r4 swap ray queues", "swap_ray_queues", include_str!("shaders/swap_ray_queues.wgsl")),
    ResetNextRayQueue => ("pbrt-r4 reset next ray queue", "reset_next_ray_queue", include_str!("shaders/reset_next_ray_queue.wgsl")),
    AccumulateSample => ("pbrt-r4 accumulate sample", "accumulate_sample", include_str!("shaders/accumulate_sample.wgsl")),
}
