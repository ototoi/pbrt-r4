use std::collections::HashMap;

use super::bssrdf::BSSRDFProbePipeline;
use super::context::Context;
use super::film::Film;
use super::noise::NoiseRuntimeResources;
use super::pipeline::Pipeline;
use super::queue::Queues;
use super::scene::Scene;
use super::stage::ComputeStageId;

mod create;
mod dispatch;
mod render;
mod tiles;

pub struct WavefrontPathIntegrator {
    context: Context,
    scene: Scene,
    camera_buffer: wgpu::Buffer,
    viewport_buffer: wgpu::Buffer,
    material_table_buffer: wgpu::Buffer,
    light_table_buffer: wgpu::Buffer,
    queues: Queues,
    film: Film,
    noise_resources: NoiseRuntimeResources,
    pipeline: Pipeline,
    bind_groups: HashMap<ComputeStageId, [wgpu::BindGroup; 2]>,
    rendered: bool,
    show_progress: bool,
    tile_width: u32,
    tile_height: u32,
    has_interface_only_instances: bool,
    bssrdf_work: wgpu::Buffer,
    _bssrdf_results: wgpu::Buffer,
    bssrdf_probe: Option<(BSSRDFProbePipeline, wgpu::BindGroup)>,
}
