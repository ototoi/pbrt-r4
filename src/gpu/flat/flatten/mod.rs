use super::texture::{compile_texture_library, TextureRootSpec};
use super::{
    build_light_bounds, build_light_bvh, identity_transform, inverse_linear_transform,
    multiply_transform, transform_swaps_handedness, AreaTriangleInput, AttributeKind, AttributeRef,
    Camera, DenseSpectrumBuilder, Film, Geometry, Instance, Light, LightBoundInput,
    LightGeometryKind, LightKind, LightSamplingModel, Material, Output, PrimitiveDistributionMap,
    Scene, Transform, TriangleDistributionEntry, UnsupportedTexturePolicy, Vertex, Viewport,
    INVALID_INDEX,
};
use crate::gpu::node::{
    complete_triangle_attributes, remove_invalid_triangles, Component,
    Integrator as NodeIntegrator, Material as NodeMaterial, NodeRef, Sampler as NodeSampler, Shape,
};
use crate::util::error::PbrtError;
use crate::util::spectrum::Spectrum;

use std::collections::HashMap;
use std::sync::Arc;

use super::geometry::{
    dot3, scale3, transform_point, transform_vector, triangle_area, triangle_geometric_normal,
};

mod material;
use material::material_index;
mod material_attributes;
mod node;
use node::flatten_node_ref;

mod shape;
use shape::geometry_index;

mod lights;
use lights::{append_area_light, area_light_record, flatten_light};

mod scene_settings;
use scene_settings::{
    register_root_component, render_settings, screen_window, viewport_resolution,
};

const MAX_GPU_RENDER_DEPTH: i32 = 32;
const MAX_LAYER_DEPTH: i32 = 32;
const MAX_LAYER_SAMPLES: i32 = 32;

const IDENTITY_LINEAR_TRANSFORM: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

pub fn flatten_node(root: NodeRef) -> Result<Scene, PbrtError> {
    flatten_node_with_material_override(root, None)
}

pub fn flatten_node_with_material_override(
    root: NodeRef,
    material_kind: Option<&str>,
) -> Result<Scene, PbrtError> {
    let mut builder = FlatBuilder::default();
    let mut stack = Vec::new();
    flatten_node_ref(
        &root,
        &identity_transform(),
        &mut builder,
        &mut stack,
        material_kind,
    )?;
    let output = builder
        .output
        .ok_or_else(|| PbrtError::error("No output was found while flattening GPU Node IR."))?;
    let camera = builder
        .camera
        .ok_or_else(|| PbrtError::error("No camera was found while flattening GPU Node IR."))?;
    let viewport = builder
        .viewport
        .ok_or_else(|| PbrtError::error("No film was found while flattening GPU Node IR."))?;
    let film = builder.film.ok_or_else(|| {
        PbrtError::error("No RGB film data was found while flattening GPU Node IR.")
    })?;
    let render_settings = render_settings(&builder.sampler, &builder.integrator)?;
    let light_bounds = build_light_bounds(&builder.light_bound_inputs)?;
    let light_bvh = build_light_bvh(&builder.lights, &light_bounds)?;
    let attribute_refs = builder
        .materials
        .iter()
        .flat_map(|material| material.attributes.iter().cloned())
        .chain(
            builder
                .lights
                .iter()
                .flat_map(|light| light.attributes.iter().cloned()),
        )
        .chain(
            builder
                .infinite_lights
                .iter()
                .flat_map(|light| light.attributes.iter().cloned()),
        )
        .collect();
    let texture_library = compile_texture_library(&builder.texture_root_specs)?;
    let scene = Scene {
        camera,
        viewport,
        film,
        output,
        render_settings,
        light_sampling_models: builder.light_sampling_models,
        light_positions: builder.light_positions,
        triangle_distributions: builder.triangle_distributions,
        lights: builder.lights,
        infinite_lights: builder.infinite_lights,
        light_bounds,
        light_bvh,
        vertices: builder.vertices,
        indices: builder.indices,
        geometries: builder.geometries,
        instances: builder.instances,
        materials: builder.materials,
        attribute_refs,
        scalar_attributes: builder.scalar_attributes,
        texture_library,
        spectrum_attributes: builder.spectrum_table_builder.finish(),
        primitive_distribution_map: PrimitiveDistributionMap {
            offsets: vec![0],
            entries: Vec::new(),
        },
    };
    let mut scene = scene;
    scene.primitive_distribution_map = build_primitive_distribution_map(&scene)?;
    scene.validate_static_views()?;
    Ok(scene)
}

fn push_scalar_attribute(
    builder: &mut FlatBuilder,
    name: &str,
    value: f32,
) -> Result<AttributeRef, PbrtError> {
    let index = u32::try_from(builder.scalar_attributes.len())
        .map_err(|_| PbrtError::error("Flat scalar attribute table exceeds u32."))?;
    builder.scalar_attributes.push(value);
    Ok(AttributeRef {
        kind: AttributeKind::Scalar,
        index,
        name: name.to_string(),
    })
}

fn push_spectrum_attribute(
    builder: &mut FlatBuilder,
    name: &str,
    value: &Spectrum,
) -> Result<AttributeRef, PbrtError> {
    let index = builder.spectrum_table_builder.intern(value)?;
    Ok(AttributeRef {
        kind: AttributeKind::Spectrum,
        index,
        name: name.to_string(),
    })
}

fn validate_layer_limits(name: &str, max_depth: i32, n_samples: i32) -> Result<(), PbrtError> {
    if !(0..=MAX_LAYER_DEPTH).contains(&max_depth) {
        return Err(PbrtError::error(&format!(
            "Material \"{}\" maxdepth {} is outside the GPU layered limit 0..={}.",
            name, max_depth, MAX_LAYER_DEPTH
        )));
    }
    if !(1..=MAX_LAYER_SAMPLES).contains(&n_samples) {
        return Err(PbrtError::error(&format!(
            "Material \"{}\" nsamples {} is outside the GPU layered limit 1..={}.",
            name, n_samples, MAX_LAYER_SAMPLES
        )));
    }
    Ok(())
}

fn build_primitive_distribution_map(scene: &Scene) -> Result<PrimitiveDistributionMap, PbrtError> {
    let area_count = scene
        .lights
        .iter()
        .filter(|light| light.kind == LightKind::Area)
        .count();
    let mut offsets = Vec::with_capacity(area_count + 1);
    let mut entries = Vec::new();
    offsets.push(0);
    let area_lights = scene
        .lights
        .iter()
        .filter(|light| light.kind == LightKind::Area)
        .map(|light| scene.light_sampling_models[light.sampling_model as usize].clone())
        .collect::<Vec<_>>();
    for (area_index, area_light) in area_lights.iter().enumerate() {
        let instance = scene
            .instances
            .get(area_light.geometry_index as usize)
            .ok_or_else(|| {
                PbrtError::error(&format!(
                    "Area light {area_index} references an invalid instance."
                ))
            })?;
        let geometry = scene
            .geometries
            .get(instance.geometry as usize)
            .ok_or_else(|| {
                PbrtError::error(&format!(
                    "Area light {area_index} references an invalid geometry."
                ))
            })?;
        let triangle_count = geometry.index_count / 3;
        let base = entries.len();
        entries.resize(base + triangle_count as usize, INVALID_INDEX);
        let start = usize::try_from(area_light.distribution_offset)
            .map_err(|_| PbrtError::error("Area-light distribution offset exceeds usize."))?;
        let count = usize::try_from(area_light.distribution_count)
            .map_err(|_| PbrtError::error("Area-light distribution count exceeds usize."))?;
        let end = start
            .checked_add(count)
            .ok_or_else(|| PbrtError::error("Area-light distribution range overflowed."))?;
        for (distribution_index, entry) in scene
            .triangle_distributions
            .get(start..end)
            .ok_or_else(|| PbrtError::error("Area-light distribution range is invalid."))?
            .iter()
            .enumerate()
        {
            if entry.primitive >= triangle_count {
                return Err(PbrtError::error(
                    "Area-light distribution primitive is invalid.",
                ));
            }
            entries[base + entry.primitive as usize] = u32::try_from(start + distribution_index)
                .map_err(|_| PbrtError::error("Distribution index exceeds u32."))?;
        }
        offsets.push(u32::try_from(entries.len()).map_err(|_| {
            PbrtError::error("Primitive distribution map exceeds the u32 index range.")
        })?);
    }
    Ok(PrimitiveDistributionMap { offsets, entries })
}

#[derive(Default)]
struct FlatBuilder {
    camera: Option<Camera>,
    viewport: Option<Viewport>,
    film: Option<Film>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    geometries: Vec<Geometry>,
    geometries_by_shape: HashMap<(usize, usize), u32>,
    instances: Vec<Instance>,
    materials: Vec<Material>,
    scalar_attributes: Vec<f32>,
    texture_root_specs: Vec<TextureRootSpec>,
    texture_roots_by_key: HashMap<(usize, u32), u32>,
    spectrum_table_builder: DenseSpectrumBuilder,
    output: Option<Output>,
    source_materials: Vec<Arc<NodeMaterial>>,
    sampler: Option<NodeSampler>,
    integrator: Option<NodeIntegrator>,
    light_sampling_models: Vec<LightSamplingModel>,
    light_positions: Vec<[f32; 3]>,
    triangle_distributions: Vec<TriangleDistributionEntry>,
    lights: Vec<Light>,
    infinite_lights: Vec<Light>,
    light_bound_inputs: Vec<LightBoundInput>,
}
