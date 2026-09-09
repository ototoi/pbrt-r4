use super::{
    build_light_bounds, build_light_bvh, identity_transform, multiply_transform,
    transform_swaps_handedness, AreaTriangleInput, AttributeKind, AttributeRef, Camera,
    DenseSpectrumBuilder, Film, Geometry, Instance, Light, LightBoundInput, LightGeometryKind,
    LightKind, LightSamplingModel, Material, Output, PrimitiveDistributionMap, RenderSettings,
    Scene, TextureNode as FlatTextureNode, Transform, TriangleDistributionEntry,
    UnsupportedTexturePolicy, Vertex, Viewport, INVALID_INDEX,
};
use crate::film::PixelSensor;
use crate::gpu::ir::node::{
    complete_triangle_attributes, remove_invalid_triangles, AreaLight as NodeAreaLight, Component,
    Integrator as NodeIntegrator, Light as NodeLight, Material as NodeMaterial, NodeRef,
    Sampler as NodeSampler, Shape, TextureComponent, TextureKind as NodeTextureKind,
    TriangleMeshShape,
};
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;
use crate::util::spectrum::{spectrum_to_photometric, Spectrum, SpectrumType};

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

const MAX_GPU_RENDER_DEPTH: i32 = 32;
const MAX_LAYER_DEPTH: i32 = 32;
const MAX_LAYER_SAMPLES: i32 = 32;

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
        .collect();
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
        light_bounds,
        light_bvh,
        vertices: builder.vertices,
        indices: builder.indices,
        geometries: builder.geometries,
        instances: builder.instances,
        materials: builder.materials,
        attribute_refs,
        scalar_attributes: builder.scalar_attributes,
        texture_attributes: builder.texture_attributes,
        texture_nodes: builder.texture_nodes,
        texture_child_indices: builder.texture_child_indices,
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

fn register_texture_node(
    node: &Arc<crate::gpu::ir::node::TextureNode>,
    builder: &mut FlatBuilder,
) -> Result<u32, PbrtError> {
    validate_texture_graph(node, 0, &mut HashSet::new())?;
    let key = Arc::as_ptr(node) as usize;
    if let Some(&index) = builder.texture_nodes_by_ptr.get(&key) {
        return Ok(index);
    }
    let index = u32::try_from(builder.texture_nodes.len())
        .map_err(|_| PbrtError::error("Flat texture node table exceeds u32."))?;
    builder.texture_nodes_by_ptr.insert(key, index);
    builder.texture_nodes.push(FlatTextureNode {
        name: node.name.clone(),
        kind: 0,
        implementation: String::new(),
        first_child: 0,
        child_count: 0,
        mipmap: None,
        mapping: crate::gpu::ir::node::Transform::default().matrix,
        swrap_mode: 0,
        twrap_mode: 0,
        filter_mode: 0,
        color_space: 0,
        operation: 0,
        mapping_kind: 0,
        constant_value: [0.0; 4],
    });
    let mut kind = 0;
    let mut implementation = String::new();
    let mut mipmap = None;
    let mut mapping = crate::gpu::ir::node::Transform::default().matrix;
    let mut swrap_mode = 0;
    let mut twrap_mode = 0;
    let mut filter_mode = 0;
    let mut color_space = 0;
    let mut operation = 0;
    let mut mapping_kind = 0;
    let mut constant_value = [0.0; 4];
    for component in &node.components {
        if let TextureComponent::Texture(texture) = component {
            kind = match texture.kind {
                NodeTextureKind::Float => 0,
                NodeTextureKind::Spectrum => 1,
            };
            implementation = texture.name.clone();
            mipmap = texture.mipmap.clone();
            color_space = mipmap
                .as_ref()
                .and_then(|mipmap| mipmap.color_space)
                .map(|space| match space {
                    crate::gpu::ir::node::ColorSpaceId::Srgb => 0,
                    crate::gpu::ir::node::ColorSpaceId::Aces2065 => 1,
                    crate::gpu::ir::node::ColorSpaceId::DciP3 => 2,
                    crate::gpu::ir::node::ColorSpaceId::Rec2020 => 3,
                })
                .unwrap_or(0);
            let wrap = texture.params.get_one_string("wrap", "repeat");
            swrap_mode = sampler_wrap_mode(&texture.params.get_one_string("swrap", &wrap));
            twrap_mode = sampler_wrap_mode(&texture.params.get_one_string("twrap", &wrap));
            filter_mode =
                sampler_filter_mode(&texture.params.get_one_string("filter", "bilinear"))?;
            if texture.name == "constant" {
                operation = 1;
                if texture.kind == NodeTextureKind::Float {
                    constant_value[0] = texture.params.get_one_float("value", 0.0) as f32;
                    constant_value[1] = constant_value[0];
                    constant_value[2] = constant_value[0];
                } else {
                    let rgb = texture
                        .params
                        .get_one_spectrum("value", &Spectrum::from(0.0))
                        .to_rgb();
                    constant_value = [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, 0.0];
                }
                constant_value[3] =
                    (constant_value[0] + constant_value[1] + constant_value[2]) / 3.0;
            } else if texture.name == "imagemap" {
                // ImageTexture's scale/invert are applied after sampling.
                constant_value[0] = texture.params.get_one_float("scale", 1.0) as f32;
                constant_value[1] = if texture.params.get_one_bool("invert", false) {
                    1.0
                } else {
                    0.0
                };
            } else if texture.name == "scale" {
                operation = 2;
                let default_scale = texture.params.get_one_float("value", 1.0);
                constant_value[0] = texture.params.get_one_float("scale", default_scale) as f32;
            } else if texture.name == "mix" {
                operation = 3;
                constant_value[0] = texture.params.get_one_float("amount", 0.5) as f32;
            } else if texture.name == "directionmix" {
                operation = 5;
            } else if texture.name == "fbm" {
                operation = 7;
                constant_value[0] = texture.params.get_one_float("roughness", 0.5) as f32;
                constant_value[1] = texture.params.get_one_int("octaves", 8) as f32;
            } else if texture.name == "wrinkled" {
                operation = 8;
                constant_value[0] = texture.params.get_one_float("roughness", 0.5) as f32;
                constant_value[1] = texture.params.get_one_int("octaves", 8) as f32;
            } else if texture.name == "windy" {
                operation = 9;
            } else if texture.name == "dots" {
                operation = 6;
            } else if texture.name == "bilerp" {
                operation = 10;
            } else if texture.name == "marble" {
                operation = 11;
                constant_value[0] = texture.params.get_one_float("roughness", 0.5) as f32;
                constant_value[1] = texture.params.get_one_int("octaves", 8) as f32;
                constant_value[2] = texture.params.get_one_float("scale", 1.0) as f32;
                constant_value[3] = texture.params.get_one_float("variation", 0.2) as f32;
            } else if texture.name.contains("checkerboard") {
                operation = if texture.name == "checkerboard3d"
                    || texture.params.get_one_int("dimension", 2) == 3
                {
                    12
                } else {
                    4
                };
            }
            builder
                .texture_nodes_by_name
                .insert((texture.kind, node.name.clone()), index);
        }
        if let TextureComponent::Mapping(mapping_component) = component {
            if let crate::gpu::ir::node::TextureMapping::Uv(uv) = mapping_component {
                mapping[0] = uv.uscale;
                mapping[5] = uv.vscale;
                mapping[3] = uv.udelta;
                mapping[7] = uv.vdelta;
            } else if let crate::gpu::ir::node::TextureMapping::PointTransform(transform) =
                mapping_component
            {
                mapping = transform.matrix;
                if operation == 5 {
                    constant_value[0] = transform.matrix[0];
                    constant_value[1] = transform.matrix[1];
                    constant_value[2] = transform.matrix[2];
                }
            } else {
                match mapping_component {
                    crate::gpu::ir::node::TextureMapping::Planar(transform) => {
                        mapping = transform.matrix;
                        mapping_kind = 1;
                    }
                    crate::gpu::ir::node::TextureMapping::Spherical(transform) => {
                        mapping = transform.matrix;
                        mapping_kind = 2;
                    }
                    crate::gpu::ir::node::TextureMapping::Cylindrical(transform) => {
                        mapping = transform.matrix;
                        mapping_kind = 3;
                    }
                    _ => {}
                }
            }
        }
    }
    let mut child_indices = Vec::with_capacity(node.children.len());
    for child in &node.children {
        child_indices.push(register_texture_node(child, builder)?);
    }
    // Child registration is recursive and may append the descendants' child
    // ranges first.  Capture the parent's range only after that recursion so
    // `first_child` points at the indices appended below, not at a descendant
    // range.
    let first_child = u32::try_from(builder.texture_child_indices.len())
        .map_err(|_| PbrtError::error("Flat texture child table exceeds u32."))?;
    builder
        .texture_child_indices
        .extend(child_indices.iter().copied());
    let child_count = u32::try_from(node.children.len())
        .map_err(|_| PbrtError::error("Flat texture child table exceeds u32."))?;
    let valid_child_count = match operation {
        0 | 1 | 7 | 8 | 9 | 11 => child_count == 0,
        2 => child_count == 1,
        3 | 5 => child_count == 2 || (operation == 3 && child_count == 3),
        4 | 6 | 12 => child_count == 2,
        10 => child_count == 4,
        _ => false,
    };
    if !valid_child_count {
        return Err(PbrtError::error(&format!(
            "Texture node \"{}\" has {} children for operation {}.",
            node.name, child_count, operation
        )));
    }
    if kind == 1 && mipmap.is_none() {
        let mut graph_color_space = None;
        for child_index in &child_indices {
            let child = &builder.texture_nodes[*child_index as usize];
            if child.kind != 1 || (child.mipmap.is_none() && child.color_space == 0) {
                continue;
            }
            if let Some(previous) = graph_color_space {
                if previous != child.color_space {
                    return Err(PbrtError::error(&format!(
                        "Spectrum texture node \"{}\" mixes incompatible color spaces.",
                        node.name
                    )));
                }
            } else {
                graph_color_space = Some(child.color_space);
            }
        }
        if let Some(graph_color_space) = graph_color_space {
            color_space = graph_color_space;
        }
    }
    let flat = &mut builder.texture_nodes[index as usize];
    flat.kind = kind;
    flat.implementation = implementation;
    flat.first_child = first_child;
    flat.child_count = child_count;
    flat.mipmap = mipmap;
    flat.mapping = mapping;
    flat.swrap_mode = swrap_mode;
    flat.twrap_mode = twrap_mode;
    flat.filter_mode = filter_mode;
    flat.color_space = color_space;
    flat.operation = operation;
    flat.mapping_kind = mapping_kind;
    flat.constant_value = constant_value;
    Ok(index)
}

const MAX_TEXTURE_GRAPH_DEPTH: usize = 32;

fn validate_texture_graph(
    node: &Arc<crate::gpu::ir::node::TextureNode>,
    depth: usize,
    visiting: &mut HashSet<usize>,
) -> Result<(), PbrtError> {
    if depth >= MAX_TEXTURE_GRAPH_DEPTH {
        return Err(PbrtError::error(&format!(
            "Texture graph exceeds the WebGPU evaluator depth limit of {}.",
            MAX_TEXTURE_GRAPH_DEPTH
        )));
    }
    let key = Arc::as_ptr(node) as usize;
    if !visiting.insert(key) {
        return Err(PbrtError::error("Texture graph contains a cycle."));
    }
    for child in &node.children {
        validate_texture_graph(child, depth + 1, visiting)?;
    }
    visiting.remove(&key);
    Ok(())
}

fn sampler_wrap_mode(mode: &str) -> u32 {
    match mode {
        "clamp" => 1,
        "black" => 2,
        _ => 0,
    }
}

fn sampler_filter_mode(mode: &str) -> Result<u32, PbrtError> {
    match mode {
        "point" | "nearest" => Ok(0),
        "bilinear" | "linear" => Ok(1),
        "trilinear" => Ok(2),
        "ewa" => Err(PbrtError::error(
            "GPU texture filter \"ewa\" is not supported yet.",
        )),
        other => Err(PbrtError::error(&format!(
            "Unknown GPU texture filter \"{other}\"."
        ))),
    }
}

fn build_material_attributes(
    source_material: &NodeMaterial,
    kind: &str,
    builder: &mut FlatBuilder,
) -> Result<Vec<AttributeRef>, PbrtError> {
    match kind {
        "mix" => {
            if let Some(attribute) = texture_attribute_ref(source_material, "amount", builder)? {
                Ok(vec![attribute])
            } else {
                let amount = source_material.params.get_one_float("amount", 0.5) as f32;
                if !amount.is_finite() {
                    return Err(PbrtError::error(&format!(
                        "Material \"{}\" has invalid mix amount.",
                        source_material.name
                    )));
                }
                Ok(vec![push_scalar_attribute(builder, "amount", amount)?])
            }
        }
        "coateddiffuse" => {
            let thickness = source_material.params.get_one_float("thickness", 1.0) as f32;
            let g = source_material.params.get_one_float("g", 0.0) as f32;
            let max_depth_i = source_material.params.get_one_int("maxdepth", 10);
            let n_samples_i = source_material.params.get_one_int("nsamples", 1);
            validate_layer_limits(&source_material.name, max_depth_i, n_samples_i)?;
            let max_depth = max_depth_i as f32;
            let n_samples = n_samples_i as f32;
            if ![thickness, g, max_depth, n_samples]
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(PbrtError::error(&format!(
                    "Material \"{}\" has invalid coateddiffuse parameters.",
                    source_material.name
                )));
            }
            let albedo_attribute = if let Some(attribute) =
                texture_attribute_ref(source_material, "reflectance", builder)?
            {
                attribute
            } else {
                let albedo = diffuse_reflectance(source_material)?;
                push_spectrum_attribute(builder, "albedo", &albedo)?
            };
            let thickness_attribute = texture_attribute_ref(source_material, "thickness", builder)?
                .unwrap_or(push_scalar_attribute(builder, "thickness", thickness)?);
            let g_attribute = texture_attribute_ref(source_material, "g", builder)?
                .unwrap_or(push_scalar_attribute(builder, "g", g)?);
            Ok(vec![
                thickness_attribute,
                albedo_attribute,
                g_attribute,
                push_scalar_attribute(builder, "maxdepth", max_depth)?,
                push_scalar_attribute(builder, "nsamples", n_samples)?,
            ])
        }
        "coatedconductor" => {
            let thickness = source_material.params.get_one_float("thickness", 1.0) as f32;
            let g = source_material.params.get_one_float("g", 0.0) as f32;
            let max_depth_i = source_material.params.get_one_int("maxdepth", 10);
            let n_samples_i = source_material.params.get_one_int("nsamples", 1);
            validate_layer_limits(&source_material.name, max_depth_i, n_samples_i)?;
            let max_depth = max_depth_i as f32;
            let n_samples = n_samples_i as f32;
            if ![thickness, g, max_depth, n_samples]
                .iter()
                .all(|v| v.is_finite())
            {
                return Err(PbrtError::error(&format!(
                    "Material \"{}\" has invalid coatedconductor parameters.",
                    source_material.name
                )));
            }
            let thickness_attribute = texture_attribute_ref(source_material, "thickness", builder)?
                .unwrap_or(push_scalar_attribute(builder, "thickness", thickness)?);
            let g_attribute = texture_attribute_ref(source_material, "g", builder)?
                .unwrap_or(push_scalar_attribute(builder, "g", g)?);
            Ok(vec![
                thickness_attribute,
                g_attribute,
                push_scalar_attribute(builder, "maxdepth", max_depth)?,
                push_scalar_attribute(builder, "nsamples", n_samples)?,
            ])
        }
        "diffuse" => {
            if let Some(attribute) = texture_attribute_ref(source_material, "reflectance", builder)?
            {
                return Ok(vec![attribute]);
            }
            let reflectance = diffuse_reflectance(source_material)?;
            Ok(vec![push_spectrum_attribute(
                builder,
                "reflectance",
                &reflectance,
            )?])
        }
        "dielectric" | "thindielectric" => {
            if let Some(attribute) =
                texture_attribute_ref_unbounded(source_material, "eta", builder)?
            {
                return Ok(vec![attribute]);
            }
            let eta = spectrum_attribute(
                source_material,
                "eta",
                &Spectrum::from(1.5),
                SpectrumType::Unbounded,
            )?;
            let dense_eta = eta.to_dense();
            if (0..crate::util::spectrum::DENSE_SPECTRUM_SAMPLES)
                .any(|index| !dense_eta[index].is_finite() || dense_eta[index] <= 0.0)
            {
                return Err(PbrtError::error(&format!(
                    "Material \"{}\" has invalid dielectric eta.",
                    source_material.name
                )));
            }
            Ok(vec![push_spectrum_attribute(builder, "eta", &eta)?])
        }
        "conductor" => {
            reject_scalar_textures(source_material, &["uroughness", "vroughness"])?;
            let eta_attribute = if let Some(attribute) =
                texture_attribute_ref_unbounded(source_material, "eta", builder)?
            {
                attribute
            } else {
                let eta = spectrum_attribute(
                    source_material,
                    "eta",
                    &Spectrum::from(0.2),
                    SpectrumType::Unbounded,
                )?;
                let dense_eta = eta.to_dense();
                if (0..crate::util::spectrum::DENSE_SPECTRUM_SAMPLES)
                    .any(|index| !dense_eta[index].is_finite() || dense_eta[index] <= 0.0)
                {
                    return Err(PbrtError::error(&format!(
                        "Material \"{}\" has invalid conductor eta.",
                        source_material.name
                    )));
                }
                push_spectrum_attribute(builder, "eta", &eta)?
            };
            let k_attribute = if let Some(attribute) =
                texture_attribute_ref_unbounded(source_material, "k", builder)?
            {
                attribute
            } else {
                let k = spectrum_attribute(
                    source_material,
                    "k",
                    &Spectrum::from(3.0),
                    SpectrumType::Unbounded,
                )?;
                push_spectrum_attribute(builder, "k", &k)?
            };
            let roughness_attribute = if let Some(attribute) =
                texture_attribute_ref(source_material, "roughness", builder)?
            {
                attribute
            } else {
                let roughness = source_material.params.get_one_float("roughness", 0.0) as f32;
                if !roughness.is_finite() || roughness < 0.0 {
                    return Err(PbrtError::error(&format!(
                        "Material \"{}\" has invalid conductor roughness.",
                        source_material.name
                    )));
                }
                push_scalar_attribute(builder, "roughness", roughness)?
            };
            Ok(vec![eta_attribute, k_attribute, roughness_attribute])
        }
        _ => Err(PbrtError::error(&format!(
            "unsupported GPU material kind: {kind}"
        ))),
    }
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
    texture_attributes: Vec<u32>,
    texture_nodes: Vec<FlatTextureNode>,
    texture_child_indices: Vec<u32>,
    texture_nodes_by_ptr: HashMap<usize, u32>,
    texture_nodes_by_name: HashMap<(crate::gpu::ir::node::TextureKind, String), u32>,
    spectrum_table_builder: DenseSpectrumBuilder,
    output: Option<Output>,
    source_materials: Vec<Arc<NodeMaterial>>,
    sampler: Option<NodeSampler>,
    integrator: Option<NodeIntegrator>,
    light_sampling_models: Vec<LightSamplingModel>,
    light_positions: Vec<[f32; 3]>,
    triangle_distributions: Vec<TriangleDistributionEntry>,
    lights: Vec<Light>,
    light_bound_inputs: Vec<LightBoundInput>,
}

fn flatten_node_ref(
    node_ref: &NodeRef,
    parent_transform: &Transform,
    builder: &mut FlatBuilder,
    stack: &mut Vec<usize>,
    material_kind: Option<&str>,
) -> Result<(), PbrtError> {
    let node_key = Arc::as_ptr(node_ref) as usize;
    if stack.contains(&node_key) {
        return Err(PbrtError::error(
            "Cycle detected while flattening GPU Node IR.",
        ));
    }
    stack.push(node_key);

    let (
        name,
        local_transform,
        camera,
        film,
        output,
        sampler,
        integrator,
        light,
        shapes,
        instances,
        children,
    ) = {
        let node = node_ref
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let material = node
            .components
            .iter()
            .find_map(|component| match component {
                Component::Material(component) => Some(Arc::clone(&component.material)),
                _ => None,
            });
        for component in &node.components {
            if let Component::Scene(component) = component {
                for texture_node in &component.scene.texture_nodes {
                    register_texture_node(texture_node, builder)?;
                }
            }
        }
        let mut shapes = Vec::new();
        let mut instances = Vec::new();
        let camera = node
            .components
            .iter()
            .find_map(|component| match component {
                Component::Camera(component) => Some(component.camera.clone()),
                _ => None,
            });
        let film = node
            .components
            .iter()
            .find_map(|component| match component {
                Component::Film(component) => Some(component.film.clone()),
                _ => None,
            });
        let output = node
            .components
            .iter()
            .find_map(|component| match component {
                Component::Output(component) => Some(component.output.clone()),
                _ => None,
            });
        let sampler = node
            .components
            .iter()
            .find_map(|component| match component {
                Component::Sampler(component) => Some(component.sampler.clone()),
                _ => None,
            });
        let integrator = node
            .components
            .iter()
            .find_map(|component| match component {
                Component::Integrator(component) => Some(component.integrator.clone()),
                _ => None,
            });
        let light = node
            .components
            .iter()
            .find_map(|component| match component {
                Component::Light(component) => Some(component.light.clone()),
                _ => None,
            });
        let area_light = node
            .components
            .iter()
            .find_map(|component| match component {
                Component::AreaLight(component) => Some(component.area_light.clone()),
                _ => None,
            });
        for (component_index, component) in node.components.iter().enumerate() {
            match component {
                Component::Shape(component) => {
                    let shape = match &component.shape {
                        Shape::TriangleMesh(mesh) => mesh.as_ref().clone(),
                        Shape::Sphere(_) => {
                            return Err(PbrtError::error(&format!(
                                "Shape node \"{}\" must be tessellated before flattening.",
                                node.name
                            )));
                        }
                        Shape::Disk(_) => {
                            return Err(PbrtError::error(&format!(
                                "Shape node \"{}\" must be tessellated before flattening.",
                                node.name
                            )));
                        }
                    };
                    let shape = remove_invalid_triangles(shape)?;
                    if shape.indices.is_empty() {
                        continue;
                    }
                    let input_normals = shape.normals.clone();
                    let shape = complete_triangle_attributes(shape, &node.name)?;
                    let material = material.clone().ok_or_else(|| {
                        PbrtError::error(&format!(
                            "Shape node \"{}\" has no Material component.",
                            node.name
                        ))
                    })?;
                    shapes.push((
                        component_index,
                        shape,
                        material,
                        area_light.clone(),
                        component.reverse_orientation,
                        input_normals,
                    ));
                }
                Component::Instance(component) => {
                    instances.push((
                        Arc::clone(&component.instance.target),
                        component.instance.transform.clone(),
                    ));
                }
                _ => {}
            }
        }
        (
            node.name.clone(),
            node.transform.matrix,
            camera,
            film,
            output,
            sampler,
            integrator,
            light,
            shapes,
            instances,
            node.children.clone(),
        )
    };

    let world_transform = multiply_transform(parent_transform, &local_transform);
    if let Some(sampler) = sampler {
        register_root_component(&mut builder.sampler, sampler, stack.len(), "Sampler")?;
    }
    if let Some(integrator) = integrator {
        register_root_component(
            &mut builder.integrator,
            integrator,
            stack.len(),
            "Integrator",
        )?;
    }
    if let Some(output) = output {
        if stack.len() != 1 {
            return Err(PbrtError::error(
                "Output component must be attached to the GPU root node.",
            ));
        }
        if builder
            .output
            .replace(Output {
                filename: output.filename,
            })
            .is_some()
        {
            return Err(PbrtError::error(
                "Multiple output components were found while flattening GPU Node IR.",
            ));
        }
    }
    if let Some(film) = film {
        if film.name != "rgb" {
            return Err(PbrtError::error(&format!(
                "WebGPU four-way rendering supports only RGB film, got \"{}\".",
                film.name
            )));
        }
        let resolution = viewport_resolution(&film.params)?;
        if builder.viewport.is_some() {
            return Err(PbrtError::error(
                "Multiple films were found while flattening GPU Node IR.",
            ));
        }
        builder.viewport = Some(Viewport { resolution });
        let sensor_name = film.params.get_one_string("sensor", "cie1931");
        let iso = film.params.get_one_float("iso", 100.0);
        let white_balance = film.params.get_one_float("whitebalance", 0.0);
        let sensor = PixelSensor::create(&sensor_name, iso, white_balance)?;
        let mut sensor_response = [0; 3];
        for (id, response) in sensor_response.iter_mut().zip(sensor.response_spectra()) {
            *id = builder.spectrum_table_builder.intern_dense(response, 0)?;
        }
        builder.film = Some(Film {
            sensor_response,
            output_rgb_from_sensor_rgb: sensor.output_rgb_from_sensor_rgb(),
            imaging_ratio: sensor.imaging_ratio(),
            scale: film.params.get_one_float("scale", 1.0),
            max_sample_luminance: film
                .params
                .get_one_float("maxcomponentvalue", f32::INFINITY),
        });
    }
    if let Some(camera) = camera {
        let fov = camera.params.get_one_float("fov", 90.0) as f32;
        if builder.camera.is_some() {
            return Err(PbrtError::error(
                "Multiple cameras were found while flattening GPU Node IR.",
            ));
        }
        let viewport = builder.viewport.as_ref().ok_or_else(|| {
            PbrtError::error("A camera must be attached to a node with a film component.")
        })?;
        builder.camera = Some(Camera {
            camera_to_world: world_transform,
            fov,
            screen_window: screen_window(&camera.params, viewport.resolution)?,
        });
    }
    if let Some(light) = light {
        let (position, intensity, intensity_max, scale) =
            point_light(&light, &world_transform, &name)?;
        let position_index = u32::try_from(builder.light_positions.len())
            .map_err(|_| PbrtError::error("The flattened GPU light position table exceeds u32."))?;
        builder.light_positions.push(position);
        let sampling_model = u32::try_from(builder.light_sampling_models.len()).map_err(|_| {
            PbrtError::error("The flattened GPU light sampling model table exceeds u32.")
        })?;
        builder.light_sampling_models.push(LightSamplingModel {
            kind: LightKind::Point,
            geometry_kind: LightGeometryKind::Position,
            geometry_index: position_index,
            distribution_offset: 0,
            distribution_count: 0,
            total_area: 0.0,
            flags: 0,
        });
        builder.light_bound_inputs.push(LightBoundInput::Point {
            handle: u32::try_from(builder.lights.len())
                .map_err(|_| PbrtError::error("The flattened GPU light table exceeds u32."))?,
            world_position: position,
            intensity_max,
            scale,
        });
        let i_attr = push_spectrum_attribute(builder, "I", &intensity)?;
        let scale_attr = push_scalar_attribute(builder, "scale", scale)?;
        builder.lights.push(Light {
            kind: LightKind::Point,
            attributes: vec![i_attr, scale_attr],
            sampling_model,
        });
    }
    for (component_index, shape, material, area_light, reverse_orientation, _input_normals) in
        shapes
    {
        let geometry = geometry_index(node_key, component_index, &name, &shape, builder)?;
        let material = material_index(&material, builder, material_kind)?;
        let instance_index = u32::try_from(builder.instances.len())
            .map_err(|_| PbrtError::error("The flattened GPU instance table exceeds u32."))?;
        let area_light_handle = if let Some(area_light) = area_light {
            let light_handle = u32::try_from(builder.lights.len())
                .map_err(|_| PbrtError::error("The flattened GPU light table exceeds u32."))?;
            let triangle_count = shape.indices.len() / 3;
            if triangle_count == 0 {
                return Err(PbrtError::error(&format!(
                    "Area-light shape node \"{name}\" contains no triangles."
                )));
            }
            let (emission, emission_max, scale, two_sided) = area_light_record(&area_light, &name)?;
            let distribution_offset =
                u32::try_from(builder.triangle_distributions.len()).map_err(|_| {
                    PbrtError::error("The flattened GPU distribution table exceeds u32.")
                })?;
            let mut total_area = 0.0;
            let mut bound_triangles = Vec::with_capacity(triangle_count);
            let mut entries = Vec::with_capacity(triangle_count);
            for primitive in 0..triangle_count {
                let primitive = u32::try_from(primitive).map_err(|_| {
                    PbrtError::error("The flattened GPU area-light primitive exceeds u32.")
                })?;
                let i0 = shape.indices[primitive as usize * 3] as usize;
                let i1 = shape.indices[primitive as usize * 3 + 1] as usize;
                let i2 = shape.indices[primitive as usize * 3 + 2] as usize;
                let positions = [
                    transform_point(&world_transform, shape.positions[i0].0),
                    transform_point(&world_transform, shape.positions[i1].0),
                    transform_point(&world_transform, shape.positions[i2].0),
                ];
                let area = triangle_area(positions);
                if !area.is_finite() {
                    return Err(PbrtError::error(&format!(
                        "Area light shape node \"{name}\" contains a non-finite triangle area."
                    )));
                }
                if area <= 0.0 {
                    continue;
                }
                let mut geometric_normal = triangle_geometric_normal(positions)?;
                if reverse_orientation ^ transform_swaps_handedness(world_transform) {
                    geometric_normal = scale3(geometric_normal, -1.0);
                }
                total_area += area;
                bound_triangles.push(AreaTriangleInput {
                    world_positions: positions,
                    area,
                    geometric_normal,
                });
                entries.push((primitive, area));
            }
            if entries.is_empty() || !total_area.is_finite() || total_area <= 0.0 {
                return Err(PbrtError::error(&format!(
                    "Area-light shape node \"{name}\" contains no valid triangles."
                )));
            }
            let mut cumulative = 0.0;
            let mut previous_cdf = 0.0;
            for (primitive, area) in entries {
                cumulative += area / total_area;
                if cumulative <= previous_cdf {
                    return Err(PbrtError::error(&format!(
                        "Area-light shape node \"{name}\" has indistinguishable adjacent CDF entries after f32 packing."
                    )));
                }
                builder
                    .triangle_distributions
                    .push(TriangleDistributionEntry {
                        primitive,
                        cdf: cumulative,
                        area,
                    });
                previous_cdf = cumulative;
            }
            if let Some(last) = builder.triangle_distributions.last_mut() {
                last.cdf = 1.0;
            }
            let sampling_model =
                u32::try_from(builder.light_sampling_models.len()).map_err(|_| {
                    PbrtError::error("The flattened GPU light sampling model table exceeds u32.")
                })?;
            builder.light_sampling_models.push(LightSamplingModel {
                kind: LightKind::Area,
                geometry_kind: LightGeometryKind::Instance,
                geometry_index: instance_index,
                distribution_offset,
                distribution_count: u32::try_from(bound_triangles.len()).map_err(|_| {
                    PbrtError::error("The flattened GPU area-light distribution exceeds u32.")
                })?,
                total_area,
                flags: u32::from(two_sided),
            });
            let emission_attr = push_spectrum_attribute(builder, "L", &emission)?;
            let scale_attr = push_scalar_attribute(builder, "scale", scale)?;
            builder.lights.push(Light {
                kind: LightKind::Area,
                attributes: vec![emission_attr, scale_attr],
                sampling_model,
            });
            builder.light_bound_inputs.push(LightBoundInput::AreaGroup {
                handle: light_handle,
                triangles: bound_triangles,
                emission_max,
                scale,
                two_sided,
            });
            light_handle
        } else {
            INVALID_INDEX
        };
        builder.instances.push(Instance {
            geometry,
            transform: world_transform,
            material,
            area_light: area_light_handle,
            reverse_orientation,
        });
    }
    for (target, instance_transform) in instances {
        let target_parent = multiply_transform(&world_transform, &instance_transform.matrix);
        flatten_node_ref(&target, &target_parent, builder, stack, material_kind)?;
    }
    for child in children {
        flatten_node_ref(&child, &world_transform, builder, stack, material_kind)?;
    }

    stack.pop();
    Ok(())
}

fn register_root_component<T>(
    destination: &mut Option<T>,
    value: T,
    depth: usize,
    kind: &str,
) -> Result<(), PbrtError> {
    if depth != 1 {
        return Err(PbrtError::error(&format!(
            "{kind} component must be attached to the GPU root node."
        )));
    }
    if destination.replace(value).is_some() {
        return Err(PbrtError::error(&format!(
            "Multiple {kind} components were found while flattening GPU Node IR."
        )));
    }
    Ok(())
}

fn render_settings(
    sampler: &Option<NodeSampler>,
    integrator: &Option<NodeIntegrator>,
) -> Result<RenderSettings, PbrtError> {
    let sampler = sampler.as_ref();
    let integrator = integrator.as_ref();
    if let Some(sampler) = sampler {
        if sampler.name != "independent" {
            log::warn!(
                "GPU sampler '{}' is not implemented; falling back to independent sampler.",
                sampler.name
            );
        }
    }
    if let Some(integrator) = integrator {
        if integrator.name != "path" && integrator.name != "volpath" {
            return Err(PbrtError::error(&format!(
                "Unsupported GPU integrator: {}.",
                integrator.name
            )));
        }
    }
    let samples_per_pixel = sampler
        .map(|sampler| sampler.params.get_one_int("pixelsamples", 4))
        .unwrap_or(4);
    let configured_max_depth = integrator
        .map(|integrator| integrator.params.get_one_int("maxdepth", 5))
        .unwrap_or(5);
    let max_depth = configured_max_depth.min(MAX_GPU_RENDER_DEPTH);
    if configured_max_depth > MAX_GPU_RENDER_DEPTH {
        log::warn!(
            "GPU maxdepth {} exceeds the backend limit {}; clamping to {}.",
            configured_max_depth,
            MAX_GPU_RENDER_DEPTH,
            MAX_GPU_RENDER_DEPTH
        );
    }
    let seed = sampler
        .map(|sampler| sampler.params.get_one_int("seed", 0))
        .unwrap_or(0);
    let light_sampler = integrator
        .map(|integrator| integrator.params.get_one_string("lightsampler", "bvh"))
        .unwrap_or_else(|| "bvh".to_string());
    if samples_per_pixel <= 0 || max_depth < 0 || seed < 0 {
        return Err(PbrtError::error(
            "GPU render settings must have positive samples and non-negative depth/seed.",
        ));
    }
    Ok(RenderSettings {
        samples_per_pixel: u32::try_from(samples_per_pixel)
            .map_err(|_| PbrtError::error("GPU samples per pixel do not fit in u32."))?,
        max_depth: u32::try_from(max_depth)
            .map_err(|_| PbrtError::error("GPU max depth does not fit in u32."))?,
        seed: u32::try_from(seed).map_err(|_| PbrtError::error("GPU seed does not fit in u32."))?,
        light_sampler,
        disable_wavelength_jitter: crate::options::PbrtOptions::get().disable_wavelength_jitter,
    })
}

fn point_light(
    light: &NodeLight,
    parent_transform: &Transform,
    node_name: &str,
) -> Result<([f32; 3], Spectrum, f32, f32), PbrtError> {
    if light.name != "point" {
        return Err(PbrtError::error(&format!(
            "Unsupported GPU light \"{}\" on node \"{}\".",
            light.name, node_name
        )));
    }
    let from = light.params.get_one_point("from", &[0.0, 0.0, 0.0]);
    if from.len() != 3 || !from.iter().all(|value| value.is_finite()) {
        return Err(PbrtError::error(&format!(
            "Point light on node \"{}\" has an invalid from parameter.",
            node_name
        )));
    }
    let light_transform = multiply_transform(parent_transform, &light.transform.matrix);
    let position = transform_point(
        &light_transform,
        [from[0] as f32, from[1] as f32, from[2] as f32],
    );
    let white = Spectrum::from(1.0);
    let intensity = light
        .params
        .get_one_spectrum_typed("I", &white, SpectrumType::Illuminant);
    let mut scale = light.params.get_one_float("scale", 1.0);
    let photometric = spectrum_to_photometric(&intensity);
    if photometric > 0.0 {
        scale /= photometric;
    }
    let power = light.params.get_one_float("power", -1.0);
    if power > 0.0 {
        scale *= power / (4.0 * std::f32::consts::PI);
    }
    let intensity_max = intensity.max_value() as f32;
    if !position.iter().all(|value| value.is_finite()) || !scale.is_finite() {
        return Err(PbrtError::error(&format!(
            "Point light on node \"{}\" contains a non-finite value.",
            node_name
        )));
    }
    Ok((position, intensity, intensity_max, scale as f32))
}

fn area_light_record(
    light: &NodeAreaLight,
    node_name: &str,
) -> Result<(Spectrum, f32, f32, bool), PbrtError> {
    if light.name != "diffuse" {
        return Err(PbrtError::error(&format!(
            "Unsupported GPU area light \"{}\" on node \"{}\".",
            light.name, node_name
        )));
    }
    if light.params.has_parameter("filename") {
        return Err(PbrtError::error(&format!(
            "Textured GPU area light on node \"{}\" is not implemented.",
            node_name
        )));
    }
    let white = Spectrum::from(1.0);
    let emission_spectrum =
        light
            .params
            .get_one_spectrum_typed("L", &white, SpectrumType::Illuminant);
    let photometric = spectrum_to_photometric(&emission_spectrum);
    let scale = light.params.get_one_float("scale", 1.0)
        / if photometric > 0.0 { photometric } else { 1.0 };
    let power = light.params.get_one_float("power", -1.0);
    if power > 0.0 {
        return Err(PbrtError::error(&format!(
            "GPU area light power on node \"{node_name}\" is not implemented."
        )));
    }
    let emission_max = emission_spectrum.max_value() as f32;
    if !scale.is_finite() {
        return Err(PbrtError::error(&format!(
            "GPU area light on node \"{}\" contains a non-finite emission value.",
            node_name
        )));
    }
    Ok((
        emission_spectrum,
        emission_max,
        scale as f32,
        light.params.get_one_bool("twosided", false),
    ))
}

fn triangle_area(positions: [[f32; 3]; 3]) -> f32 {
    let edge0 = [
        positions[1][0] - positions[0][0],
        positions[1][1] - positions[0][1],
        positions[1][2] - positions[0][2],
    ];
    let edge1 = [
        positions[2][0] - positions[0][0],
        positions[2][1] - positions[0][1],
        positions[2][2] - positions[0][2],
    ];
    let cross = [
        edge0[1] * edge1[2] - edge0[2] * edge1[1],
        edge0[2] * edge1[0] - edge0[0] * edge1[2],
        edge0[0] * edge1[1] - edge0[1] * edge1[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

fn triangle_geometric_normal(positions: [[f32; 3]; 3]) -> Result<[f32; 3], PbrtError> {
    let edge0 = sub3(positions[1], positions[0]);
    let edge1 = sub3(positions[2], positions[0]);
    let cross = [
        edge0[1] * edge1[2] - edge0[2] * edge1[1],
        edge0[2] * edge1[0] - edge0[0] * edge1[2],
        edge0[0] * edge1[1] - edge0[1] * edge1[0],
    ];
    let length = dot3(cross, cross).sqrt();
    if !length.is_finite() || length == 0.0 {
        return Err(PbrtError::error(
            "Area light triangle geometric normal is invalid.",
        ));
    }
    Ok(scale3(cross, 1.0 / length))
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scale3(v: [f32; 3], scale: f32) -> [f32; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

fn transform_point(matrix: &Transform, point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
        matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
        matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
    ]
}

fn viewport_resolution(params: &ParameterDictionary) -> Result<[u32; 2], PbrtError> {
    let xresolution = params.get_one_int("xresolution", 1280);
    let yresolution = params.get_one_int("yresolution", 720);
    let resolution = [
        u32::try_from(xresolution)
            .map_err(|_| PbrtError::error("Film xresolution must be positive and fit in u32."))?,
        u32::try_from(yresolution)
            .map_err(|_| PbrtError::error("Film yresolution must be positive and fit in u32."))?,
    ];
    if resolution.contains(&0) {
        return Err(PbrtError::error("Film resolution must be positive."));
    }
    Ok(resolution)
}

fn screen_window(
    params: &ParameterDictionary,
    resolution: [u32; 2],
) -> Result<[f32; 4], PbrtError> {
    if let Some(values) = params.get_floats_ref("screenwindow") {
        if values.len() != 4 {
            return Err(PbrtError::error(
                "Camera screenwindow must contain four values.",
            ));
        }
        return Ok([
            values[0] as f32,
            values[1] as f32,
            values[2] as f32,
            values[3] as f32,
        ]);
    }

    let frame = params.get_one_float(
        "frameaspectratio",
        resolution[0] as f32 / resolution[1] as f32,
    ) as f32;
    if frame > 1.0 {
        Ok([-frame, frame, -1.0, 1.0])
    } else {
        Ok([-1.0, 1.0, -1.0 / frame, 1.0 / frame])
    }
}

fn geometry_index(
    node_key: usize,
    component_index: usize,
    node_name: &str,
    shape: &TriangleMeshShape,
    builder: &mut FlatBuilder,
) -> Result<u32, PbrtError> {
    let key = (node_key, component_index);
    if let Some(&geometry) = builder.geometries_by_shape.get(&key) {
        return Ok(geometry);
    }

    let vertex_count = u32::try_from(shape.positions.len()).map_err(|_| {
        PbrtError::error(&format!(
            "Too many vertices in shape node \"{}\".",
            node_name
        ))
    })?;
    let index_count = u32::try_from(shape.indices.len()).map_err(|_| {
        PbrtError::error(&format!(
            "Too many indices in shape node \"{}\".",
            node_name
        ))
    })?;
    if shape.indices.len() % 3 != 0 {
        return Err(PbrtError::error(&format!(
            "Shape node \"{}\" has an index count that is not divisible by three.",
            node_name
        )));
    }

    validate_attribute_len(
        node_name,
        shape.normals.as_deref(),
        shape.positions.len(),
        "normal",
    )?;
    validate_attribute_len(
        node_name,
        shape.tangents.as_deref(),
        shape.positions.len(),
        "tangent",
    )?;
    validate_attribute_len(node_name, shape.uvs.as_deref(), shape.positions.len(), "UV")?;

    let first_vertex = u32::try_from(builder.vertices.len()).map_err(|_| {
        PbrtError::error("The flattened GPU vertex buffer exceeds the u32 index range.")
    })?;
    let first_index = u32::try_from(builder.indices.len()).map_err(|_| {
        PbrtError::error("The flattened GPU index buffer exceeds the u32 index range.")
    })?;
    for (index, position) in shape.positions.iter().enumerate() {
        builder.vertices.push(Vertex {
            position: position.0,
            normal: shape
                .normals
                .as_ref()
                .map(|normals| normals[index].0)
                .unwrap_or([0.0; 3]),
            tangent: shape
                .tangents
                .as_ref()
                .map(|tangents| tangents[index].0)
                .unwrap_or([0.0; 3]),
            uv: shape
                .uvs
                .as_ref()
                .map(|uvs| uvs[index].0)
                .unwrap_or([0.0; 2]),
        });
    }
    for &index in &shape.indices {
        if index >= vertex_count {
            return Err(PbrtError::error(&format!(
                "Shape node \"{}\" contains an out-of-range vertex index.",
                node_name
            )));
        }
        let flattened_index = first_vertex.checked_add(index).ok_or_else(|| {
            PbrtError::error("The flattened GPU vertex buffer exceeds the u32 index range.")
        })?;
        builder.indices.push(flattened_index);
    }

    let geometry = u32::try_from(builder.geometries.len()).map_err(|_| {
        PbrtError::error("The flattened GPU geometry table exceeds the u32 index range.")
    })?;
    builder.geometries.push(Geometry {
        first_vertex,
        vertex_count,
        first_index,
        index_count,
    });
    builder.geometries_by_shape.insert(key, geometry);
    Ok(geometry)
}

fn material_index(
    source_material: &Arc<NodeMaterial>,
    builder: &mut FlatBuilder,
    material_kind: Option<&str>,
) -> Result<u32, PbrtError> {
    if let Some(index) = builder
        .source_materials
        .iter()
        .position(|material| Arc::ptr_eq(material, source_material))
    {
        return u32::try_from(index).map_err(|_| {
            PbrtError::error("The flattened GPU material table exceeds the u32 index range.")
        });
    }
    let requested_kind = material_kind.unwrap_or(&source_material.kind);
    let source_kind = source_material.kind.as_str();
    let supported = matches!(
        requested_kind,
        "diffuse"
            | "dielectric"
            | "thindielectric"
            | "conductor"
            | "mix"
            | "coateddiffuse"
            | "coatedconductor"
    );
    let texture_fallback = if supported && has_texture_attribute(source_material) {
        match UnsupportedTexturePolicy::from_environment()? {
            UnsupportedTexturePolicy::Error => false,
            UnsupportedTexturePolicy::DiagnosticMagenta => {
                log::warn!(
                    "GPU material \"{}\" contains unsupported textures; using diffuse reflectance (1, 0, 1).",
                    source_material.name
                );
                true
            }
        }
    } else {
        false
    };
    let (kind, mut attributes) = if texture_fallback {
        let magenta = Spectrum::from_rgb(&[1.0, 0.0, 1.0], SpectrumType::Albedo);
        (
            "diffuse",
            vec![push_spectrum_attribute(builder, "reflectance", &magenta)?],
        )
    } else if !supported {
        log::warn!(
            concat!(
                "GPU material \"{}\" of kind \"{}\" is unsupported; ",
                "using diffuse reflectance (1, 1, 0)."
            ),
            source_material.name,
            requested_kind,
        );
        ("diffuse", {
            let yellow = Spectrum::from_rgb(&[1.0, 1.0, 0.0], SpectrumType::Albedo);
            vec![push_spectrum_attribute(builder, "reflectance", &yellow)?]
        })
    } else {
        (
            requested_kind,
            build_material_attributes(source_material, requested_kind, builder)?,
        )
    };
    for (name, texture_node) in &source_material.texture_attributes {
        // Displacement is consumed during CPU shape realization and is not a
        // material-evaluation attribute in the WebGPU backend.
        if name == "displacement" {
            continue;
        }
        let texture = texture_node
            .components
            .iter()
            .find_map(|component| match component {
                TextureComponent::Texture(texture) => Some(texture),
                TextureComponent::Mapping(_) => None,
            });
        if let Some(texture) = texture {
            if let Some(&index) = builder
                .texture_nodes_by_name
                .get(&(texture.kind, texture_node.name.clone()))
            {
                if !attributes.iter().any(|attribute| attribute.name == *name) {
                    attributes.push(AttributeRef {
                        kind: AttributeKind::Texture,
                        index,
                        name: name.clone(),
                    });
                }
            }
        }
    }
    if matches!(kind, "coateddiffuse" | "coatedconductor") {
        let child_kinds: &[&str] = if kind == "coateddiffuse" {
            &["dielectric", "diffuse"]
        } else {
            &["dielectric", "conductor"]
        };
        let mut children = Vec::with_capacity(2);
        for child_kind in child_kinds {
            let mut child_params = source_material.params.clone();
            // The synthetic substrate material must not inherit texture
            // references from the coating itself.  The coating stores its
            // textured albedo/eta/k as attributes on the wrapper; retaining
            // those parameter references would make the synthetic child try
            // to evaluate them a second time.
            for key in child_params.get_keys() {
                if child_params.get_key_type(&key) == "texture" {
                    child_params.remove_parameter(&key);
                }
            }
            let child = Arc::new(NodeMaterial {
                name: format!("{}:{}", source_material.name, child_kind),
                kind: (*child_kind).to_string(),
                params: child_params,
                material_attributes: Vec::new(),
                texture_attributes: Vec::new(),
            });
            children.push(AttributeRef {
                kind: AttributeKind::Material,
                index: material_index(&child, builder, Some(child_kind))?,
                name: (*child_kind).to_string(),
            });
        }
        children.append(&mut attributes);
        attributes = children;
    } else if kind == "mix" {
        let mut material_attributes = Vec::new();
        for (name, child) in &source_material.material_attributes {
            let child_index = material_index(child, builder, None)?;
            material_attributes.push(AttributeRef {
                kind: AttributeKind::Material,
                index: child_index,
                name: name.clone(),
            });
        }
        if material_attributes.len() != 2 {
            return Err(PbrtError::error(
                "GPU mix material must contain exactly two material references.",
            ));
        }
        material_attributes.append(&mut attributes);
        attributes = material_attributes;
    }
    let index = u32::try_from(builder.materials.len()).map_err(|_| {
        PbrtError::error("The flattened GPU material table exceeds the u32 index range.")
    })?;
    builder.materials.push(Material {
        kind: kind.to_string(),
        source_kind: source_kind.to_string(),
        attributes: attributes.clone(),
    });
    builder.source_materials.push(Arc::clone(source_material));
    Ok(index)
}

fn diffuse_reflectance(source_material: &NodeMaterial) -> Result<Spectrum, PbrtError> {
    let default_reflectance = Spectrum::from(0.5);
    spectrum_attribute(
        source_material,
        "reflectance",
        &default_reflectance,
        SpectrumType::Albedo,
    )
}

fn texture_attribute_ref(
    source_material: &NodeMaterial,
    key: &str,
    builder: &FlatBuilder,
) -> Result<Option<AttributeRef>, PbrtError> {
    let Some((name, node)) = source_material
        .texture_attributes
        .iter()
        .find(|(name, _)| name == key)
    else {
        return Ok(None);
    };
    let texture = node
        .components
        .iter()
        .find_map(|component| match component {
            TextureComponent::Texture(texture) => Some(texture),
            TextureComponent::Mapping(_) => None,
        });
    let Some(texture) = texture else {
        return Err(PbrtError::error(&format!(
            "Texture attribute \"{key}\" has no texture component."
        )));
    };
    let Some(&index) = builder
        .texture_nodes_by_name
        .get(&(texture.kind, node.name.clone()))
    else {
        return Err(PbrtError::error(&format!(
            "Texture attribute \"{key}\" references an unregistered texture node."
        )));
    };
    Ok(Some(AttributeRef {
        kind: AttributeKind::Texture,
        index,
        name: name.clone(),
    }))
}

fn texture_attribute_ref_unbounded(
    source_material: &NodeMaterial,
    key: &str,
    builder: &FlatBuilder,
) -> Result<Option<AttributeRef>, PbrtError> {
    let Some(mut attribute) = texture_attribute_ref(source_material, key, builder)? else {
        return Ok(None);
    };
    attribute.kind = AttributeKind::TextureUnbounded;
    Ok(Some(attribute))
}

fn reject_scalar_textures(source_material: &NodeMaterial, keys: &[&str]) -> Result<(), PbrtError> {
    if let Some(key) = source_material
        .params
        .get_keys()
        .iter()
        .find_map(|stored_key| {
            let is_texture = source_material.params.get_key_type(stored_key) == "texture";
            let name = source_material.params.get_key_name(stored_key);
            (is_texture && keys.iter().any(|key| *key == name)).then_some(name)
        })
    {
        return Err(PbrtError::error(&format!(
            "Material \"{}\" uses unsupported scalar texture attribute \"{key}\".",
            source_material.name
        )));
    }
    Ok(())
}

fn has_texture_attribute(source_material: &NodeMaterial) -> bool {
    source_material.params.get_keys().iter().any(|key| {
        source_material.params.get_key_type(key) == "texture"
            && source_material.params.get_key_name(key) != "displacement"
    })
}

fn spectrum_attribute(
    source_material: &NodeMaterial,
    key: &str,
    default: &Spectrum,
    spectrum_type: SpectrumType,
) -> Result<Spectrum, PbrtError> {
    let has_texture = source_material.params.get_keys().iter().any(|stored_key| {
        source_material.params.get_key_type(stored_key) == "texture"
            && source_material.params.get_key_name(stored_key) == key
    });
    if !has_texture {
        return Ok(source_material.params.get_one_spectrum(key, default));
    }
    match UnsupportedTexturePolicy::from_environment()? {
        UnsupportedTexturePolicy::Error => Err(PbrtError::error(&format!(
            "Material \"{}\" uses unsupported texture attribute \"{key}\".",
            source_material.name
        ))),
        UnsupportedTexturePolicy::DiagnosticMagenta => {
            log::warn!(
                "Material \"{}\" texture attribute \"{key}\" uses diagnostic magenta.",
                source_material.name
            );
            Ok(Spectrum::from_rgb(&[1.0, 0.0, 1.0], spectrum_type))
        }
    }
}

fn validate_attribute_len<T>(
    node_name: &str,
    attribute: Option<&[T]>,
    vertex_count: usize,
    attribute_name: &str,
) -> Result<(), PbrtError> {
    if let Some(attribute) = attribute {
        if attribute.len() != vertex_count {
            return Err(PbrtError::error(&format!(
                "Shape node \"{}\" has {} {} values for {} vertices.",
                node_name,
                attribute.len(),
                attribute_name,
                vertex_count
            )));
        }
    }
    Ok(())
}
