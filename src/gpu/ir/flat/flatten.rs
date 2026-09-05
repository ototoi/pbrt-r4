use super::{
    build_light_bounds, build_light_bvh, identity_transform, multiply_transform, transform_normal,
    transform_swaps_handedness, AreaLight, AreaTriangleInput, Camera, Geometry, Instance,
    LightBoundInput, LightKind, LightRecord, Material, PointLight, RenderSettings, Scene,
    Transform, TriangleDistributionEntry, TriangleDistributionRange, Vertex, Viewport,
    INVALID_INDEX,
};
use crate::gpu::ir::node::{
    complete_triangle_attributes, AreaLight as NodeAreaLight, Component,
    Integrator as NodeIntegrator, Light as NodeLight, Material as NodeMaterial, NodeRef,
    Sampler as NodeSampler, Shape, TriangleMeshShape,
};
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;
use crate::util::spectrum::{Spectrum, SpectrumType};

use std::collections::HashMap;
use std::sync::Arc;

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
    let render_settings = render_settings(&builder.sampler, &builder.integrator)?;
    let light_bounds = build_light_bounds(&builder.light_bound_inputs)?;
    let light_bvh = build_light_bvh(&builder.lights, &light_bounds)?;
    Ok(Scene {
        camera,
        viewport,
        output,
        render_settings,
        point_lights: builder.point_lights,
        area_lights: builder.area_lights,
        triangle_distributions: builder.triangle_distributions,
        lights: builder.lights,
        light_bounds,
        light_bvh,
        vertices: builder.vertices,
        indices: builder.indices,
        geometries: builder.geometries,
        instances: builder.instances,
        materials: builder.materials,
    })
}

#[derive(Default)]
struct FlatBuilder {
    camera: Option<Camera>,
    viewport: Option<Viewport>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    geometries: Vec<Geometry>,
    geometries_by_shape: HashMap<(usize, usize), u32>,
    instances: Vec<Instance>,
    materials: Vec<Material>,
    output: Option<super::Output>,
    source_materials: Vec<Arc<NodeMaterial>>,
    sampler: Option<NodeSampler>,
    integrator: Option<NodeIntegrator>,
    point_lights: Vec<PointLight>,
    area_lights: Vec<AreaLight>,
    triangle_distributions: Vec<TriangleDistributionEntry>,
    lights: Vec<LightRecord>,
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
            .replace(super::Output {
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
        let resolution = viewport_resolution(&film.params)?;
        if builder.viewport.is_some() {
            return Err(PbrtError::error(
                "Multiple films were found while flattening GPU Node IR.",
            ));
        }
        builder.viewport = Some(Viewport { resolution });
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
        let point_light_index = u32::try_from(builder.point_lights.len())
            .map_err(|_| PbrtError::error("The flattened GPU point-light table exceeds u32."))?;
        let (point, intensity_max, scale) = point_light(&light, &world_transform, &name)?;
        builder.point_lights.push(point);
        builder.light_bound_inputs.push(LightBoundInput::Point {
            handle: u32::try_from(builder.lights.len())
                .map_err(|_| PbrtError::error("The flattened GPU light table exceeds u32."))?,
            world_position: builder.point_lights.last().unwrap().position,
            intensity_max,
            scale,
        });
        builder.lights.push(LightRecord {
            kind: LightKind::Point,
            payload: point_light_index,
        });
    }
    for (component_index, shape, material, area_light, reverse_orientation, input_normals) in shapes
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
                let normals = input_normals.as_ref().map(|normals| {
                    [
                        transform_normal(world_transform, normals[i0].0),
                        transform_normal(world_transform, normals[i1].0),
                        transform_normal(world_transform, normals[i2].0),
                    ]
                });
                let normals = match normals {
                    Some([Ok(n0), Ok(n1), Ok(n2)]) => Some([n0, n1, n2]),
                    Some(_) => {
                        return Err(PbrtError::error(&format!(
                            "Area light shape node \"{name}\" has a singular normal transform."
                        )))
                    }
                    None => None,
                };
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
                if let Some(normals) = normals {
                    let normal_sum = add3(add3(normals[0], normals[1]), normals[2]);
                    if dot3(geometric_normal, normal_sum) < 0.0 {
                        geometric_normal = scale3(geometric_normal, -1.0);
                    }
                } else if reverse_orientation ^ transform_swaps_handedness(world_transform) {
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
            let area_light_index = u32::try_from(builder.area_lights.len())
                .map_err(|_| PbrtError::error("The flattened GPU area-light table exceeds u32."))?;
            builder.area_lights.push(AreaLight {
                instance: instance_index,
                distribution: TriangleDistributionRange {
                    offset: distribution_offset,
                    count: u32::try_from(bound_triangles.len()).map_err(|_| {
                        PbrtError::error("The flattened GPU area-light distribution exceeds u32.")
                    })?,
                    total_area,
                },
                emission,
                two_sided,
            });
            builder.lights.push(LightRecord {
                kind: LightKind::Area,
                payload: area_light_index,
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
    let max_depth = integrator
        .map(|integrator| integrator.params.get_one_int("maxdepth", 5))
        .unwrap_or(5);
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
    })
}

fn point_light(
    light: &NodeLight,
    parent_transform: &Transform,
    node_name: &str,
) -> Result<(PointLight, f32, f32), PbrtError> {
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
    // See the area-light emission note below: the v4 photometric division
    // cancels the illuminant scale carried by the spectral Sample(). Since we
    // emit the nominal RGB from `to_rgb()`, dividing by the photometric here
    // would darken the light by ~photometric (~107x for a white illuminant).
    let mut scale = light.params.get_one_float("scale", 1.0);
    let power = light.params.get_one_float("power", -1.0);
    if power > 0.0 {
        scale *= power / (4.0 * std::f32::consts::PI);
    }
    let intensity_max = intensity.max_value() as f32;
    let rgb = intensity.to_rgb();
    let intensity = [
        (rgb[0] * scale) as f32,
        (rgb[1] * scale) as f32,
        (rgb[2] * scale) as f32,
    ];
    if !position
        .iter()
        .chain(intensity.iter())
        .all(|value| value.is_finite())
    {
        return Err(PbrtError::error(&format!(
            "Point light on node \"{}\" contains a non-finite value.",
            node_name
        )));
    }
    Ok((
        PointLight {
            position,
            intensity,
        },
        intensity_max,
        scale as f32,
    ))
}

fn area_light_record(
    light: &NodeAreaLight,
    node_name: &str,
) -> Result<([f32; 3], f32, f32, bool), PbrtError> {
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
    // pbrt-v4 divides `scale` by SpectrumToPhotometric(L) (lights.cpp:910),
    // but that normalization exactly cancels the illuminant's photometric
    // scale carried by `L->Sample(lambda)` when the emitted spectrum is
    // converted back to RGB. Because we emit the nominal RGB from
    // `to_rgb()` directly (which already excludes that factor), applying the
    // photometric division here would darken the light by ~photometric
    // (~107x for a white illuminant). So the RGB emission is just the user
    // `scale` times the nominal RGB.
    let scale = light.params.get_one_float("scale", 1.0);
    let power = light.params.get_one_float("power", -1.0);
    if power > 0.0 {
        return Err(PbrtError::error(&format!(
            "GPU area light power on node \"{node_name}\" is not implemented."
        )));
    }
    let rgb = emission_spectrum.to_rgb();
    let emission_max = emission_spectrum.max_value() as f32;
    let emission = [
        (rgb[0] * scale) as f32,
        (rgb[1] * scale) as f32,
        (rgb[2] * scale) as f32,
    ];
    if !emission.iter().all(|value| value.is_finite()) {
        return Err(PbrtError::error(&format!(
            "GPU area light on node \"{}\" contains a non-finite emission value.",
            node_name
        )));
    }
    Ok((
        emission,
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

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
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
    let index = u32::try_from(builder.materials.len()).map_err(|_| {
        PbrtError::error("The flattened GPU material table exceeds the u32 index range.")
    })?;
    builder.materials.push(Material {
        kind: material_kind.unwrap_or(&source_material.kind).to_string(),
    });
    builder.source_materials.push(Arc::clone(source_material));
    Ok(index)
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
