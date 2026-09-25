use super::{
    append_area_light, flatten_light, geometry_index, multiply_transform, register_material_source,
    register_root_component, screen_window, viewport_resolution, Camera, Component, Film,
    FlatBuilder, Instance, NodeRef, Output, Shape, Transform, Viewport, INVALID_INDEX,
};
use crate::film::PixelSensor;
use crate::util::error::PbrtError;
use std::sync::Arc;

pub fn flatten_node_ref(
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
                        Shape::Cylinder(_) => {
                            return Err(PbrtError::error(&format!(
                                "Shape node \"{}\" must be tessellated before flattening.",
                                node.name
                            )));
                        }
                        Shape::Cone(_) => {
                            return Err(PbrtError::error(&format!(
                                "Shape node \"{}\" must be tessellated before flattening.",
                                node.name
                            )));
                        }
                        Shape::Paraboloid(_) => {
                            return Err(PbrtError::error(
                                "Shape must be tessellated before flattening.",
                            ))
                        }
                        Shape::HeightField(_) => {
                            return Err(PbrtError::error(
                                "Shape must be tessellated before flattening.",
                            ));
                        }
                        Shape::BilinearMesh(_) => {
                            return Err(PbrtError::error(
                                "Shape must be tessellated before flattening.",
                            ))
                        }
                        Shape::Hyperboloid(_) => {
                            return Err(PbrtError::error(
                                "Shape must be tessellated before flattening.",
                            ))
                        }
                        Shape::Nurbs(_) => {
                            return Err(PbrtError::error(
                                "Shape must be tessellated before flattening.",
                            ))
                        }
                    };
                    if shape.indices.is_empty() {
                        continue;
                    }
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
        match film.name.as_str() {
            "rgb" => {}
            "gbuffer" => log::warn!(
                "WebGPU does not produce GBuffer AOVs; treating the gbuffer film as RGB."
            ),
            name => {
                return Err(PbrtError::error(&format!(
                    "WebGPU rendering does not support film \"{name}\"."
                )))
            }
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
        flatten_light(light, &world_transform, &name, builder)?;
    }
    for (component_index, shape, material, area_light, reverse_orientation) in shapes {
        let geometry = geometry_index(node_key, component_index, &name, &shape, builder)?;
        let material = register_material_source(&material, builder, material_kind)?;
        let instance_index = u32::try_from(builder.instances.len())
            .map_err(|_| PbrtError::error("The flattened GPU instance table exceeds u32."))?;
        let area_light_handle = if let Some(area_light) = area_light {
            append_area_light(
                area_light,
                &shape,
                &name,
                &world_transform,
                instance_index,
                material,
                reverse_orientation,
                builder,
            )?
        } else {
            INVALID_INDEX
        };
        builder.instances.push(Instance {
            geometry,
            transform: world_transform,
            material_root: material,
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
