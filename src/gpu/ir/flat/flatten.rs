use super::{
    identity_transform, multiply_transform, Camera, Geometry, Instance, Material, Scene, Transform,
    Vertex, Viewport,
};
use crate::gpu::ir::node::{
    Component, Material as NodeMaterial, NodeRef, Shape, TriangleMeshShape,
};
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;

use std::collections::HashMap;
use std::sync::Arc;

pub fn flatten_node(root: NodeRef) -> Result<Scene, PbrtError> {
    let mut builder = FlatBuilder::default();
    let mut stack = Vec::new();
    flatten_node_ref(&root, &identity_transform(), &mut builder, &mut stack)?;
    let camera = builder
        .camera
        .ok_or_else(|| PbrtError::error("No camera was found while flattening GPU Node IR."))?;
    let viewport = builder
        .viewport
        .ok_or_else(|| PbrtError::error("No film was found while flattening GPU Node IR."))?;
    Ok(Scene {
        camera,
        viewport,
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
    source_materials: Vec<Arc<NodeMaterial>>,
}

fn flatten_node_ref(
    node_ref: &NodeRef,
    parent_transform: &Transform,
    builder: &mut FlatBuilder,
    stack: &mut Vec<usize>,
) -> Result<(), PbrtError> {
    let node_key = Arc::as_ptr(node_ref) as usize;
    if stack.contains(&node_key) {
        return Err(PbrtError::error(
            "Cycle detected while flattening GPU Node IR.",
        ));
    }
    stack.push(node_key);

    let (name, local_transform, camera, film, shapes, instances, children) = {
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
                    };
                    let material = material.clone().ok_or_else(|| {
                        PbrtError::error(&format!(
                            "Shape node \"{}\" has no Material component.",
                            node.name
                        ))
                    })?;
                    shapes.push((component_index, shape, material));
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
            shapes,
            instances,
            node.children.clone(),
        )
    };

    let world_transform = multiply_transform(parent_transform, &local_transform);
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
    for (component_index, shape, material) in shapes {
        let geometry = geometry_index(node_key, component_index, &name, &shape, builder)?;
        let material = material_index(&material, builder)?;
        builder.instances.push(Instance {
            geometry,
            transform: world_transform,
            material,
        });
    }
    for (target, instance_transform) in instances {
        let target_parent = multiply_transform(&world_transform, &instance_transform.matrix);
        flatten_node_ref(&target, &target_parent, builder, stack)?;
    }
    for child in children {
        flatten_node_ref(&child, &world_transform, builder, stack)?;
    }

    stack.pop();
    Ok(())
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
        kind: source_material.kind.clone(),
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
