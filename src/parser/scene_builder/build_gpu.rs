use super::scene_entity::{InstanceSceneEntity, ShapeSceneEntity};

use crate::gpu::ir::flat::flatten_node;
use crate::gpu::ir::node::{
    loop_subdiv_mesh_from_params, node_ref_to_json_string, tessellate_shapes,
    triangle_mesh_from_params, Accelerator, AcceleratorComponent, AreaLight as NodeAreaLight,
    AreaLightComponent, Camera, CameraComponent, Component, DiskShape, Film, FilmComponent, Filter,
    FilterComponent, Instance, InstanceComponent, Integrator, IntegratorComponent, Light,
    LightComponent, Material, MaterialComponent, Medium, MediumComponent, Node, NodeRef, Output,
    OutputComponent, Sampler, SamplerComponent, Scene, SceneComponent, Shape, ShapeComponent,
    SphereShape, Texture, TextureComponent, TextureKind as NodeTextureKind, TextureMapping,
    TextureNode, Transform, UvMapping,
};
use crate::gpu::wavefront::WavefrontPathIntegrator;
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;
use crate::util::imageio::{read_raw_image_with_encoding, ColorEncoding};
use crate::util::spectrum::Spectrum;

use super::path_resolver::make_absolute_path;
use super::SceneBuilder;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

impl SceneBuilder {
    /// Realise the accumulated entities directly into an `Integrator` on GPU.
    pub fn build_gpu(&self) -> Result<Arc<RwLock<WavefrontPathIntegrator>>, PbrtError> {
        self.build_gpu_with_progress(false)
    }

    pub fn build_gpu_with_progress(
        &self,
        show_progress: bool,
    ) -> Result<Arc<RwLock<WavefrontPathIntegrator>>, PbrtError> {
        if let Some(error) = self.import_errors.first() {
            return Err(PbrtError::error(error));
        }
        if let Some(error) = self.option_errors.first() {
            return Err(PbrtError::error(error));
        }

        // Build the declarative GPU Node IR for the scene.
        log::info!("GPU build: building Node IR");
        let ir_node = self.build_gpu_ir_node()?;
        log::info!("GPU build: Node IR built; serializing before tessellation");
        match node_ref_to_json_string(&ir_node) {
            Ok(json) => println!("GPU Node IR before tessellation:\n{json}"),
            Err(error) => eprintln!("Failed to serialize GPU Node IR before tessellation: {error}"),
        }

        // Tessellate shapes in the IR node to ensure all shapes are represented as triangle meshes.
        {
            log::info!("GPU build: tessellating shapes");
            let mut ir_node = ir_node
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            tessellate_shapes(&mut ir_node)?;
        }
        log::info!("GPU build: tessellation complete; serializing after tessellation");
        match node_ref_to_json_string(&ir_node) {
            Ok(json) => println!("GPU Node IR after tessellation:\n{json}"),
            Err(error) => eprintln!("Failed to serialize GPU Node IR after tessellation: {error}"),
        }

        // Lower the IR node to a flat scene representation.
        log::info!("GPU build: flattening Node IR");
        let flat_scene = flatten_node(ir_node)?;
        log::info!("GPU build: flatten complete; creating WebGPU integrator");

        // Create the WavefrontPathIntegrator from the flat scene.
        let integrator = WavefrontPathIntegrator::create_with_progress(flat_scene, show_progress)?;
        log::info!("GPU build: WebGPU integrator created");
        return Ok(Arc::new(RwLock::new(integrator)));
    }

    pub fn build_gpu_ir_node(&self) -> Result<Arc<RwLock<Node>>, PbrtError> {
        let texture_nodes = self.build_texture_resources()?;
        let texture_lookup = texture_node_lookup(&texture_nodes);
        let mut root_node = Node::new("root");
        root_node.add_component(Component::Scene(SceneComponent {
            scene: Scene { texture_nodes },
        }));
        root_node.add_component(Component::Output(OutputComponent {
            output: Output {
                filename: self.film_params.get_one_string("filename", "pbrt.exr"),
            },
        }));
        root_node.add_component(Component::Sampler(SamplerComponent {
            sampler: Sampler {
                name: self.sampler_name.clone(),
                params: self.sampler_params.clone(),
            },
        }));
        root_node.add_component(Component::Integrator(IntegratorComponent {
            integrator: Integrator {
                name: self.integrator_name.clone(),
                params: self.integrator_params.clone(),
            },
        }));
        root_node.add_component(Component::Accelerator(AcceleratorComponent {
            accelerator: Accelerator {
                name: self.accelerator_name.clone(),
                params: self.accelerator_params.clone(),
            },
        }));
        root_node.add_component(Component::Filter(FilterComponent {
            filter: Filter {
                name: self.filter_name.clone(),
                params: self.filter_params.clone(),
            },
        }));

        let mut materials = self.build_material_resources(&texture_lookup)?;
        self.populate_gpu_material_references(&mut materials)?;
        let named_materials = self.build_named_material_resources(&materials);

        for medium in self.media.values() {
            root_node.add_component(Component::Medium(MediumComponent {
                medium: Medium {
                    name: medium.base.name.clone(),
                    params: medium.base.params.clone(),
                    transform: node_transform(&medium.render_from_medium.primary()),
                },
            }));
        }

        root_node.add_child(self.build_camera_node());

        for shape in &self.shapes {
            if let Some(node) = self.realize_gpu_shape(shape, &materials, &named_materials)? {
                root_node.add_child(node);
            }
        }
        for shape in &self.animated_shapes {
            if let Some(node) = self.realize_gpu_shape(shape, &materials, &named_materials)? {
                root_node.add_child(node);
            }
        }

        for light in &self.lights {
            let mut node = Node::new(&light.base.base.name);
            node.add_component(Component::Light(LightComponent {
                light: Light {
                    name: light.base.base.name.clone(),
                    params: light.base.base.params.clone(),
                    transform: node_transform(&light.base.render_from_object.primary()),
                    medium: light.medium.clone(),
                },
            }));
            root_node.add_child(Arc::new(RwLock::new(node)));
        }

        let definitions = self.build_instance_definitions(&materials, &named_materials)?;
        for instance in &self.instance_uses {
            root_node.add_child(self.build_instance_node(instance, &definitions)?);
        }

        return Ok(Arc::new(RwLock::new(root_node)));
    }

    fn populate_gpu_material_references(
        &self,
        materials: &mut [Arc<Material>],
    ) -> Result<(), PbrtError> {
        for index in 0..materials.len() {
            if materials[index].kind != "mix" {
                continue;
            }
            let names = materials[index]
                .params
                .get_strings_ref("materials")
                .filter(|names| names.len() >= 2)
                .map(|names| (names[0].clone(), names[1].clone()))
                .unwrap_or_else(|| {
                    (
                        materials[index].params.get_one_string("namedmaterial1", ""),
                        materials[index].params.get_one_string("namedmaterial2", ""),
                    )
                });
            if names.0.is_empty() || names.1.is_empty() {
                return Err(PbrtError::error(
                    "Mix material is missing named-material references.",
                ));
            }
            let first_index = *self.named_materials.get(&names.0).ok_or_else(|| {
                PbrtError::error(&format!(
                    "Mix material references unknown material \"{}\".",
                    names.0
                ))
            })?;
            let second_index = *self.named_materials.get(&names.1).ok_or_else(|| {
                PbrtError::error(&format!(
                    "Mix material references unknown material \"{}\".",
                    names.1
                ))
            })?;
            let first = materials
                .get(first_index)
                .cloned()
                .ok_or_else(|| PbrtError::error("Mix material first reference is out of range."))?;
            let second = materials.get(second_index).cloned().ok_or_else(|| {
                PbrtError::error("Mix material second reference is out of range.")
            })?;
            Arc::get_mut(&mut materials[index])
                .ok_or_else(|| {
                    PbrtError::error("GPU material graph has unexpected shared ownership.")
                })?
                .material_attributes = vec![("a".to_string(), first), ("b".to_string(), second)];
        }
        Ok(())
    }

    fn build_camera_node(&self) -> NodeRef {
        let mut node = Node::new("camera");
        // `camera_to_world` is historically named but stores pbrt's
        // cameraFromWorld transform. GPU Node IR stores the camera-to-world
        // transform, matching the transform used to generate world-space rays.
        let camera_to_world = self.camera_to_world.to_transform().inverse();
        node.transform = node_transform(&camera_to_world);
        node.add_component(Component::Camera(CameraComponent {
            camera: Camera {
                params: self.camera_params.clone(),
                medium: String::new(),
            },
        }));
        let mut film_params = self.film_params.clone();
        film_params.remove_parameter("filename");
        node.add_component(Component::Film(FilmComponent {
            film: Film {
                name: self.film_name.clone(),
                params: film_params,
            },
        }));
        Arc::new(RwLock::new(node))
    }

    fn build_material_resources(
        &self,
        texture_lookup: &HashMap<(NodeTextureKind, String), Arc<TextureNode>>,
    ) -> Result<Vec<Arc<Material>>, PbrtError> {
        self.materials
            .iter()
            .map(|material| {
                let texture_attributes = texture_attributes_for_material(material, texture_lookup)?;
                Ok(Arc::new(Material {
                    name: material.base.name.clone(),
                    kind: material.base.name.clone(),
                    params: material.base.params.clone(),
                    material_attributes: Vec::new(),
                    texture_attributes,
                }))
            })
            .collect()
    }

    fn build_texture_resources(&self) -> Result<Vec<Arc<TextureNode>>, PbrtError> {
        let mut raw_textures =
            Vec::with_capacity(self.float_textures.len() + self.spectrum_textures.len());
        let float_names = texture_names_by_index(&self.named_float_textures);
        let spectrum_names = texture_names_by_index(&self.named_spectrum_textures);
        for (index, texture) in self.float_textures.iter().enumerate() {
            let node_name = float_names
                .get(&index)
                .map(String::as_str)
                .unwrap_or(&texture.base.name);
            raw_textures.push(texture_node(
                node_name,
                &texture.base.name,
                NodeTextureKind::Float,
                &texture.base.params,
                &texture.render_from_texture,
                &self.work_dirs,
            )?);
        }
        for (index, texture) in self.spectrum_textures.iter().enumerate() {
            let node_name = spectrum_names
                .get(&index)
                .map(String::as_str)
                .unwrap_or(&texture.base.name);
            raw_textures.push(texture_node(
                node_name,
                &texture.base.name,
                NodeTextureKind::Spectrum,
                &texture.base.params,
                &texture.render_from_texture,
                &self.work_dirs,
            )?);
        }
        let mut lookup = HashMap::new();
        for (index, node) in raw_textures.iter().enumerate() {
            let kind = node
                .components
                .iter()
                .find_map(|component| match component {
                    TextureComponent::Texture(texture) => Some(texture.kind),
                    TextureComponent::Mapping(_) => None,
                })
                .ok_or_else(|| PbrtError::error("TextureNode has no texture component."))?;
            lookup.insert((kind, node.name.clone()), index);
        }
        let mut memo = vec![None; raw_textures.len()];
        let mut visiting = vec![false; raw_textures.len()];
        (0..raw_textures.len())
            .map(|index| {
                materialize_texture_node(index, &raw_textures, &lookup, &mut memo, &mut visiting)
            })
            .collect()
    }

    fn build_named_material_resources(
        &self,
        materials: &[Arc<Material>],
    ) -> HashMap<String, Arc<Material>> {
        self.named_materials
            .iter()
            .filter_map(|(name, index)| {
                materials
                    .get(*index)
                    .map(|material| (name.clone(), Arc::clone(material)))
            })
            .collect()
    }

    fn realize_gpu_shape(
        &self,
        shape: &ShapeSceneEntity,
        materials: &[Arc<Material>],
        named_materials: &HashMap<String, Arc<Material>>,
    ) -> Result<Option<NodeRef>, PbrtError> {
        let resolved_params;
        let params = if shape.base.name == "plymesh" {
            resolved_params = make_absolute_path(&shape.base.params, &self.seen_work_dirs);
            &resolved_params
        } else {
            &shape.base.params
        };
        let shape_value = match shape.base.name.as_str() {
            "sphere" => Shape::Sphere(Box::new(SphereShape {
                params: shape.base.params.clone(),
            })),
            "disk" => Shape::Disk(Box::new(DiskShape {
                params: shape.base.params.clone(),
            })),
            "trianglemesh" | "plymesh" => {
                match triangle_mesh_from_params(shape.base.name.as_str(), params)? {
                    Some(mesh) => Shape::TriangleMesh(Box::new(mesh)),
                    None => return Ok(None),
                }
            }
            "loopsubdiv" => {
                match loop_subdiv_mesh_from_params(params, shape.reverse_orientation)? {
                    Some(mesh) => Shape::TriangleMesh(Box::new(mesh)),
                    None => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        let mut node = Node::new(&shape.base.name);
        node.transform = node_transform(&shape.render_from_object.primary());
        node.add_component(Component::Shape(ShapeComponent {
            shape: shape_value,
            reverse_orientation: shape.reverse_orientation,
        }));

        // Object definitions are shared by ObjectInstance and must not create
        // one AreaLight per definition. The occurrence-specific light is a
        // later lowering concern.
        if shape.instance_name.is_none() {
            if let Some(area_light_index) = shape.area_light_index {
                let area_light = self.area_lights.get(area_light_index).ok_or_else(|| {
                    PbrtError::error(&format!(
                        "Shape node \"{}\" references an invalid area light.",
                        shape.base.name
                    ))
                })?;
                node.add_component(Component::AreaLight(AreaLightComponent {
                    area_light: NodeAreaLight {
                        name: area_light.base.name.clone(),
                        params: area_light.base.params.clone(),
                    },
                }));
            }
        }

        if let Some(material) = self.resolve_gpu_material(shape, materials, named_materials) {
            node.add_component(Component::Material(MaterialComponent { material }));
        }
        Ok(Some(Arc::new(RwLock::new(node))))
    }

    fn resolve_gpu_material(
        &self,
        shape: &ShapeSceneEntity,
        materials: &[Arc<Material>],
        named_materials: &HashMap<String, Arc<Material>>,
    ) -> Option<Arc<Material>> {
        if shape.material_is_default {
            return Some(Arc::new(Material {
                name: "default".to_string(),
                kind: "diffuse".to_string(),
                params: Default::default(),
                material_attributes: Vec::new(),
                texture_attributes: Vec::new(),
            }));
        }
        if let Some(name) = &shape.material_name {
            return named_materials.get(name).cloned();
        }
        if shape.material_index != usize::MAX {
            return materials.get(shape.material_index).cloned();
        }
        None
    }

    fn build_instance_definitions(
        &self,
        materials: &[Arc<Material>],
        named_materials: &HashMap<String, Arc<Material>>,
    ) -> Result<HashMap<String, NodeRef>, PbrtError> {
        let mut definitions = HashMap::new();
        for (name, definition) in &self.instance_definitions {
            let definition_node = Arc::new(RwLock::new(Node::new(name)));
            for shape in &definition.shapes {
                if let Some(child) = self.realize_gpu_shape(shape, materials, named_materials)? {
                    definition_node.write().unwrap().add_child(child);
                }
            }
            for shape in &definition.animated_shapes {
                if let Some(child) = self.realize_gpu_shape(shape, materials, named_materials)? {
                    definition_node.write().unwrap().add_child(child);
                }
            }
            definitions.insert(name.clone(), definition_node);
        }
        Ok(definitions)
    }

    fn build_instance_node(
        &self,
        instance: &InstanceSceneEntity,
        definitions: &HashMap<String, NodeRef>,
    ) -> Result<NodeRef, PbrtError> {
        let target = definitions.get(&instance.name).ok_or_else(|| {
            PbrtError::error(&format!("Unknown object instance \"{}\".", instance.name))
        })?;
        let mut node = Node::new(&format!("instance:{}", instance.name));
        node.add_component(Component::Instance(InstanceComponent {
            instance: Instance {
                target: Arc::clone(target),
                transform: node_transform(&instance.render_from_instance.primary()),
            },
        }));
        Ok(Arc::new(RwLock::new(node)))
    }
}

fn node_transform(transform: &crate::util::transform::Transform) -> Transform {
    Transform {
        matrix: transform.m.m.map(|value| value as f32),
    }
}

fn texture_node(
    name: &str,
    implementation_name: &str,
    kind: NodeTextureKind,
    params: &crate::paramdict::ParameterDictionary,
    render_from_texture: &crate::util::transform::Transform,
    work_dirs: &[String],
) -> Result<TextureNode, PbrtError> {
    if !matches!(
        implementation_name,
        "constant"
            | "imagemap"
            | "scale"
            | "mix"
            | "directionmix"
            | "fbm"
            | "wrinkled"
            | "windy"
            | "dots"
            | "bilerp"
            | "marble"
            | "checkerboard"
            | "checkerboard3d"
    ) && !implementation_name.contains("checkerboard")
    {
        return Err(PbrtError::error(&format!(
            "GPU texture implementation \"{}\" is not supported yet for texture \"{}\".",
            implementation_name, name
        )));
    }
    let mut node = TextureNode::new(name);
    let mipmap = if implementation_name == "imagemap" {
        let params = make_absolute_path(params, work_dirs);
        let filename = params.get_one_string("filename", "");
        if filename.is_empty() {
            None
        } else {
            let default_encoding = if filename.to_ascii_lowercase().ends_with(".png") {
                "sRGB"
            } else {
                "linear"
            };
            let encoding_name = params.get_one_string("encoding", default_encoding);
            let encoding = ColorEncoding::parse(&encoding_name)?;
            let (raw, color_space) = if filename.to_ascii_lowercase().ends_with(".exr") {
                let path = std::path::Path::new(&filename);
                let (raw, _, metadata) =
                    crate::util::imageio::read_image_exr::read_raw_image_exr_with_channels_and_metadata(path)?;
                let color_space = metadata
                    .color_space
                    .map(|space| match space.name {
                        "ACES2065-1" => crate::gpu::ir::node::ColorSpaceId::Aces2065,
                        "DCI-P3" => crate::gpu::ir::node::ColorSpaceId::DciP3,
                        "Rec2020" => crate::gpu::ir::node::ColorSpaceId::Rec2020,
                        _ => crate::gpu::ir::node::ColorSpaceId::Srgb,
                    })
                    .unwrap_or(crate::gpu::ir::node::ColorSpaceId::Srgb);
                (raw, color_space)
            } else {
                (
                    read_raw_image_with_encoding(&filename, encoding)?,
                    crate::gpu::ir::node::ColorSpaceId::Srgb,
                )
            };
            let channels = raw.channels;
            let mut resolution = [raw.resolution.x as u32, raw.resolution.y as u32];
            let mut data = raw.data_f32();
            let mut levels = Vec::new();
            loop {
                levels.push(crate::gpu::ir::node::MipmapLevel {
                    resolution,
                    channels: channels as u32,
                    data: crate::gpu::ir::node::MipmapLevelData::F32(data.clone()),
                });
                if resolution == [1, 1] {
                    break;
                }
                let next_resolution = [(resolution[0] / 2).max(1), (resolution[1] / 2).max(1)];
                let mut next =
                    vec![0.0; next_resolution[0] as usize * next_resolution[1] as usize * channels];
                for y in 0..next_resolution[1] {
                    for x in 0..next_resolution[0] {
                        let mut count = 0.0f32;
                        for oy in 0..2 {
                            for ox in 0..2 {
                                let sx = (2 * x + ox).min(resolution[0] - 1);
                                let sy = (2 * y + oy).min(resolution[1] - 1);
                                let source = (sy * resolution[0] + sx) as usize * channels;
                                let target = (y * next_resolution[0] + x) as usize * channels;
                                for channel in 0..channels {
                                    next[target + channel] += data[source + channel];
                                }
                                count += 1.0;
                            }
                        }
                        let target = (y * next_resolution[0] + x) as usize * channels;
                        for channel in 0..channels {
                            next[target + channel] /= count;
                        }
                    }
                }
                resolution = next_resolution;
                data = next;
            }
            Some(Arc::new(crate::gpu::ir::node::Mipmap {
                levels,
                // The samples themselves are already linear; retain the image
                // primaries until the material spectrum boundary.
                color_space: Some(color_space),
            }))
        }
    } else {
        None
    };
    node.components.push(TextureComponent::Texture(Texture {
        name: implementation_name.to_string(),
        kind,
        params: params.clone(),
        mipmap,
    }));
    if implementation_name == "checkerboard3d"
        || (implementation_name == "checkerboard" && params.get_one_int("dimension", 2) == 3)
    {
        node.components
            .push(TextureComponent::Mapping(TextureMapping::PointTransform(
                node_transform(&render_from_texture.inverse()),
            )));
        return Ok(node);
    }
    if matches!(implementation_name, "fbm" | "wrinkled" | "windy" | "marble") {
        // TextureMapping3D in pbrt-v4 is a PointTransformMapping whose
        // matrix maps render-space points into texture space.
        node.components
            .push(TextureComponent::Mapping(TextureMapping::PointTransform(
                node_transform(&render_from_texture.inverse()),
            )));
        return Ok(node);
    }
    let mapping = params.get_one_string("mapping", "uv");
    if implementation_name == "directionmix" {
        let direction =
            params.get_one_vector3f("dir", &crate::util::base::Vector3f::new(0.0, 1.0, 0.0));
        let direction = render_from_texture.transform_vector(&direction).normalize();
        node.components
            .push(TextureComponent::Mapping(TextureMapping::PointTransform(
                Transform {
                    matrix: [
                        direction.x as f32,
                        direction.y as f32,
                        direction.z as f32,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                },
            )));
        return Ok(node);
    }
    if mapping == "planar" {
        let v1 = params.get_one_vector3f("v1", &crate::util::base::Vector3f::new(1.0, 0.0, 0.0));
        let v2 = params.get_one_vector3f("v2", &crate::util::base::Vector3f::new(0.0, 1.0, 0.0));
        let m = render_from_texture.inverse().m.m.map(|value| value as f64);
        let udelta = params.get_one_float("udelta", 0.0) as f32;
        let vdelta = params.get_one_float("vdelta", 0.0) as f32;
        let planar = Transform {
            matrix: [
                (v1.x as f64 * m[0] + v1.y as f64 * m[4] + v1.z as f64 * m[8]) as f32,
                (v1.x as f64 * m[1] + v1.y as f64 * m[5] + v1.z as f64 * m[9]) as f32,
                (v1.x as f64 * m[2] + v1.y as f64 * m[6] + v1.z as f64 * m[10]) as f32,
                (v1.x as f64 * m[3] + v1.y as f64 * m[7] + v1.z as f64 * m[11]) as f32 + udelta,
                (v2.x as f64 * m[0] + v2.y as f64 * m[4] + v2.z as f64 * m[8]) as f32,
                (v2.x as f64 * m[1] + v2.y as f64 * m[5] + v2.z as f64 * m[9]) as f32,
                (v2.x as f64 * m[2] + v2.y as f64 * m[6] + v2.z as f64 * m[10]) as f32,
                (v2.x as f64 * m[3] + v2.y as f64 * m[7] + v2.z as f64 * m[11]) as f32 + vdelta,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        };
        node.components
            .push(TextureComponent::Mapping(TextureMapping::Planar(planar)));
        return Ok(node);
    }
    if mapping == "spherical" || mapping == "cylindrical" {
        let transform = node_transform(&render_from_texture.inverse());
        node.components
            .push(TextureComponent::Mapping(if mapping == "spherical" {
                TextureMapping::Spherical(transform)
            } else {
                TextureMapping::Cylindrical(transform)
            }));
        return Ok(node);
    }
    if mapping != "uv" {
        return Err(PbrtError::error(&format!(
            "GPU texture mapping \"{}\" is not supported yet for texture \"{}\".",
            mapping, name
        )));
    }
    node.components
        .push(TextureComponent::Mapping(TextureMapping::Uv(UvMapping {
            uscale: params.get_one_float("uscale", 1.0) as f32,
            vscale: params.get_one_float("vscale", 1.0) as f32,
            udelta: params.get_one_float("udelta", 0.0) as f32,
            vdelta: params.get_one_float("vdelta", 0.0) as f32,
        })));
    Ok(node)
}

fn materialize_texture_node(
    index: usize,
    raw_nodes: &[TextureNode],
    lookup: &HashMap<(NodeTextureKind, String), usize>,
    memo: &mut [Option<Arc<TextureNode>>],
    visiting: &mut [bool],
) -> Result<Arc<TextureNode>, PbrtError> {
    if let Some(node) = &memo[index] {
        return Ok(Arc::clone(node));
    }
    if visiting[index] {
        return Err(PbrtError::error("Texture graph contains a cycle."));
    }
    visiting[index] = true;
    let node = &raw_nodes[index];
    let (kind, params) = node
        .components
        .iter()
        .find_map(|component| match component {
            TextureComponent::Texture(texture) => Some((texture.kind, texture.params.clone())),
            TextureComponent::Mapping(_) => None,
        })
        .ok_or_else(|| PbrtError::error("TextureNode has no texture component."))?;
    // Texture operation semantics use positional children.  Do not depend on
    // the dictionary's insertion order: `mix` and checkerboard explicitly
    // define tex1/tex2 in pbrt-v4.
    let preferred_keys: &[&str] = match node
        .components
        .iter()
        .find_map(|component| match component {
            TextureComponent::Texture(texture) => Some(texture.name.as_str()),
            TextureComponent::Mapping(_) => None,
        })
        .unwrap_or("")
    {
        "scale" => &["tex"],
        "mix" => &["tex1", "tex2", "amount"],
        "directionmix" => &["tex1", "tex2"],
        "dots" => &["outside", "inside"],
        "bilerp" => &["v00", "v01", "v10", "v11"],
        name if name.contains("checkerboard") => &["tex1", "tex2"],
        _ => &[],
    };
    let mut texture_keys = params
        .get_keys()
        .into_iter()
        .filter(|key| params.get_key_type(key) == "texture")
        .collect::<Vec<_>>();
    texture_keys.sort_by_key(|key| {
        preferred_keys
            .iter()
            .position(|preferred| *preferred == params.get_key_name(key))
            .unwrap_or(usize::MAX)
    });
    let operation_name = implementation_name(node).unwrap_or("");
    let mut children = Vec::new();

    // pbrt-v4 allows the two operands of mix, directionmix, and checkerboard
    // to be either texture references or literal values.  Materialize literal
    // operands as ordinary constant texture nodes so the flattened evaluator
    // has one uniform child representation for both cases.
    if matches!(operation_name, "mix" | "directionmix" | "dots" | "bilerp")
        || operation_name.contains("checkerboard")
    {
        let slots: &[(&str, f32)] = if operation_name == "dots" {
            &[("outside", 0.0_f32), ("inside", 1.0_f32)]
        } else if operation_name == "bilerp" {
            &[
                ("v00", 0.0_f32),
                ("v01", 1.0_f32),
                ("v10", 0.0_f32),
                ("v11", 1.0_f32),
            ]
        } else {
            &[("tex1", 0.0_f32), ("tex2", 1.0_f32)]
        };
        for (slot, default) in slots {
            let texture_key = texture_keys
                .iter()
                .find(|key| params.get_key_name(key) == *slot)
                .cloned();
            if let Some(key) = texture_key {
                append_texture_children(
                    &mut children,
                    &key,
                    node,
                    kind,
                    &params,
                    lookup,
                    raw_nodes,
                    memo,
                    visiting,
                )?;
            } else {
                children.push(constant_texture_child(
                    &node.name, slot, kind, &params, *default,
                ));
            }
        }
        texture_keys.retain(|key| {
            let name = params.get_key_name(key);
            if operation_name == "dots" {
                name != "outside" && name != "inside"
            } else if operation_name == "bilerp" {
                !matches!(name.as_str(), "v00" | "v01" | "v10" | "v11")
            } else {
                name != "tex1" && name != "tex2"
            }
        });
    }
    for key in texture_keys {
        append_texture_children(
            &mut children,
            &key,
            node,
            kind,
            &params,
            lookup,
            raw_nodes,
            memo,
            visiting,
        )?;
    }
    let mut materialized = node.clone();
    materialized.children = children;
    let materialized = Arc::new(materialized);
    visiting[index] = false;
    memo[index] = Some(Arc::clone(&materialized));
    Ok(materialized)
}

#[allow(clippy::too_many_arguments)]
fn append_texture_children(
    children: &mut Vec<Arc<TextureNode>>,
    key: &str,
    node: &TextureNode,
    kind: NodeTextureKind,
    params: &ParameterDictionary,
    lookup: &HashMap<(NodeTextureKind, String), usize>,
    raw_nodes: &[TextureNode],
    memo: &mut [Option<Arc<TextureNode>>],
    visiting: &mut [bool],
) -> Result<(), PbrtError> {
    let names = params.get_textures_ref(key).ok_or_else(|| {
        PbrtError::error(&format!(
            "Texture node \"{}\" has an invalid texture parameter \"{}\".",
            node.name,
            params.get_key_name(key)
        ))
    })?;
    for name in names.iter() {
        let child_kind =
            if implementation_name(node) == Some("mix") && params.get_key_name(key) == "amount" {
                NodeTextureKind::Float
            } else {
                kind
            };
        let child_index = *lookup.get(&(child_kind, name.clone())).ok_or_else(|| {
            PbrtError::error(&format!(
                "Texture node \"{}\" references unknown child texture \"{}\".",
                node.name, name
            ))
        })?;
        children.push(materialize_texture_node(
            child_index,
            raw_nodes,
            lookup,
            memo,
            visiting,
        )?);
    }
    Ok(())
}

fn constant_texture_child(
    parent_name: &str,
    slot: &str,
    kind: NodeTextureKind,
    params: &ParameterDictionary,
    default: f32,
) -> Arc<TextureNode> {
    let mut constant_params = ParameterDictionary::new();
    let mut node = TextureNode::new(format!("{parent_name}:{slot}:constant"));
    match kind {
        NodeTextureKind::Float => {
            constant_params.add_float("value", params.get_one_float(slot, default as _));
        }
        NodeTextureKind::Spectrum => {
            let spectrum = params.get_one_spectrum(slot, &Spectrum::from(default));
            constant_params.add_spectrum("value", &spectrum);
        }
    }
    node.components.push(TextureComponent::Texture(Texture {
        name: "constant".to_string(),
        kind,
        params: constant_params,
        mipmap: None,
    }));
    Arc::new(node)
}

fn implementation_name(node: &TextureNode) -> Option<&str> {
    node.components
        .iter()
        .find_map(|component| match component {
            TextureComponent::Texture(texture) => Some(texture.name.as_str()),
            TextureComponent::Mapping(_) => None,
        })
}

fn texture_node_lookup(
    texture_nodes: &[Arc<TextureNode>],
) -> HashMap<(NodeTextureKind, String), Arc<TextureNode>> {
    texture_nodes
        .iter()
        .filter_map(|node| {
            let texture = node
                .components
                .iter()
                .find_map(|component| match component {
                    TextureComponent::Texture(texture) => Some(texture),
                    TextureComponent::Mapping(_) => None,
                })?;
            Some(((texture.kind, node.name.clone()), Arc::clone(node)))
        })
        .collect()
}

fn texture_names_by_index(names: &HashMap<String, usize>) -> HashMap<usize, String> {
    names
        .iter()
        .map(|(name, &index)| (index, name.clone()))
        .collect()
}

fn texture_attributes_for_material(
    material: &super::scene_entity::MaterialSceneEntity,
    texture_lookup: &HashMap<(NodeTextureKind, String), Arc<TextureNode>>,
) -> Result<Vec<(String, Arc<TextureNode>)>, PbrtError> {
    let mut attributes = Vec::new();
    for key in material.base.params.get_keys() {
        if material.base.params.get_key_type(&key) != "texture" {
            continue;
        }
        let key_name = material.base.params.get_key_name(&key);
        let names = material.base.params.get_textures_ref(&key).ok_or_else(|| {
            PbrtError::error(&format!("Texture parameter \"{key_name}\" has no value."))
        })?;
        if names.len() != 1 {
            return Err(PbrtError::error(&format!(
                "Texture parameter \"{key_name}\" must name exactly one texture."
            )));
        }
        let texture_name = names.first().ok_or_else(|| {
            PbrtError::error(&format!(
                "Texture parameter \"{key_name}\" has no texture name."
            ))
        })?;
        let float = texture_lookup.get(&(NodeTextureKind::Float, texture_name.clone()));
        let spectrum = texture_lookup.get(&(NodeTextureKind::Spectrum, texture_name.clone()));
        let node = match (float, spectrum) {
            (Some(node), None) => Arc::clone(node),
            (None, Some(node)) => Arc::clone(node),
            (Some(_), Some(_)) => {
                return Err(PbrtError::error(&format!(
                "Texture parameter \"{key_name}\" references ambiguous texture \"{texture_name}\"."
            )))
            }
            (None, None) => {
                return Err(PbrtError::error(&format!(
                "Texture parameter \"{key_name}\" references unknown texture \"{texture_name}\"."
            )))
            }
        };
        attributes.push((key_name, node));
    }
    Ok(attributes)
}
