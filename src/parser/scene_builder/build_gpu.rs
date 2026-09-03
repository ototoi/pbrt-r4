use super::scene_entity::{InstanceSceneEntity, ShapeSceneEntity};

use crate::gpu::ir::flat::flatten_node;
use crate::gpu::ir::node::{
    node_ref_to_json_string, tessellate_shapes, triangle_mesh_from_params, Accelerator,
    AcceleratorComponent, Camera, CameraComponent, Component, Film, FilmComponent, Filter,
    FilterComponent, Instance, InstanceComponent, Integrator, IntegratorComponent, Light,
    LightComponent, Material, MaterialComponent, Medium, MediumComponent, Node, NodeRef, Output,
    OutputComponent, Sampler, SamplerComponent, Scene, SceneComponent, Shape, ShapeComponent,
    SphereShape, Texture, TextureKind as NodeTextureKind, Transform,
};
use crate::gpu::wavefront::WavefrontPathIntegrator;
use crate::util::error::PbrtError;

use super::path_resolver::make_absolute_path;
use super::SceneBuilder;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

impl SceneBuilder {
    /// Realise the accumulated entities directly into an `Integrator` on GPU.
    pub fn build_gpu(&self) -> Result<Arc<RwLock<WavefrontPathIntegrator>>, PbrtError> {
        if let Some(error) = self.import_errors.first() {
            return Err(PbrtError::error(error));
        }
        if let Some(error) = self.option_errors.first() {
            return Err(PbrtError::error(error));
        }

        // Build the declarative GPU Node IR for the scene.
        let ir_node = self.build_gpu_ir_node()?;
        match node_ref_to_json_string(&ir_node) {
            Ok(json) => println!("GPU Node IR before tessellation:\n{json}"),
            Err(error) => eprintln!("Failed to serialize GPU Node IR before tessellation: {error}"),
        }

        // Tessellate shapes in the IR node to ensure all shapes are represented as triangle meshes.
        {
            let mut ir_node = ir_node
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            tessellate_shapes(&mut ir_node);
        }
        match node_ref_to_json_string(&ir_node) {
            Ok(json) => println!("GPU Node IR after tessellation:\n{json}"),
            Err(error) => eprintln!("Failed to serialize GPU Node IR after tessellation: {error}"),
        }

        // Lower the IR node to a flat scene representation.
        let flat_scene = flatten_node(ir_node)?;

        // Create the WavefrontPathIntegrator from the flat scene.
        let integrator = WavefrontPathIntegrator::create(flat_scene)?;
        return Ok(Arc::new(RwLock::new(integrator)));
    }

    pub fn build_gpu_ir_node(&self) -> Result<Arc<RwLock<Node>>, PbrtError> {
        let mut root_node = Node::new("root");
        root_node.add_component(Component::Scene(SceneComponent {
            scene: Scene {
                textures: self.build_texture_resources(),
            },
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

        let materials = self.build_material_resources();
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

    fn build_material_resources(&self) -> Vec<Arc<Material>> {
        self.materials
            .iter()
            .map(|material| {
                Arc::new(Material {
                    name: material.base.name.clone(),
                    kind: material.base.name.clone(),
                    params: material.base.params.clone(),
                })
            })
            .collect()
    }

    fn build_texture_resources(&self) -> Vec<Arc<Texture>> {
        let mut textures =
            Vec::with_capacity(self.float_textures.len() + self.spectrum_textures.len());
        for texture in &self.float_textures {
            textures.push(Arc::new(Texture {
                name: texture.base.name.clone(),
                kind: NodeTextureKind::Float,
                params: texture.base.params.clone(),
                transform: node_transform(&texture.render_from_texture),
            }));
        }
        for texture in &self.spectrum_textures {
            textures.push(Arc::new(Texture {
                name: texture.base.name.clone(),
                kind: NodeTextureKind::Spectrum,
                params: texture.base.params.clone(),
                transform: node_transform(&texture.render_from_texture),
            }));
        }
        textures
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
            "trianglemesh" | "plymesh" => {
                match triangle_mesh_from_params(shape.base.name.as_str(), params)? {
                    Some(mesh) => Shape::TriangleMesh(Box::new(mesh)),
                    None => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        let mut node = Node::new(&shape.base.name);
        node.transform = node_transform(&shape.render_from_object.primary());
        node.add_component(Component::Shape(ShapeComponent { shape: shape_value }));

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
