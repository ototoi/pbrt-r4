use super::scene_entity::{
    InstanceDefinitionSceneEntity, InstanceSceneEntity, MediumInterfaceNames, RenderFromObject,
    ShapeSceneEntity,
};

use crate::gpu::ir::flat::Scene as FlatScene;
use crate::gpu::ir::node::Node;
use crate::gpu::wavefront::WavefrontPathIntegrator;
use crate::util::error::PbrtError;

use super::SceneBuilder;

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

        // Create the IR node for the scene. This is a placeholder for future implementation.
        let ir_node = self.build_gpu_ir_node()?;

        // Lower the IR node to a flat scene representation.
        let flat_scene = self.lower_node_to_flat(ir_node)?;

        // Create the WavefrontPathIntegrator from the flat scene.
        let integrator = WavefrontPathIntegrator::create(flat_scene)?;
        return Ok(Arc::new(RwLock::new(integrator)));
    }

    pub fn build_gpu_ir_node(&self) -> Result<Arc<Node>, PbrtError> {
        let mut root_node = Node::new("root");

        for shape in &self.shapes {
            let node = self.realize_gpu_shape(shape)?;
            root_node.add_child(node);
        }
        for shape in &self.animated_shapes {
            let node = self.realize_gpu_shape(shape)?;
            root_node.add_child(node);
        }

        return Ok(Arc::new(root_node));
    }

    fn realize_gpu_shape(&self, shape: &ShapeSceneEntity) -> Result<Arc<Node>, PbrtError> {
        let name = &shape.base.name;
        match name.as_str() {
            "trianglemesh" => {
                // Placeholder for triangle mesh processing logic.
            }
            "plymesh" => {
                // Placeholder for PLY mesh processing logic.
            }
            _ => {
                //Do not return an error for unrecognized shapes; just skip them for now.
            }
        }
        return Err(PbrtError::error("GPU build not implemented yet"));
    }

    pub fn lower_node_to_flat(&self, _node: Arc<Node>) -> Result<FlatScene, PbrtError> {
        return Err(PbrtError::error("GPU build not implemented yet"));
    }
}
