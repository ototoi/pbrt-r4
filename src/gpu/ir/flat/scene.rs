use super::{
    AreaLight, Camera, DielectricMaterialData, DiffuseMaterialData, Geometry, Instance, LightBVH,
    LightBounds, LightRecord, Material, Output, PointLight, RenderSettings, ScatteringChildRefs,
    ScatteringModel, ScatteringNode, TriangleDistributionEntry, Vertex, Viewport, INVALID_INDEX,
};

#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub camera: Camera,
    pub viewport: Viewport,
    pub output: Output,
    pub render_settings: RenderSettings,
    pub point_lights: Vec<PointLight>,
    pub area_lights: Vec<AreaLight>,
    pub triangle_distributions: Vec<TriangleDistributionEntry>,
    pub lights: Vec<LightRecord>,
    pub light_bounds: Vec<LightBounds>,
    pub light_bvh: LightBVH,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<Instance>,
    pub materials: Vec<Material>,
    pub diffuse_bxdf_data: Vec<DiffuseMaterialData>,
    pub dielectric_bxdf_data: Vec<DielectricMaterialData>,
    pub scattering_models: Vec<ScatteringModel>,
    pub scattering_nodes: Vec<ScatteringNode>,
    pub scattering_child_refs: ScatteringChildRefs,
}

impl Scene {
    pub fn validate_scattering_models(&self) -> Result<(), crate::util::error::PbrtError> {
        validate_scattering_graph(
            &self.scattering_models,
            &self.scattering_nodes,
            &self.scattering_child_refs,
        )
    }
}

pub fn validate_scattering_graph(
    models: &[ScatteringModel],
    nodes: &[ScatteringNode],
    child_refs: &ScatteringChildRefs,
) -> Result<(), crate::util::error::PbrtError> {
    for (model_index, model) in models.iter().enumerate() {
        if model.surface_root != INVALID_INDEX {
            validate_node(
                nodes,
                child_refs,
                model.surface_root,
                model_index,
                &mut Vec::new(),
            )?;
        }
        if model.bssrdf_root != INVALID_INDEX {
            return Err(crate::util::error::PbrtError::error(&format!(
                "Flat scattering model {model_index} references unsupported BSSRDF root.",
            )));
        }
    }
    Ok(())
}

fn validate_node(
    nodes: &[ScatteringNode],
    child_refs: &ScatteringChildRefs,
    node_id: u32,
    model_index: usize,
    stack: &mut Vec<u32>,
) -> Result<(), crate::util::error::PbrtError> {
    let node = nodes.get(node_id as usize).ok_or_else(|| {
        crate::util::error::PbrtError::error(&format!(
            "Flat scattering model {model_index} references an invalid node {node_id}.",
        ))
    })?;
    if stack.contains(&node_id) {
        return Err(crate::util::error::PbrtError::error(&format!(
            "Cycle detected in flat scattering model {model_index} at node {node_id}.",
        )));
    }
    let end = node
        .child_offset
        .checked_add(node.child_count)
        .ok_or_else(|| {
            crate::util::error::PbrtError::error("Scattering child range overflowed.")
        })?;
    if end as usize > child_refs.node_ids.len() {
        return Err(crate::util::error::PbrtError::error(&format!(
            "Flat scattering node {node_id} has an invalid child range.",
        )));
    }
    stack.push(node_id);
    for child in &child_refs.node_ids[node.child_offset as usize..end as usize] {
        validate_node(nodes, child_refs, *child, model_index, stack)?;
    }
    stack.pop();
    Ok(())
}
