use crate::util::error::PbrtError;
use bytemuck::Zeroable;

use super::abi::{DielectricMaterialData, DiffuseMaterialData, LayeredBxDFData, MaterialRecord};
use crate::gpu::ir::flat;

pub struct MaterialTable {
    pub records: Vec<MaterialRecord>,
    pub diffuse: Vec<DiffuseMaterialData>,
    pub dielectric: Vec<DielectricMaterialData>,
    pub layered: Vec<LayeredBxDFData>,
}

impl MaterialTable {
    pub fn from_flat(scene: &flat::Scene) -> Result<Self, PbrtError> {
        scene.validate_scattering_models()?;
        for node in &scene.scattering_nodes {
            let valid = match node.kind.as_str() {
                "diffuse" => {
                    node.child_count == 0
                        && (node.data_index as usize) < node_table_len(scene, "diffuse")
                }
                "dielectric" => {
                    node.child_count == 0
                        && (node.data_index as usize) < dielectric_table_len(scene)
                }
                "thindielectric" => {
                    node.child_count == 0
                        && (node.data_index as usize) < dielectric_table_len(scene)
                }
                "layered" => {
                    if node.child_count != 2
                        || (node.data_index as usize) >= node_table_len(scene, "layered")
                    {
                        false
                    } else {
                        let children = node.child_offset.checked_add(2).and_then(|end| {
                            scene
                                .scattering_child_refs
                                .node_ids
                                .get(node.child_offset as usize..end as usize)
                        });
                        children.is_some_and(|children| {
                            scene
                                .scattering_nodes
                                .get(children[0] as usize)
                                .is_some_and(|n| n.kind == "dielectric")
                                && scene
                                    .scattering_nodes
                                    .get(children[1] as usize)
                                    .is_some_and(|n| n.kind == "diffuse")
                        })
                    }
                }
                _ => false,
            };
            if !valid {
                return Err(PbrtError::error(
                    "Invalid or unsupported WebGPU scattering node data or children.",
                ));
            }
        }
        let records = scene
            .materials
            .iter()
            .map(|material| {
                let model = scene
                    .scattering_models
                    .get(material.scattering_model as usize)
                    .ok_or_else(|| {
                        PbrtError::error("Material references an invalid scattering model.")
                    })?;
                let node = scene
                    .scattering_nodes
                    .get(model.surface_root as usize)
                    .ok_or_else(|| {
                        PbrtError::error("Material references an invalid surface root.")
                    })?;
                Ok(MaterialRecord {
                    kind_tag: MaterialKind::from_flat(&material.kind)?.tag(),
                    data_index: node.data_index,
                    scattering_model: material.scattering_model,
                    padding: 0,
                })
            })
            .collect::<Result<Vec<_>, PbrtError>>()?;
        let (diffuse, dielectric, layered) = build_abi_tables(scene)?;
        Ok(Self {
            records,
            diffuse,
            dielectric,
            layered,
        })
    }
}

fn build_abi_tables(
    scene: &flat::Scene,
) -> Result<
    (
        Vec<DiffuseMaterialData>,
        Vec<DielectricMaterialData>,
        Vec<LayeredBxDFData>,
    ),
    PbrtError,
> {
    let mut diffuse = vec![DiffuseMaterialData::zeroed(); node_table_len(scene, "diffuse")];
    let dielectric_len = dielectric_table_len(scene);
    let mut dielectric = vec![DielectricMaterialData::zeroed(); dielectric_len];
    let mut layered = vec![LayeredBxDFData::zeroed(); node_table_len(scene, "layered")];
    for (node_id, node) in scene.scattering_nodes.iter().enumerate() {
        let Some(material) = owner_material(scene, node_id as u32) else {
            return Err(PbrtError::error("Scattering node has no owning material."));
        };
        match node.kind.as_str() {
            "diffuse" => {
                let value = spectrum_attribute(scene, material, "reflectance")?;
                let slot = diffuse.get_mut(node.data_index as usize).ok_or_else(|| {
                    PbrtError::error("Diffuse node references an invalid ABI slot.")
                })?;
                slot.reflectance = [value[0], value[1], value[2], 0.0];
            }
            "dielectric" | "thindielectric" => {
                let eta = scalar_attribute(scene, material, "eta")?;
                let slot = dielectric
                    .get_mut(node.data_index as usize)
                    .ok_or_else(|| {
                        PbrtError::error("Dielectric node references an invalid ABI slot.")
                    })?;
                slot.eta = eta;
            }
            "layered" => {
                let slot = layered.get_mut(node.data_index as usize).ok_or_else(|| {
                    PbrtError::error("Layered node references an invalid ABI slot.")
                })?;
                slot.thickness = scalar_attribute(scene, material, "thickness")?;
                slot.g = scalar_attribute(scene, material, "g")?;
                slot.max_depth = scalar_attribute(scene, material, "maxdepth")? as u32;
                slot.n_samples = scalar_attribute(scene, material, "nsamples")? as u32;
                let albedo = spectrum_attribute(scene, material, "albedo")?;
                slot.albedo = [albedo[0], albedo[1], albedo[2], 0.0];
                slot.two_sided = scalar_attribute(scene, material, "twosided")? as u32;
            }
            _ => return Err(PbrtError::error("Unsupported scattering node kind.")),
        }
    }
    Ok((diffuse, dielectric, layered))
}

fn node_table_len(scene: &flat::Scene, kind: &str) -> usize {
    scene
        .scattering_nodes
        .iter()
        .filter(|node| node.kind == kind)
        .count()
}

fn dielectric_table_len(scene: &flat::Scene) -> usize {
    node_table_len(scene, "dielectric")
        .checked_add(node_table_len(scene, "thindielectric"))
        .unwrap_or(usize::MAX)
}

fn owner_material<'a>(scene: &'a flat::Scene, node_id: u32) -> Option<&'a flat::Material> {
    scene.materials.iter().find(|material| {
        let Some(model) = scene
            .scattering_models
            .get(material.scattering_model as usize)
        else {
            return false;
        };
        if model.surface_root == node_id {
            return true;
        }
        let Some(root) = scene.scattering_nodes.get(model.surface_root as usize) else {
            return false;
        };
        let start = root.child_offset as usize;
        let end = start.saturating_add(root.child_count as usize);
        scene
            .scattering_child_refs
            .node_ids
            .get(start..end)
            .is_some_and(|children| children.contains(&node_id))
    })
}

fn scalar_attribute(
    scene: &flat::Scene,
    material: &flat::Material,
    name: &str,
) -> Result<f32, PbrtError> {
    let attribute = material
        .attributes
        .iter()
        .find(|attribute| attribute.name == name && attribute.kind == flat::AttributeKind::Scalar)
        .ok_or_else(|| PbrtError::error(&format!("Missing scalar material attribute '{name}'.")))?;
    scene
        .attribute_tables
        .scalars
        .get(attribute.index as usize)
        .copied()
        .ok_or_else(|| PbrtError::error("Scalar material attribute index is invalid."))
}

fn spectrum_attribute(
    scene: &flat::Scene,
    material: &flat::Material,
    name: &str,
) -> Result<[f32; 3], PbrtError> {
    let attribute = material
        .attributes
        .iter()
        .find(|attribute| attribute.name == name && attribute.kind == flat::AttributeKind::Spectrum)
        .ok_or_else(|| {
            PbrtError::error(&format!("Missing spectrum material attribute '{name}'."))
        })?;
    let value = scene
        .attribute_tables
        .spectra
        .get(attribute.index as usize)
        .ok_or_else(|| PbrtError::error("Spectrum material attribute index is invalid."))?;
    Ok([value.0[0], value.0[1], value.0[2]])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialKind {
    Normal,
    Uv,
    Diffuse,
    Lambert,
    Dielectric,
    Layered,
    ThinDielectric,
}

impl MaterialKind {
    pub fn tag(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::Uv => 1,
            Self::Diffuse | Self::Lambert => 2,
            Self::Dielectric => 3,
            Self::Layered => 4,
            Self::ThinDielectric => 5,
        }
    }

    pub fn from_flat(kind: &str) -> Result<Self, PbrtError> {
        match kind {
            "normal" => Ok(Self::Normal),
            "uv" => Ok(Self::Uv),
            "diffuse" => Ok(Self::Diffuse),
            "lambert" => Ok(Self::Lambert),
            "dielectric" => Ok(Self::Dielectric),
            "coateddiffuse" => Ok(Self::Layered),
            "thindielectric" => Ok(Self::ThinDielectric),
            other => Err(PbrtError::error(&format!(
                "Unsupported initial WebGPU material kind: {other}."
            ))),
        }
    }

    pub fn from_debug_environment() -> Result<Option<Self>, PbrtError> {
        match std::env::var("PBRT_R4_GPU_DEBUG_MATERIAL") {
            Ok(kind) => match kind.as_str() {
                "normal" => Ok(Some(Self::Normal)),
                "uv" => Ok(Some(Self::Uv)),
                "lambert" => Ok(Some(Self::Lambert)),
                other => Err(PbrtError::error(&format!(
                    "Unsupported WebGPU debug material kind: {other}. Use normal, uv, or lambert."
                ))),
            },
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(std::env::VarError::NotUnicode(_)) => Err(PbrtError::error(
                "PBRT_R4_GPU_DEBUG_MATERIAL must be valid UTF-8.",
            )),
        }
    }
}

pub fn scattering_node_tag(kind: &str) -> Result<u32, PbrtError> {
    match kind {
        "diffuse" => Ok(0),
        "dielectric" => Ok(1),
        "thindielectric" => Ok(3),
        "layered" => Ok(2),
        other => Err(PbrtError::error(&format!(
            "Unsupported initial WebGPU scattering node kind: {other}."
        ))),
    }
}
