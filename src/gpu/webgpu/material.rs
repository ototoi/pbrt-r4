use crate::util::error::PbrtError;

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
                        && (node.data_index as usize) < scene.diffuse_bxdf_data.len()
                }
                "dielectric" => {
                    node.child_count == 0
                        && (node.data_index as usize) < scene.dielectric_bxdf_data.len()
                }
                "layered" => {
                    if node.child_count != 2
                        || (node.data_index as usize) >= scene.layered_bxdf_data.len()
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
        Ok(Self {
            records,
            diffuse: scene
                .diffuse_bxdf_data
                .iter()
                .map(|d| DiffuseMaterialData {
                    reflectance: [d.reflectance[0], d.reflectance[1], d.reflectance[2], 0.0],
                })
                .collect(),
            dielectric: scene
                .dielectric_bxdf_data
                .iter()
                .map(|d| DielectricMaterialData {
                    eta: d.eta,
                    padding: [0; 3],
                })
                .collect(),
            layered: scene
                .layered_bxdf_data
                .iter()
                .map(|d| LayeredBxDFData {
                    thickness: d.thickness,
                    g: d.g,
                    max_depth: d.max_depth,
                    n_samples: d.n_samples,
                    albedo: [d.albedo[0], d.albedo[1], d.albedo[2], 0.0],
                    two_sided: u32::from(d.two_sided),
                    padding: [0; 3],
                })
                .collect(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialKind {
    Normal,
    Uv,
    Diffuse,
    Lambert,
    Dielectric,
    Layered,
}

impl MaterialKind {
    pub fn tag(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::Uv => 1,
            Self::Diffuse | Self::Lambert => 2,
            Self::Dielectric => 3,
            Self::Layered => 4,
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
        "layered" => Ok(2),
        other => Err(PbrtError::error(&format!(
            "Unsupported initial WebGPU scattering node kind: {other}."
        ))),
    }
}
