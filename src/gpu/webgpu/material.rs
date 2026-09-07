use crate::util::error::PbrtError;

use super::abi::{AttributeRef, MaterialRecord};
use crate::gpu::ir::flat;

pub struct MaterialTable {
    pub records: Vec<MaterialRecord>,
    pub attributes: Vec<AttributeRef>,
}

impl MaterialTable {
    pub fn from_flat(scene: &flat::Scene) -> Result<Self, PbrtError> {
        scene.validate_scattering_models()?;
        for node in &scene.scattering_nodes {
            let valid = match node.kind.as_str() {
                "diffuse" | "dielectric" | "thindielectric" | "conductor" => node.child_count == 0,
                "layered" => {
                    if node.child_count != 2 {
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
        let mut attributes = Vec::new();
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
                scene
                    .scattering_nodes
                    .get(model.surface_root as usize)
                    .ok_or_else(|| {
                        PbrtError::error("Material references an invalid surface root.")
                    })?;
                let offset = attributes.len() as u32;
                for attr in &material.attributes {
                    let kind = match attr.kind {
                        flat::AttributeKind::Scalar => 0,
                        flat::AttributeKind::Spectrum => 1,
                        flat::AttributeKind::Texture => 2,
                    };
                    attributes.push(AttributeRef {
                        kind,
                        index: attr.index,
                    });
                }
                Ok(MaterialRecord {
                    kind_tag: MaterialKind::from_flat(&material.kind)?.tag(),
                    attribute_offset: offset,
                    attribute_count: material.attributes.len() as u32,
                    scattering_model: material.scattering_model,
                })
            })
            .collect::<Result<Vec<_>, PbrtError>>()?;
        Ok(Self {
            records,
            attributes,
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
    ThinDielectric,
    Conductor,
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
            Self::Conductor => 6,
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
            "conductor" => Ok(Self::Conductor),
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
        "conductor" => Ok(4),
        "layered" => Ok(2),
        other => Err(PbrtError::error(&format!(
            "Unsupported initial WebGPU scattering node kind: {other}."
        ))),
    }
}
