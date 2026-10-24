use crate::util::error::PbrtError;

use super::abi::{AttributeRef, MaterialNode, MaterialRoot};
use crate::gpu::flat;

pub struct MaterialTable {
    pub roots: Vec<MaterialRoot>,
    pub nodes: Vec<MaterialNode>,
    pub attributes: Vec<AttributeRef>,
}

impl MaterialTable {
    pub fn from_flat(scene: &flat::Scene) -> Result<Self, PbrtError> {
        scene.materials.validate()?;
        let mut attributes = Vec::new();
        let nodes = scene
            .materials
            .nodes
            .iter()
            .map(|material| {
                let offset = attributes.len() as u32;
                for attr in &material.attributes {
                    let kind = match attr.kind {
                        flat::AttributeKind::Scalar => 0,
                        flat::AttributeKind::Spectrum => 1,
                        flat::AttributeKind::Texture => 2,
                        flat::AttributeKind::Measured => 3,
                    };
                    attributes.push(AttributeRef {
                        kind,
                        index: attr.index,
                    });
                }
                MaterialNode {
                    kind: MaterialKind::from(material.kind).tag(),
                    attribute_offset: offset,
                    attribute_count: material.attributes.len() as u32,
                    parent: material.parent,
                    parent_slot: material.parent_slot,
                    child0: material.child0,
                    child1: material.child1,
                    displacement_texture_root: material.displacement_texture_root,
                    bssrdf_index: material.bssrdf_index,
                    padding: [0; 3],
                }
            })
            .collect();
        let roots = scene
            .materials
            .roots
            .iter()
            .map(|layout| MaterialRoot {
                node_offset: layout.node_offset,
                node_count: layout.node_count,
            })
            .collect();
        Ok(Self {
            roots,
            nodes,
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
    ThinDielectric,
    DiffuseTransmission,
    // Reserved ABI tag retained for numeric stability; never emitted by Flat IR.
    UnusedConductor,
    ConductorEtaK,
    ConductorReflectance,
    Mix,
    CoatedDiffuse,
    CoatedConductor,
    Measured,
    Subsurface,
}

impl MaterialKind {
    pub fn tag(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::Uv => 1,
            Self::Diffuse | Self::Lambert => 2,
            Self::Dielectric => 3,
            Self::ThinDielectric => 5,
            Self::DiffuseTransmission => 13,
            Self::UnusedConductor => 6,
            Self::ConductorEtaK => 11,
            Self::ConductorReflectance => 12,
            Self::Mix => 7,
            Self::CoatedDiffuse => 8,
            Self::CoatedConductor => 9,
            Self::Measured => 10,
            Self::Subsurface => 15,
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

impl From<flat::MaterialKind> for MaterialKind {
    fn from(kind: flat::MaterialKind) -> Self {
        match kind {
            flat::MaterialKind::Diffuse => Self::Diffuse,
            flat::MaterialKind::Dielectric => Self::Dielectric,
            flat::MaterialKind::ThinDielectric => Self::ThinDielectric,
            flat::MaterialKind::DiffuseTransmission => Self::DiffuseTransmission,
            flat::MaterialKind::ConductorEtaK => Self::ConductorEtaK,
            flat::MaterialKind::ConductorReflectance => Self::ConductorReflectance,
            flat::MaterialKind::Mix => Self::Mix,
            flat::MaterialKind::CoatedDiffuse => Self::CoatedDiffuse,
            flat::MaterialKind::CoatedConductor => Self::CoatedConductor,
            flat::MaterialKind::Measured => Self::Measured,
            flat::MaterialKind::Subsurface => Self::Subsurface,
        }
    }
}
