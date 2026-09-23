use crate::util::error::PbrtError;

use super::abi::{AttributeRef, MaterialNode};
use crate::gpu::flat;

pub struct MaterialTable {
    pub nodes: Vec<MaterialNode>,
    pub attributes: Vec<AttributeRef>,
}

impl MaterialTable {
    pub fn from_flat(scene: &flat::Scene) -> Result<Self, PbrtError> {
        let mut attributes = Vec::new();
        let nodes = scene
            .material_nodes
            .iter()
            .map(|material| {
                validate_material_attributes(material)?;
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
                Ok(MaterialNode {
                    kind: MaterialKind::from_flat(&material.kind)?.tag(),
                    attribute_offset: offset,
                    attribute_count: material.attributes.len() as u32,
                    parent: material.parent,
                    parent_slot: material.parent_slot,
                    child0: material.child0,
                    child1: material.child1,
                    padding: 0,
                })
            })
            .collect::<Result<Vec<_>, PbrtError>>()?;
        Ok(Self { nodes, attributes })
    }
}

fn validate_material_attributes(material: &flat::MaterialNode) -> Result<(), PbrtError> {
    let expected = match material.kind.as_str() {
        "diffuse" => &[(0, flat::AttributeKind::Spectrum)][..],
        "dielectric" => &[
            (0, flat::AttributeKind::Spectrum),
            (1, flat::AttributeKind::Scalar),
            (2, flat::AttributeKind::Scalar),
            (3, flat::AttributeKind::Scalar),
        ][..],
        "thindielectric" => &[(0, flat::AttributeKind::Spectrum)][..],
        "diffusetransmission" => &[
            (0, flat::AttributeKind::Spectrum),
            (1, flat::AttributeKind::Spectrum),
            (2, flat::AttributeKind::Scalar),
        ][..],
        "conductor_eta_k" => &[
            (0, flat::AttributeKind::Spectrum),
            (1, flat::AttributeKind::Spectrum),
            (2, flat::AttributeKind::Scalar),
        ][..],
        "conductor_reflectance" => &[
            (0, flat::AttributeKind::Spectrum),
            (1, flat::AttributeKind::Scalar),
        ][..],
        "mix" => &[(0, flat::AttributeKind::Scalar)][..],
        "coateddiffuse" => &[
            (0, flat::AttributeKind::Scalar),
            (1, flat::AttributeKind::Spectrum),
            (2, flat::AttributeKind::Scalar),
            (3, flat::AttributeKind::Scalar),
            (4, flat::AttributeKind::Scalar),
            (5, flat::AttributeKind::Spectrum),
            (6, flat::AttributeKind::Spectrum),
            (7, flat::AttributeKind::Scalar),
            (8, flat::AttributeKind::Scalar),
            (9, flat::AttributeKind::Scalar),
        ][..],
        "coatedconductor" => &[
            (0, flat::AttributeKind::Scalar),
            (1, flat::AttributeKind::Spectrum),
            (2, flat::AttributeKind::Scalar),
            (3, flat::AttributeKind::Scalar),
            (4, flat::AttributeKind::Scalar),
            (5, flat::AttributeKind::Spectrum),
            (6, flat::AttributeKind::Scalar),
            (7, flat::AttributeKind::Scalar),
            (8, flat::AttributeKind::Spectrum),
            (9, flat::AttributeKind::Spectrum),
            (10, flat::AttributeKind::Scalar),
            (11, flat::AttributeKind::Scalar),
            (12, flat::AttributeKind::Scalar),
            (13, flat::AttributeKind::Spectrum),
            (14, flat::AttributeKind::Scalar),
        ][..],
        "measured" => &[(0, flat::AttributeKind::Measured)][..],
        other => {
            return Err(PbrtError::error(&format!(
                "Unsupported WebGPU material kind in attribute validation: {other}."
            )))
        }
    };
    if material.attributes.len() != expected.len()
        || material
            .attributes
            .iter()
            .zip(expected)
            .any(|(actual, (_, expected_kind))| {
                actual.kind != *expected_kind
                    && !(*expected_kind == flat::AttributeKind::Spectrum
                        && actual.kind == flat::AttributeKind::Texture)
                    && !(*expected_kind == flat::AttributeKind::Scalar
                        && actual.kind == flat::AttributeKind::Texture)
            })
    {
        let actual = material
            .attributes
            .iter()
            .map(|attribute| {
                format!(
                    "{}:{}",
                    attribute.name,
                    format_attribute_kind(attribute.kind)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(PbrtError::error(&format!(
            "Material \"{}\" kind \"{}\" has invalid GPU attributes: [{actual}].",
            material.source_kind, material.kind
        )));
    }
    Ok(())
}

fn format_attribute_kind(kind: flat::AttributeKind) -> &'static str {
    match kind {
        flat::AttributeKind::Scalar => "scalar",
        flat::AttributeKind::Spectrum => "spectrum",
        flat::AttributeKind::Texture => "texture",
        flat::AttributeKind::Measured => "measured",
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
        }
    }

    pub fn from_flat(kind: &str) -> Result<Self, PbrtError> {
        match kind {
            "normal" => Ok(Self::Normal),
            "uv" => Ok(Self::Uv),
            "diffuse" => Ok(Self::Diffuse),
            "lambert" => Ok(Self::Lambert),
            "dielectric" => Ok(Self::Dielectric),
            "thindielectric" => Ok(Self::ThinDielectric),
            "diffusetransmission" => Ok(Self::DiffuseTransmission),
            "conductor_eta_k" => Ok(Self::ConductorEtaK),
            "conductor_reflectance" => Ok(Self::ConductorReflectance),
            "mix" => Ok(Self::Mix),
            "coateddiffuse" => Ok(Self::CoatedDiffuse),
            "coatedconductor" => Ok(Self::CoatedConductor),
            "measured" => Ok(Self::Measured),
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
