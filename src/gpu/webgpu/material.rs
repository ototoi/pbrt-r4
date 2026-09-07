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
                validate_material_attributes(material)?;
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

fn validate_material_attributes(material: &flat::Material) -> Result<(), PbrtError> {
    let expected = match material.kind.as_str() {
        "diffuse" => &[(0, flat::AttributeKind::Spectrum)][..],
        "dielectric" | "thindielectric" => &[(0, flat::AttributeKind::Spectrum)][..],
        "conductor" => &[
            (0, flat::AttributeKind::Spectrum),
            (1, flat::AttributeKind::Spectrum),
            (2, flat::AttributeKind::Scalar),
        ][..],
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
            .any(|(actual, (_, expected_kind))| actual.kind != *expected_kind)
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
    Conductor,
}

impl MaterialKind {
    pub fn tag(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::Uv => 1,
            Self::Diffuse | Self::Lambert => 2,
            Self::Dielectric => 3,
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
            "coateddiffuse" => Err(PbrtError::error(
                "coateddiffuse is not supported by the WebGPU backend.",
            )),
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
        other => Err(PbrtError::error(&format!(
            "Unsupported initial WebGPU scattering node kind: {other}."
        ))),
    }
}
