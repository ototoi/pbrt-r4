use crate::util::error::PbrtError;

use super::abi::{DielectricMaterialData, DiffuseMaterialData, MaterialRecord, INVALID_INDEX};
use crate::gpu::ir::flat;

pub struct MaterialTable {
    pub records: Vec<MaterialRecord>,
    pub diffuse: Vec<DiffuseMaterialData>,
    pub dielectric: Vec<DielectricMaterialData>,
}

impl MaterialTable {
    pub fn from_flat(materials: &[flat::Material]) -> Result<Self, PbrtError> {
        let mut table = Self {
            records: Vec::with_capacity(materials.len()),
            diffuse: Vec::new(),
            dielectric: Vec::new(),
        };
        for material in materials {
            let kind = MaterialKind::from_flat(&material.kind)?;
            let data_index = match &material.data {
                flat::MaterialData::Diffuse(data) => {
                    if kind != MaterialKind::Diffuse {
                        return Err(PbrtError::error(
                            "Flat diffuse material data has a non-diffuse kind.",
                        ));
                    }
                    let index = u32::try_from(table.diffuse.len()).map_err(|_| {
                        PbrtError::error("WebGPU diffuse-material table exceeds u32.")
                    })?;
                    table.diffuse.push(DiffuseMaterialData {
                        reflectance: [
                            data.reflectance[0],
                            data.reflectance[1],
                            data.reflectance[2],
                            0.0,
                        ],
                    });
                    index
                }
                flat::MaterialData::Dielectric(data) => {
                    if kind != MaterialKind::Dielectric {
                        return Err(PbrtError::error(
                            "Flat dielectric material data has a non-dielectric kind.",
                        ));
                    }
                    let index = u32::try_from(table.dielectric.len()).map_err(|_| {
                        PbrtError::error("WebGPU dielectric-material table exceeds u32.")
                    })?;
                    table.dielectric.push(DielectricMaterialData {
                        eta: data.eta,
                        padding: [0; 3],
                    });
                    index
                }
                flat::MaterialData::Unsupported => INVALID_INDEX,
            };
            table.records.push(MaterialRecord {
                kind_tag: kind.tag(),
                data_index,
                padding: [0; 2],
            });
        }
        Ok(table)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialKind {
    Normal,
    Uv,
    Diffuse,
    Lambert,
    Dielectric,
}

impl MaterialKind {
    pub fn tag(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::Uv => 1,
            Self::Diffuse | Self::Lambert => 2,
            Self::Dielectric => 3,
        }
    }

    pub fn from_flat(kind: &str) -> Result<Self, PbrtError> {
        match kind {
            "normal" => Ok(Self::Normal),
            "uv" => Ok(Self::Uv),
            "diffuse" => Ok(Self::Diffuse),
            "lambert" => Ok(Self::Lambert),
            "dielectric" => Ok(Self::Dielectric),
            other => Err(PbrtError::error(&format!(
                "Unsupported initial WebGPU material kind: {other}."
            ))),
        }
    }

    pub fn from_debug_environment() -> Result<Self, PbrtError> {
        match std::env::var("PBRT_R4_GPU_DEBUG_MATERIAL") {
            Ok(kind) => match kind.as_str() {
                "normal" => Ok(Self::Normal),
                "uv" => Ok(Self::Uv),
                "lambert" => Ok(Self::Lambert),
                other => Err(PbrtError::error(&format!(
                    "Unsupported WebGPU debug material kind: {other}. Use normal, uv, or lambert."
                ))),
            },
            Err(std::env::VarError::NotPresent) => Ok(Self::Lambert),
            Err(std::env::VarError::NotUnicode(_)) => Err(PbrtError::error(
                "PBRT_R4_GPU_DEBUG_MATERIAL must be valid UTF-8.",
            )),
        }
    }
}
