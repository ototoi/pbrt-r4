#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub kind: String,
    pub data: MaterialData,
    pub source_kind: String,
    pub source_data: MaterialSourceData,
    pub scattering_model: u32,
    pub attributes: Vec<AttributeRef>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttributeKind {
    Scalar,
    Spectrum,
    Texture,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttributeRef {
    pub kind: AttributeKind,
    pub index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectrumValue(pub [f32; 4]);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AttributeTables {
    pub scalars: Vec<f32>,
    pub spectra: Vec<SpectrumValue>,
    pub textures: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum UnsupportedTexturePolicy {
    #[default]
    Error,
    DiagnosticMagenta,
}

impl UnsupportedTexturePolicy {
    pub fn from_environment() -> Result<Self, crate::util::error::PbrtError> {
        match std::env::var("PBRT_R4_GPU_UNSUPPORTED_TEXTURE") {
            Ok(value) if value == "magenta" => Ok(Self::DiagnosticMagenta),
            Ok(value) if value == "error" => Ok(Self::Error),
            Ok(value) => Err(crate::util::error::PbrtError::error(&format!(
                "PBRT_R4_GPU_UNSUPPORTED_TEXTURE must be 'error' or 'magenta', got '{value}'."
            ))),
            Err(std::env::VarError::NotPresent) => Ok(Self::Error),
            Err(std::env::VarError::NotUnicode(_)) => Err(crate::util::error::PbrtError::error(
                "PBRT_R4_GPU_UNSUPPORTED_TEXTURE must be valid UTF-8.",
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MaterialData {
    Diffuse(DiffuseMaterialData),
    Dielectric(DielectricMaterialData),
    ThinDielectric(DielectricMaterialData),
    Layered(LayeredBxDFData),
    Unsupported,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MaterialSourceData {
    Diffuse(DiffuseMaterialSourceData),
    Dielectric(DielectricMaterialSourceData),
    ThinDielectric(DielectricMaterialSourceData),
    Layered(LayeredMaterialSourceData),
    Unsupported,
}

pub const EVENT_REFLECTION: u32 = 1 << 0;
pub const EVENT_TRANSMISSION: u32 = 1 << 1;
pub const EVENT_DIFFUSE: u32 = 1 << 2;
pub const EVENT_GLOSSY: u32 = 1 << 3;
pub const EVENT_SPECULAR: u32 = 1 << 4;

#[derive(Clone, Debug, PartialEq)]
pub struct DiffuseMaterialData {
    pub reflectance: [f32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct DielectricMaterialData {
    pub eta: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiffuseMaterialSourceData {
    pub reflectance: [f32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct DielectricMaterialSourceData {
    pub eta: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayeredBxDFData {
    pub thickness: f32,
    pub albedo: [f32; 3],
    pub g: f32,
    pub max_depth: u32,
    pub n_samples: u32,
    pub two_sided: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayeredMaterialSourceData {
    pub thickness: f32,
    pub albedo: [f32; 3],
    pub g: f32,
    pub max_depth: u32,
    pub n_samples: u32,
    pub two_sided: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScatteringModel {
    pub surface_root: u32,
    pub bssrdf_root: u32,
}

/// A resolved view of a scattering model for backend lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedScatteringModel {
    pub root_node: u32,
    pub root_kind: String,
    pub event_flags: u32,
    pub data_index: u32,
    pub child_nodes: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScatteringNode {
    pub kind: String,
    pub event_flags: u32,
    pub data_index: u32,
    pub child_offset: u32,
    pub child_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScatteringChildRefs {
    pub node_ids: Vec<u32>,
}
