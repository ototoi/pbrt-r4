#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub kind: String,
    pub source_kind: String,
    pub scattering_model: u32,
    pub attributes: Vec<AttributeRef>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttributeKind {
    Scalar,
    Spectrum,
    Texture,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttributeRef {
    pub kind: AttributeKind,
    pub index: u32,
    /// Canonical Node IR parameter name retained for diagnostics and CPU-side evaluation.
    pub name: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AttributeTables {
    pub scalars: Vec<f32>,
    pub textures: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum UnsupportedTexturePolicy {
    #[default]
    Error,
    DiagnosticMagenta,
}

impl UnsupportedTexturePolicy {
    pub fn from_value(value: Option<&str>) -> Result<Self, crate::util::error::PbrtError> {
        match value {
            None | Some("error") => Ok(Self::Error),
            Some("magenta") => Ok(Self::DiagnosticMagenta),
            Some(value) => Err(crate::util::error::PbrtError::error(&format!(
                "PBRT_R4_GPU_UNSUPPORTED_TEXTURE must be 'error' or 'magenta', got '{value}'."
            ))),
        }
    }

    pub fn from_environment() -> Result<Self, crate::util::error::PbrtError> {
        match std::env::var("PBRT_R4_GPU_UNSUPPORTED_TEXTURE") {
            Ok(value) => Self::from_value(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::from_value(None),
            Err(std::env::VarError::NotUnicode(_)) => Err(crate::util::error::PbrtError::error(
                "PBRT_R4_GPU_UNSUPPORTED_TEXTURE must be valid UTF-8.",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::UnsupportedTexturePolicy;

    #[test]
    fn unsupported_texture_policy_accepts_only_documented_values() {
        assert_eq!(
            UnsupportedTexturePolicy::from_value(None).unwrap(),
            UnsupportedTexturePolicy::Error
        );
        assert_eq!(
            UnsupportedTexturePolicy::from_value(Some("error")).unwrap(),
            UnsupportedTexturePolicy::Error
        );
        assert_eq!(
            UnsupportedTexturePolicy::from_value(Some("magenta")).unwrap(),
            UnsupportedTexturePolicy::DiagnosticMagenta
        );
        assert!(UnsupportedTexturePolicy::from_value(Some("MAGENTA")).is_err());
    }
}

pub const EVENT_REFLECTION: u32 = 1 << 0;
pub const EVENT_TRANSMISSION: u32 = 1 << 1;
pub const EVENT_DIFFUSE: u32 = 1 << 2;
pub const EVENT_GLOSSY: u32 = 1 << 3;
pub const EVENT_SPECULAR: u32 = 1 << 4;

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
