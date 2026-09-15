/// The kind of value referenced by a flattened material or light parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttributeKind {
    Scalar,
    Spectrum,
    Texture,
    /// Index of another material in the Flat IR material table.
    Material,
    /// Spectrum texture evaluated with v4's unbounded RGB conversion.
    TextureUnbounded,
}

/// A reference into the scene-wide attribute tables.
///
/// The name is retained in Flat IR for diagnostics and CPU-side evaluation;
/// WebGPU uploads only `kind` and `index`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttributeRef {
    pub kind: AttributeKind,
    pub index: u32,
    pub name: String,
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
