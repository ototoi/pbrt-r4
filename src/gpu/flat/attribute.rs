use super::DenseSpectrum;

/// Scene-wide scalar and dense spectrum tables referenced by flat attributes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AttributeResources {
    /// Values referenced by `AttributeKind::Scalar` indices.
    pub scalars: Vec<f32>,
    /// Values referenced by `AttributeKind::Spectrum` indices, including medium spectra.
    pub spectra: Vec<DenseSpectrum>,
}

/// The kind of value referenced by a flattened material or light parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttributeKind {
    Scalar,
    Spectrum,
    Texture,
    Measured,
}

/// A reference into a scene-wide Flat IR resource table.
///
/// The name is retained in Flat IR for diagnostics and CPU-side evaluation;
/// WebGPU uploads only `kind` and `index`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttributeRef {
    /// Selects the table addressed by `index`.
    pub kind: AttributeKind,
    /// Index in the table selected by `kind`.
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
