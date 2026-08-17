use crate::util::base::{gamma_correct, inverse_gamma_correct, Float};
use crate::util::error::PbrtError;

/// The color encoding used to convert byte image samples to and from linear
/// values. This is the CPU-side counterpart of pbrt-v4's ColorEncoding.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ColorEncoding {
    Linear,
    SRgb,
    Gamma(Float),
}

impl ColorEncoding {
    pub fn parse(name: &str) -> Result<Self, PbrtError> {
        match name {
            "linear" => Ok(Self::Linear),
            "sRGB" | "srgb" => Ok(Self::SRgb),
            _ => {
                let mut parts = name.split_whitespace();
                if parts.next() != Some("gamma") {
                    return Err(PbrtError::error(&format!(
                        "Unknown color encoding \"{}\".",
                        name
                    )));
                }
                let gamma = parts.next().ok_or_else(|| {
                    PbrtError::error(&format!(
                        "Color encoding \"{}\" requires a gamma value.",
                        name
                    ))
                })?;
                if parts.next().is_some() {
                    return Err(PbrtError::error(&format!(
                        "Invalid color encoding \"{}\".",
                        name
                    )));
                }
                let gamma = gamma.parse::<Float>().map_err(|_| {
                    PbrtError::error(&format!("Invalid color encoding gamma \"{}\".", gamma))
                })?;
                if gamma <= 0.0 {
                    return Err(PbrtError::error(&format!(
                        "Color encoding gamma must be positive: {}.",
                        gamma
                    )));
                }
                Ok(Self::Gamma(gamma))
            }
        }
    }

    pub fn from_legacy_gamma(gamma: bool) -> Self {
        if gamma {
            Self::SRgb
        } else {
            Self::Linear
        }
    }

    pub fn to_linear(self, encoded: Float) -> Float {
        match self {
            Self::Linear => encoded,
            Self::SRgb => inverse_gamma_correct(encoded),
            Self::Gamma(gamma) => encoded.powf(gamma),
        }
    }

    pub fn from_linear(self, linear: Float) -> Float {
        match self {
            Self::Linear => linear,
            Self::SRgb => gamma_correct(linear),
            Self::Gamma(gamma) => linear.max(0.0).powf(1.0 / gamma),
        }
    }

    pub fn name(self) -> String {
        match self {
            Self::Linear => "linear".to_string(),
            Self::SRgb => "sRGB".to_string(),
            Self::Gamma(gamma) => format!("gamma {}", gamma),
        }
    }
}
