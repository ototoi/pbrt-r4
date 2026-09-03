use crate::util::error::PbrtError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialKind {
    Normal,
    Uv,
}

impl MaterialKind {
    pub fn tag(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::Uv => 1,
        }
    }

    pub fn from_flat(kind: &str) -> Result<Self, PbrtError> {
        match kind {
            "normal" => Ok(Self::Normal),
            "uv" => Ok(Self::Uv),
            other => Err(PbrtError::error(&format!(
                "Unsupported initial WebGPU material kind: {other}."
            ))),
        }
    }
}
