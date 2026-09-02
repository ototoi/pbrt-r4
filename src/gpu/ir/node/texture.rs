use super::transform::Transform;
use crate::paramdict::ParameterDictionary;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TextureKind {
    Float,
    Spectrum,
}

#[derive(Clone)]
pub struct Texture {
    pub name: String,
    pub kind: TextureKind,
    pub params: ParameterDictionary,
    pub transform: Transform,
}
