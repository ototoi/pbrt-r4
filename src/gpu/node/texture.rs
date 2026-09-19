use super::transform::Transform;
use crate::paramdict::ParameterDictionary;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureKind {
    Float,
    Spectrum,
}

#[derive(Clone)]
pub struct Texture {
    pub name: String,
    pub kind: TextureKind,
    pub params: ParameterDictionary,
}

impl Texture {
    pub fn image_path(&self) -> Option<PathBuf> {
        if self.name != "imagemap" {
            return None;
        }
        let filename = self.params.get_one_string("filename", "");
        (!filename.is_empty()).then(|| filename.into())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvMapping {
    pub uscale: f32,
    pub vscale: f32,
    pub udelta: f32,
    pub vdelta: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TextureMapping {
    Uv(UvMapping),
    Planar(Transform),
    Spherical(Transform),
    Cylindrical(Transform),
    PointTransform(Transform),
}

/// A texture graph node is separate from the Scene Node hierarchy.
#[derive(Clone)]
pub struct TextureNode {
    pub name: String,
    pub components: Vec<TextureComponent>,
    pub children: Vec<Arc<TextureNode>>,
}

#[derive(Clone)]
pub enum TextureComponent {
    Texture(Texture),
    Mapping(TextureMapping),
}

impl TextureNode {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            components: Vec::new(),
            children: Vec::new(),
        }
    }
}
