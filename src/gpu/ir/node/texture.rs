use super::transform::Transform;
use crate::paramdict::ParameterDictionary;
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
    pub mipmap: Option<Arc<Mipmap>>,
}

/// The color space of RGB texels stored by an image texture.
///
/// The texel value returned by texture evaluation is linear RGB in this color
/// space. The color space is retained until the material attribute boundary,
/// where RGB-to-spectrum conversion is performed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpaceId {
    Srgb,
    Aces2065,
    DciP3,
    Rec2020,
}

/// Storage for one immutable image mip level in the Node IR.
///
/// `F16` contains the IEEE-754 binary16 bit pattern in native-endian `u16`
/// values. The WebGPU adapter converts the selected storage to its upload
/// format without changing the Node IR ownership model.
#[derive(Clone, Debug, PartialEq)]
pub enum MipmapLevelData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    U8(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct MipmapLevel {
    pub resolution: [u32; 2],
    pub channels: u32,
    pub data: MipmapLevelData,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mipmap {
    pub levels: Vec<MipmapLevel>,
    pub color_space: Option<ColorSpaceId>,
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
