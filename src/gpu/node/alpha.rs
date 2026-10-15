//! Shape coverage, independent of the surface BSDF (v4 GeometricPrimitive::alpha).
use std::sync::Arc;

use super::TextureNode;

#[derive(Clone, Default)]
pub enum AlphaMask {
    #[default]
    Opaque,
    Constant(f32),
    Texture(Arc<TextureNode>),
}
