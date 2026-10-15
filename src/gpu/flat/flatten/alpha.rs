use std::sync::Arc;

use crate::gpu::flat::AlphaMask;
use crate::gpu::node::{AlphaMask as NodeAlphaMask, TextureComponent, TextureKind};
use crate::util::error::PbrtError;

use super::FlatBuilder;

pub fn flatten_alpha(
    alpha: &NodeAlphaMask,
    builder: &mut FlatBuilder,
) -> Result<AlphaMask, PbrtError> {
    match alpha {
        NodeAlphaMask::Opaque => Ok(AlphaMask::Opaque),
        NodeAlphaMask::Constant(value) => {
            if !value.is_finite() {
                return Err(PbrtError::error("Shape alpha must be finite."));
            }
            Ok(AlphaMask::Constant(*value))
        }
        NodeAlphaMask::Texture(texture) => {
            if texture.components.iter().any(|component| {
                matches!(component,
                TextureComponent::Texture(texture) if texture.kind != TextureKind::Float)
            }) {
                return Err(PbrtError::error("Shape alpha texture must be Float."));
            }
            Ok(AlphaMask::Texture(
                builder.textures.intern_root(Arc::clone(texture), 0)?,
            ))
        }
    }
}
