use super::super::texture::TextureRootSpec;
use super::TextureBuilder;
use crate::gpu::node::{TextureComponent, TextureKind, TextureNode};
use crate::util::error::PbrtError;
use crate::util::spectrum::SpectrumType;
use std::sync::Arc;

impl TextureBuilder {
    pub fn intern_root(
        &mut self,
        texture_node: Arc<TextureNode>,
        spectrum_type: u32,
    ) -> Result<u32, PbrtError> {
        let texture = texture_node
            .components
            .iter()
            .find_map(|component| match component {
                TextureComponent::Texture(texture) => Some(texture),
                TextureComponent::Mapping(_) => None,
            })
            .ok_or_else(|| PbrtError::error("Texture root has no texture component."))?;
        let spectrum_type = match spectrum_type {
            1 => SpectrumType::Unbounded,
            2 => SpectrumType::Illuminant,
            _ => SpectrumType::Albedo,
        };
        let key = (
            Arc::as_ptr(&texture_node) as usize,
            match texture.kind {
                TextureKind::Float => 0,
                TextureKind::Spectrum => match spectrum_type {
                    SpectrumType::Albedo => 1,
                    SpectrumType::Unbounded => 2,
                    SpectrumType::Illuminant => 3,
                },
            },
        );
        if let Some(&index) = self.roots_by_key.get(&key) {
            return Ok(index);
        }
        let index = u32::try_from(self.root_specs.len())
            .map_err(|_| PbrtError::error("Flat texture root table exceeds u32."))?;
        self.root_specs.push(match texture.kind {
            TextureKind::Float => TextureRootSpec::Float { node: texture_node },
            TextureKind::Spectrum => TextureRootSpec::Spectrum {
                node: texture_node,
                spectrum_type,
            },
        });
        self.roots_by_key.insert(key, index);
        Ok(index)
    }
}
