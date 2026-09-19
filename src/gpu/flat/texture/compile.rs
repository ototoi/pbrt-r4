//! Compilation of Node IR texture roots into Flat IR resources.

use std::collections::HashMap;
use std::sync::Arc;

use crate::gpu::node::TextureNode;
use crate::util::error::PbrtError;
use crate::util::spectrum::SpectrumType;

use super::image::{
    ImageCompiler, ImageDecoder, ImageFilterMode, ImageView, ImageWrapMode, Mipmap,
};
use super::optimize::equivalent_program;
use super::program::{compile_texture_program, Instruction, TypedTextureProgram};

/// A texture entry point exported by a material attribute.
///
/// The enum keeps Float roots from carrying a spectrum interpretation. The
/// color space of RGB values is inferred from the graph's image resources;
/// callers only choose the final spectrum interpretation here.
#[derive(Clone)]
pub enum TextureRootSpec {
    Float {
        node: Arc<TextureNode>,
    },
    Spectrum {
        node: Arc<TextureNode>,
        spectrum_type: SpectrumType,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum TextureRoot {
    Float {
        program: u32,
    },
    Spectrum {
        program: u32,
        spectrum_type: SpectrumType,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextureLibrary {
    pub programs: Vec<TypedTextureProgram>,
    pub roots: Vec<TextureRoot>,
    pub mipmaps: Vec<Arc<Mipmap>>,
    pub image_views: Vec<ImageView>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ImageViewKey {
    mipmap: u32,
    value_type: super::image::ImageValueType,
    swrap: ImageWrapMode,
    twrap: ImageWrapMode,
    filter: ImageFilterMode,
    scale: u32,
    invert: bool,
}

/// Compile exported texture roots while keeping root interpretation separate
/// from the shared graph program.
pub fn compile_texture_library(roots: &[TextureRootSpec]) -> Result<TextureLibrary, PbrtError> {
    let mut programs = Vec::with_capacity(roots.len());
    let mut compiled_roots = Vec::with_capacity(roots.len());
    let mut programs_by_root = HashMap::new();
    let mut image_views = Vec::new();
    let mut image_views_by_key = HashMap::new();
    let mut mipmaps = Vec::new();
    let mut mipmaps_by_identity = HashMap::new();
    let mut image_compiler = ImageCompiler::default();
    let mut image_decoder = ImageDecoder::default();

    for root in roots {
        let (node, spectrum_type) = match root {
            TextureRootSpec::Float { node } => (node, None),
            TextureRootSpec::Spectrum {
                node,
                spectrum_type,
            } => (node, Some(*spectrum_type)),
        };
        let key = Arc::as_ptr(node) as usize;
        let program = if let Some(&program) = programs_by_root.get(&key) {
            program
        } else {
            let mut compiled = compile_texture_program(node, &mut image_decoder)?;
            let mut remap = Vec::with_capacity(compiled.image_views.len());
            for view in &compiled.image_views {
                let source = compiled
                    .mipmaps
                    .get(view.mipmap as usize)
                    .ok_or_else(|| PbrtError::error("Texture view has an invalid mipmap."))?;
                let mipmap = image_compiler.compile(source, view.value_type)?;
                let mipmap_identity = Arc::as_ptr(&mipmap) as usize;
                let mipmap = if let Some(&index) = mipmaps_by_identity.get(&mipmap_identity) {
                    index
                } else {
                    let index = u32::try_from(mipmaps.len())
                        .map_err(|_| PbrtError::error("Texture mipmap table exceeds u32."))?;
                    mipmaps.push(mipmap);
                    mipmaps_by_identity.insert(mipmap_identity, index);
                    index
                };
                let optimized = ImageView {
                    mipmap,
                    ..view.clone()
                };
                let key = ImageViewKey {
                    mipmap: optimized.mipmap,
                    value_type: optimized.value_type,
                    swrap: optimized.swrap,
                    twrap: optimized.twrap,
                    filter: optimized.filter,
                    scale: optimized.scale.to_bits(),
                    invert: optimized.invert,
                };
                let global = if let Some(&global) = image_views_by_key.get(&key) {
                    global
                } else {
                    let global = u32::try_from(image_views.len())
                        .map_err(|_| PbrtError::error("Texture image view table exceeds u32."))?;
                    image_views.push(optimized);
                    image_views_by_key.insert(key, global);
                    global
                };
                remap.push(global);
            }
            for instruction in &mut compiled.program.instructions {
                if let Instruction::SampleImage { image_view, .. } = instruction {
                    *image_view = *remap.get(*image_view as usize).ok_or_else(|| {
                        PbrtError::error("Texture instruction references an invalid image view.")
                    })?;
                }
            }
            let program = if let Some(program) = programs
                .iter()
                .position(|candidate| equivalent_program(candidate, &compiled.program))
            {
                u32::try_from(program)
                    .map_err(|_| PbrtError::error("Texture program table exceeds u32."))?
            } else {
                let program = u32::try_from(programs.len())
                    .map_err(|_| PbrtError::error("Texture program table exceeds u32."))?;
                programs.push(compiled.program);
                program
            };
            programs_by_root.insert(key, program);
            program
        };
        compiled_roots.push(match spectrum_type {
            Some(spectrum_type) => TextureRoot::Spectrum {
                program,
                spectrum_type,
            },
            None => TextureRoot::Float { program },
        });
    }

    Ok(TextureLibrary {
        programs,
        roots: compiled_roots,
        mipmaps,
        image_views,
    })
}
