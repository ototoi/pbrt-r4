//! Compilation of Node IR texture roots into backend-independent resources.

use std::collections::HashMap;
use std::sync::Arc;

use crate::gpu::node::TextureNode;
use crate::util::error::PbrtError;
use crate::util::spectrum::SpectrumType;

use super::typed_program::TypedTextureProgram;

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

#[derive(Clone)]
pub enum TextureRoot {
    Float {
        program: u32,
    },
    Spectrum {
        program: u32,
        spectrum_type: SpectrumType,
    },
}

pub struct TextureLibrary {
    pub programs: Vec<TypedTextureProgram>,
    pub roots: Vec<TextureRoot>,
}

/// Compile exported texture roots while keeping root interpretation separate
/// from the shared graph program.
pub fn compile_texture_library(roots: &[TextureRootSpec]) -> Result<TextureLibrary, PbrtError> {
    let mut programs = Vec::with_capacity(roots.len());
    let mut compiled_roots = Vec::with_capacity(roots.len());
    let mut programs_by_root = HashMap::new();

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
            let program = u32::try_from(programs.len())
                .map_err(|_| PbrtError::error("Texture program table exceeds u32."))?;
            programs.push(TypedTextureProgram::compile(node)?);
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
    })
}
