use std::sync::Arc;

use crate::util::error::PbrtError;
use crate::util::spectrum::SpectrumType;

use super::image::{ImageView, Mipmap};
use super::program::{Instruction, TypedTextureProgram, ValueType};

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

impl TextureLibrary {
    pub fn validate(&self) -> Result<(), PbrtError> {
        for root in &self.roots {
            let program = match root {
                TextureRoot::Float { program } | TextureRoot::Spectrum { program, .. } => *program,
            };
            if program as usize >= self.programs.len() {
                return Err(PbrtError::error(
                    "Texture root references an invalid program.",
                ));
            }
        }
        for program in &self.programs {
            if program.instructions.len() != program.slot_types.len()
                || program.instructions.len() != program.slot_last_use.len()
            {
                return Err(PbrtError::error(
                    "Texture program instruction and slot tables differ in length.",
                ));
            }
            if program.result as usize >= program.instructions.len() {
                return Err(PbrtError::error(
                    "Texture program has an invalid result slot.",
                ));
            }
            for instruction in &program.instructions {
                let valid_slot = |slot: u32| (slot as usize) < program.instructions.len();
                match instruction {
                    Instruction::ConstantFloat { .. }
                    | Instruction::ConstantRgb { .. }
                    | Instruction::SampleImage { .. } => {}
                    Instruction::Scale {
                        dst, input, scale, ..
                    } => {
                        if !valid_slot(*dst)
                            || !valid_slot(*input)
                            || scale.is_some_and(|slot| !valid_slot(slot))
                        {
                            return Err(PbrtError::error(
                                "Texture scale references an invalid slot.",
                            ));
                        }
                        if *input >= *dst || scale.is_some_and(|slot| slot >= *dst) {
                            return Err(PbrtError::error(
                                "Texture scale operands are not in post-order.",
                            ));
                        }
                        if program.slot_types[*input as usize] != program.slot_types[*dst as usize]
                        {
                            return Err(PbrtError::error(
                                "Texture scale input has an incompatible value type.",
                            ));
                        }
                        if scale.is_some_and(|slot| {
                            program.slot_types[slot as usize] != ValueType::Float
                        }) {
                            return Err(PbrtError::error(
                                "Texture scale factor must be a float value.",
                            ));
                        }
                    }
                    Instruction::Mix {
                        first,
                        second,
                        amount,
                        ..
                    } => {
                        if !valid_slot(*first)
                            || !valid_slot(*second)
                            || amount.is_some_and(|slot| !valid_slot(slot))
                        {
                            return Err(PbrtError::error(
                                "Texture mix references an invalid slot.",
                            ));
                        }
                    }
                    Instruction::Procedural { operands, .. } => {
                        if operands.iter().any(|slot| !valid_slot(*slot)) {
                            return Err(PbrtError::error(
                                "Texture procedural operation references an invalid slot.",
                            ));
                        }
                    }
                }
                if let Instruction::SampleImage { image_view, .. } = instruction {
                    let view = self.image_views.get(*image_view as usize).ok_or_else(|| {
                        PbrtError::error("Texture instruction references an invalid image view.")
                    })?;
                    if view.mipmap as usize >= self.mipmaps.len() {
                        return Err(PbrtError::error(
                            "Texture image view references an invalid mipmap.",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
