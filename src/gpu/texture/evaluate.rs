//! Reference evaluator for the backend-independent typed texture program.

use crate::util::error::PbrtError;

use super::typed_program::{Instruction, TypedTextureProgram};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextureValue {
    Float(f32),
    LinearRgb([f32; 3]),
}

pub fn evaluate_texture_program(program: &TypedTextureProgram) -> Result<TextureValue, PbrtError> {
    let mut values = Vec::with_capacity(program.slot_types.len());
    for instruction in &program.instructions {
        let value = match instruction {
            Instruction::ConstantFloat { value, .. } => TextureValue::Float(*value),
            Instruction::ConstantRgb { value, .. } => TextureValue::LinearRgb(*value),
            Instruction::Scale { input, factor, .. } => match values.get(*input as usize) {
                Some(TextureValue::Float(value)) => TextureValue::Float(value * factor),
                Some(TextureValue::LinearRgb(value)) => {
                    TextureValue::LinearRgb(value.map(|value| value * factor))
                }
                None => return Err(invalid_slot(*input)),
            },
            Instruction::Mix {
                first,
                second,
                amount,
                constant_amount,
                ..
            } => {
                let amount = match amount {
                    Some(slot) => match values.get(*slot as usize) {
                        Some(TextureValue::Float(value)) => *value,
                        Some(TextureValue::LinearRgb(_)) => {
                            return Err(PbrtError::error(
                                "Texture mix amount must be a float value.",
                            ));
                        }
                        None => return Err(invalid_slot(*slot)),
                    },
                    None => *constant_amount,
                };
                match (values.get(*first as usize), values.get(*second as usize)) {
                    (Some(TextureValue::Float(first)), Some(TextureValue::Float(second))) => {
                        TextureValue::Float(first * (1.0 - amount) + second * amount)
                    }
                    (
                        Some(TextureValue::LinearRgb(first)),
                        Some(TextureValue::LinearRgb(second)),
                    ) => TextureValue::LinearRgb(std::array::from_fn(|index| {
                        first[index] * (1.0 - amount) + second[index] * amount
                    })),
                    (Some(_), Some(_)) => {
                        return Err(PbrtError::error(
                            "Texture mix operands must have matching value types.",
                        ));
                    }
                    (None, _) => return Err(invalid_slot(*first)),
                    (_, None) => return Err(invalid_slot(*second)),
                }
            }
            Instruction::SampleImage { .. } => {
                return Err(PbrtError::error(
                    "Reference texture evaluator does not sample images yet.",
                ));
            }
            Instruction::Procedural { name, .. } => {
                return Err(PbrtError::error(&format!(
                    "Reference texture evaluator does not support procedural texture \"{name}\" yet."
                )));
            }
        };
        values.push(value);
    }
    values
        .get(program.result as usize)
        .copied()
        .ok_or_else(|| PbrtError::error("Texture program has no result value."))
}

fn invalid_slot(slot: u32) -> PbrtError {
    PbrtError::error(&format!("Texture program references invalid slot {slot}."))
}
