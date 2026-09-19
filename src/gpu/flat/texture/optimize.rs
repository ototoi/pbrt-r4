//! Validation and optimization passes for Flat IR texture programs.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::gpu::node::TextureMapping;
use crate::util::error::PbrtError;

use super::image::{ImageView, Mipmap};
use super::program::{Instruction, TypedTextureProgram, ValueType};

pub fn optimize_texture_program(
    instructions: Vec<Instruction>,
    slot_types: Vec<ValueType>,
    mipmaps: Vec<Arc<Mipmap>>,
    image_views: Vec<ImageView>,
    result: u32,
) -> Result<TypedTextureProgram, PbrtError> {
    let mut optimizer = Optimizer {
        source_types: slot_types,
        aliases: Vec::new(),
        instructions: Vec::new(),
        slot_types: Vec::new(),
    };
    for instruction in instructions {
        optimizer.push(instruction)?;
    }
    let result = optimizer.resolve(result)?;
    compact_program(
        optimizer.instructions,
        optimizer.slot_types,
        mipmaps,
        image_views,
        result,
    )
}

struct Optimizer {
    source_types: Vec<ValueType>,
    aliases: Vec<u32>,
    instructions: Vec<Instruction>,
    slot_types: Vec<ValueType>,
}

impl Optimizer {
    fn push(&mut self, instruction: Instruction) -> Result<(), PbrtError> {
        let source_dst = instruction_dst(&instruction);
        let value_type = *self
            .source_types
            .get(source_dst as usize)
            .ok_or_else(|| PbrtError::error("Texture instruction has an invalid destination."))?;
        if source_dst as usize != self.aliases.len() {
            return Err(PbrtError::error(
                "Texture program destinations are not in post-order.",
            ));
        }

        let optimized = match instruction {
            Instruction::ConstantFloat { value, .. } => {
                self.append(Instruction::ConstantFloat { dst: 0, value }, value_type)?
            }
            Instruction::ConstantRgb {
                value, color_space, ..
            } => self.append(
                Instruction::ConstantRgb {
                    dst: 0,
                    value,
                    color_space,
                },
                value_type,
            )?,
            Instruction::SampleImage {
                image_view,
                mapping,
                value_type: instruction_type,
                ..
            } => {
                require_type("image", value_type, instruction_type)?;
                self.append(
                    Instruction::SampleImage {
                        dst: 0,
                        image_view,
                        mapping,
                        value_type: instruction_type,
                    },
                    value_type,
                )?
            }
            Instruction::Scale { input, factor, .. } => {
                let input = self.resolve(input)?;
                require_type("scale", value_type, self.type_of(input)?)?;
                if factor == 1.0 {
                    input
                } else if let Some(constant) = self.scaled_constant(input, factor, value_type) {
                    self.append(constant, value_type)?
                } else {
                    self.append(
                        Instruction::Scale {
                            dst: 0,
                            input,
                            factor,
                        },
                        value_type,
                    )?
                }
            }
            Instruction::Mix {
                first,
                second,
                amount,
                constant_amount,
                ..
            } => {
                let first = self.resolve(first)?;
                let second = self.resolve(second)?;
                require_type("mix", value_type, self.type_of(first)?)?;
                require_type("mix", value_type, self.type_of(second)?)?;
                let (amount, constant_amount) = match amount {
                    Some(amount) => {
                        let amount = self.resolve(amount)?;
                        require_type("mix amount", ValueType::Float, self.type_of(amount)?)?;
                        match self.constant_float(amount) {
                            Some(value) => (None, value),
                            None => (Some(amount), constant_amount),
                        }
                    }
                    None => (None, constant_amount),
                };
                if amount.is_none() {
                    if let Some(constant) =
                        self.mixed_constant(first, second, constant_amount, value_type)
                    {
                        self.append(constant, value_type)?
                    } else {
                        self.append(
                            Instruction::Mix {
                                dst: 0,
                                first,
                                second,
                                amount: None,
                                constant_amount,
                            },
                            value_type,
                        )?
                    }
                } else {
                    self.append(
                        Instruction::Mix {
                            dst: 0,
                            first,
                            second,
                            amount,
                            constant_amount,
                        },
                        value_type,
                    )?
                }
            }
            Instruction::Procedural {
                name,
                operands,
                parameters,
                value_type: instruction_type,
                ..
            } => {
                require_type("procedural", value_type, instruction_type)?;
                let operands = operands
                    .into_iter()
                    .map(|slot| self.resolve(slot))
                    .collect::<Result<Vec<_>, _>>()?;
                self.append(
                    Instruction::Procedural {
                        dst: 0,
                        name,
                        operands,
                        parameters,
                        value_type: instruction_type,
                    },
                    value_type,
                )?
            }
        };
        self.aliases.push(optimized);
        Ok(())
    }

    fn append(
        &mut self,
        mut instruction: Instruction,
        value_type: ValueType,
    ) -> Result<u32, PbrtError> {
        set_instruction_dst(&mut instruction, 0);
        if let Some(existing) =
            self.instructions
                .iter()
                .enumerate()
                .find_map(|(slot, candidate)| {
                    (self.slot_types[slot] == value_type
                        && equivalent_instruction(candidate, &instruction))
                    .then_some(slot)
                })
        {
            return u32::try_from(existing)
                .map_err(|_| PbrtError::error("Texture program instruction table exceeds u32."));
        }
        let dst = u32::try_from(self.instructions.len())
            .map_err(|_| PbrtError::error("Texture program instruction table exceeds u32."))?;
        set_instruction_dst(&mut instruction, dst);
        self.instructions.push(instruction);
        self.slot_types.push(value_type);
        Ok(dst)
    }

    fn resolve(&self, source: u32) -> Result<u32, PbrtError> {
        self.aliases
            .get(source as usize)
            .copied()
            .ok_or_else(|| PbrtError::error("Texture instruction reads an uninitialized slot."))
    }

    fn type_of(&self, slot: u32) -> Result<ValueType, PbrtError> {
        self.slot_types
            .get(slot as usize)
            .copied()
            .ok_or_else(|| PbrtError::error("Texture instruction has an invalid operand."))
    }

    fn constant_float(&self, slot: u32) -> Option<f32> {
        match self.instructions.get(slot as usize) {
            Some(Instruction::ConstantFloat { value, .. }) => Some(*value),
            _ => None,
        }
    }

    fn scaled_constant(
        &self,
        input: u32,
        factor: f32,
        value_type: ValueType,
    ) -> Option<Instruction> {
        match (self.instructions.get(input as usize), value_type) {
            (Some(Instruction::ConstantFloat { value, .. }), ValueType::Float) => {
                Some(Instruction::ConstantFloat {
                    dst: 0,
                    value: value * factor,
                })
            }
            (
                Some(Instruction::ConstantRgb {
                    value, color_space, ..
                }),
                ValueType::LinearRgb(expected),
            ) if *color_space == expected => Some(Instruction::ConstantRgb {
                dst: 0,
                value: value.map(|value| value * factor),
                color_space: *color_space,
            }),
            _ => None,
        }
    }

    fn mixed_constant(
        &self,
        first: u32,
        second: u32,
        amount: f32,
        value_type: ValueType,
    ) -> Option<Instruction> {
        match (
            self.instructions.get(first as usize),
            self.instructions.get(second as usize),
            value_type,
        ) {
            (
                Some(Instruction::ConstantFloat { value: first, .. }),
                Some(Instruction::ConstantFloat { value: second, .. }),
                ValueType::Float,
            ) => Some(Instruction::ConstantFloat {
                dst: 0,
                value: first * (1.0 - amount) + second * amount,
            }),
            (
                Some(Instruction::ConstantRgb {
                    value: first,
                    color_space: first_space,
                    ..
                }),
                Some(Instruction::ConstantRgb {
                    value: second,
                    color_space: second_space,
                    ..
                }),
                ValueType::LinearRgb(expected),
            ) if *first_space == expected && *second_space == expected => {
                Some(Instruction::ConstantRgb {
                    dst: 0,
                    value: std::array::from_fn(|index| {
                        first[index] * (1.0 - amount) + second[index] * amount
                    }),
                    color_space: expected,
                })
            }
            _ => None,
        }
    }
}

fn equivalent_instruction(first: &Instruction, second: &Instruction) -> bool {
    match (first, second) {
        (
            Instruction::ConstantFloat { value: first, .. },
            Instruction::ConstantFloat { value: second, .. },
        ) => first.to_bits() == second.to_bits(),
        (
            Instruction::ConstantRgb {
                value: first,
                color_space: first_space,
                ..
            },
            Instruction::ConstantRgb {
                value: second,
                color_space: second_space,
                ..
            },
        ) => float_array_equal(first, second) && first_space == second_space,
        (
            Instruction::SampleImage {
                image_view: first_view,
                mapping: first_mapping,
                value_type: first_type,
                ..
            },
            Instruction::SampleImage {
                image_view: second_view,
                mapping: second_mapping,
                value_type: second_type,
                ..
            },
        ) => {
            first_view == second_view
                && first_type == second_type
                && mapping_equal(first_mapping.as_ref(), second_mapping.as_ref())
        }
        (
            Instruction::Scale {
                input: first_input,
                factor: first_factor,
                ..
            },
            Instruction::Scale {
                input: second_input,
                factor: second_factor,
                ..
            },
        ) => first_input == second_input && first_factor.to_bits() == second_factor.to_bits(),
        (
            Instruction::Mix {
                first: first_a,
                second: first_b,
                amount: first_amount,
                constant_amount: first_constant,
                ..
            },
            Instruction::Mix {
                first: second_a,
                second: second_b,
                amount: second_amount,
                constant_amount: second_constant,
                ..
            },
        ) => {
            first_a == second_a
                && first_b == second_b
                && first_amount == second_amount
                && first_constant.to_bits() == second_constant.to_bits()
        }
        (
            Instruction::Procedural {
                name: first_name,
                operands: first_operands,
                parameters: first_parameters,
                value_type: first_type,
                ..
            },
            Instruction::Procedural {
                name: second_name,
                operands: second_operands,
                parameters: second_parameters,
                value_type: second_type,
                ..
            },
        ) => {
            first_name == second_name
                && first_operands == second_operands
                && float_array_equal(first_parameters, second_parameters)
                && first_type == second_type
        }
        _ => false,
    }
}

pub(super) fn equivalent_program(
    first: &TypedTextureProgram,
    second: &TypedTextureProgram,
) -> bool {
    first.slot_types == second.slot_types
        && first.slot_last_use == second.slot_last_use
        && first.result == second.result
        && first.instructions.len() == second.instructions.len()
        && first
            .instructions
            .iter()
            .zip(&second.instructions)
            .all(|(first, second)| equivalent_instruction(first, second))
}

fn float_array_equal<const N: usize>(first: &[f32; N], second: &[f32; N]) -> bool {
    first
        .iter()
        .zip(second)
        .all(|(first, second)| first.to_bits() == second.to_bits())
}

fn mapping_equal(first: Option<&TextureMapping>, second: Option<&TextureMapping>) -> bool {
    match (first, second) {
        (None, None) => true,
        (Some(TextureMapping::Uv(first)), Some(TextureMapping::Uv(second))) => [
            first.uscale.to_bits() == second.uscale.to_bits(),
            first.vscale.to_bits() == second.vscale.to_bits(),
            first.udelta.to_bits() == second.udelta.to_bits(),
            first.vdelta.to_bits() == second.vdelta.to_bits(),
        ]
        .into_iter()
        .all(|equal| equal),
        (Some(TextureMapping::Planar(first)), Some(TextureMapping::Planar(second)))
        | (Some(TextureMapping::Spherical(first)), Some(TextureMapping::Spherical(second)))
        | (Some(TextureMapping::Cylindrical(first)), Some(TextureMapping::Cylindrical(second)))
        | (
            Some(TextureMapping::PointTransform(first)),
            Some(TextureMapping::PointTransform(second)),
        ) => first
            .matrix
            .iter()
            .zip(second.matrix.iter())
            .all(|(first, second)| first.to_bits() == second.to_bits()),
        _ => false,
    }
}

fn compact_program(
    instructions: Vec<Instruction>,
    slot_types: Vec<ValueType>,
    mipmaps: Vec<Arc<Mipmap>>,
    image_views: Vec<ImageView>,
    result: u32,
) -> Result<TypedTextureProgram, PbrtError> {
    let mut reachable = vec![false; instructions.len()];
    let mut stack = vec![result];
    while let Some(slot) = stack.pop() {
        let reachable_slot = reachable.get_mut(slot as usize).ok_or_else(|| {
            PbrtError::error("Texture program result references an invalid slot.")
        })?;
        if *reachable_slot {
            continue;
        }
        *reachable_slot = true;
        stack.extend(instruction_operands(&instructions[slot as usize]));
    }

    let mut slot_remap = vec![u32::MAX; instructions.len()];
    let mut compacted = Vec::new();
    let mut compacted_types = Vec::new();
    for (old_slot, mut instruction) in instructions.into_iter().enumerate() {
        if !reachable[old_slot] {
            continue;
        }
        remap_operands(&mut instruction, &slot_remap)?;
        let new_slot = u32::try_from(compacted.len())
            .map_err(|_| PbrtError::error("Texture program instruction table exceeds u32."))?;
        set_instruction_dst(&mut instruction, new_slot);
        slot_remap[old_slot] = new_slot;
        compacted.push(instruction);
        compacted_types.push(slot_types[old_slot]);
    }
    let result = *slot_remap
        .get(result as usize)
        .filter(|slot| **slot != u32::MAX)
        .ok_or_else(|| PbrtError::error("Texture program result was removed."))?;

    let mut used_views = BTreeSet::new();
    for instruction in &compacted {
        if let Instruction::SampleImage { image_view, .. } = instruction {
            used_views.insert(*image_view);
        }
    }
    let mut view_remap = vec![u32::MAX; image_views.len()];
    let mut compacted_views = Vec::with_capacity(used_views.len());
    for old_view in used_views {
        let view = image_views
            .get(old_view as usize)
            .ok_or_else(|| PbrtError::error("Texture instruction has an invalid image view."))?;
        let new_view = u32::try_from(compacted_views.len())
            .map_err(|_| PbrtError::error("Texture image view table exceeds u32."))?;
        view_remap[old_view as usize] = new_view;
        compacted_views.push(view.clone());
    }
    for instruction in &mut compacted {
        if let Instruction::SampleImage { image_view, .. } = instruction {
            *image_view = view_remap[*image_view as usize];
        }
    }

    let mut used_mipmaps = BTreeSet::new();
    for view in &compacted_views {
        used_mipmaps.insert(view.mipmap);
    }
    let mut mipmap_remap = vec![u32::MAX; mipmaps.len()];
    let mut compacted_mipmaps = Vec::with_capacity(used_mipmaps.len());
    for old_mipmap in used_mipmaps {
        let mipmap = mipmaps
            .get(old_mipmap as usize)
            .ok_or_else(|| PbrtError::error("Texture view has an invalid mipmap."))?;
        let new_mipmap = u32::try_from(compacted_mipmaps.len())
            .map_err(|_| PbrtError::error("Texture mipmap table exceeds u32."))?;
        mipmap_remap[old_mipmap as usize] = new_mipmap;
        compacted_mipmaps.push(mipmap.clone());
    }
    for view in &mut compacted_views {
        view.mipmap = mipmap_remap[view.mipmap as usize];
    }

    let slot_last_use = calculate_last_use(&compacted, result)?;
    Ok(TypedTextureProgram {
        instructions: compacted,
        slot_types: compacted_types,
        mipmaps: compacted_mipmaps,
        image_views: compacted_views,
        slot_last_use,
        result,
    })
}

fn calculate_last_use(instructions: &[Instruction], result: u32) -> Result<Vec<u32>, PbrtError> {
    let mut last_use = vec![0; instructions.len()];
    for (instruction_index, instruction) in instructions.iter().enumerate() {
        let instruction_index = u32::try_from(instruction_index)
            .map_err(|_| PbrtError::error("Texture instruction index exceeds u32."))?;
        for operand in instruction_operands(instruction) {
            let entry = last_use
                .get_mut(operand as usize)
                .ok_or_else(|| PbrtError::error("Texture instruction has an invalid operand."))?;
            *entry = (*entry).max(instruction_index);
        }
    }
    last_use[result as usize] = u32::try_from(instructions.len().saturating_sub(1))
        .map_err(|_| PbrtError::error("Texture instruction index exceeds u32."))?;
    Ok(last_use)
}

fn require_type(operation: &str, expected: ValueType, actual: ValueType) -> Result<(), PbrtError> {
    if expected == actual {
        Ok(())
    } else {
        Err(PbrtError::error(&format!(
            "Texture {operation} has incompatible value types: expected {expected:?}, got {actual:?}."
        )))
    }
}

fn instruction_dst(instruction: &Instruction) -> u32 {
    match instruction {
        Instruction::ConstantFloat { dst, .. }
        | Instruction::ConstantRgb { dst, .. }
        | Instruction::SampleImage { dst, .. }
        | Instruction::Scale { dst, .. }
        | Instruction::Mix { dst, .. }
        | Instruction::Procedural { dst, .. } => *dst,
    }
}

fn set_instruction_dst(instruction: &mut Instruction, dst: u32) {
    match instruction {
        Instruction::ConstantFloat { dst: value, .. }
        | Instruction::ConstantRgb { dst: value, .. }
        | Instruction::SampleImage { dst: value, .. }
        | Instruction::Scale { dst: value, .. }
        | Instruction::Mix { dst: value, .. }
        | Instruction::Procedural { dst: value, .. } => *value = dst,
    }
}

fn instruction_operands(instruction: &Instruction) -> Vec<u32> {
    match instruction {
        Instruction::Scale { input, .. } => vec![*input],
        Instruction::Mix {
            first,
            second,
            amount,
            ..
        } => amount.iter().copied().chain([*first, *second]).collect(),
        Instruction::Procedural { operands, .. } => operands.clone(),
        Instruction::ConstantFloat { .. }
        | Instruction::ConstantRgb { .. }
        | Instruction::SampleImage { .. } => Vec::new(),
    }
}

fn remap_operands(instruction: &mut Instruction, remap: &[u32]) -> Result<(), PbrtError> {
    let remap_slot = |slot: &mut u32| -> Result<(), PbrtError> {
        *slot = *remap
            .get(*slot as usize)
            .filter(|slot| **slot != u32::MAX)
            .ok_or_else(|| PbrtError::error("Texture instruction operand was removed."))?;
        Ok(())
    };
    match instruction {
        Instruction::Scale { input, .. } => remap_slot(input)?,
        Instruction::Mix {
            first,
            second,
            amount,
            ..
        } => {
            remap_slot(first)?;
            remap_slot(second)?;
            if let Some(amount) = amount {
                remap_slot(amount)?;
            }
        }
        Instruction::Procedural { operands, .. } => {
            for operand in operands {
                remap_slot(operand)?;
            }
        }
        Instruction::ConstantFloat { .. }
        | Instruction::ConstantRgb { .. }
        | Instruction::SampleImage { .. } => {}
    }
    Ok(())
}
