//! Typed post-order texture programs.

use std::collections::HashMap;
use std::sync::Arc;

use crate::gpu::node::{ColorSpace, TextureComponent, TextureKind, TextureNode};
use crate::gpu::texture::Mipmap;
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;
use crate::util::spectrum::Spectrum;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValueType {
    Float,
    LinearRgb(ColorSpace),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    ConstantFloat {
        dst: u32,
        value: f32,
    },
    ConstantRgb {
        dst: u32,
        value: [f32; 3],
        color_space: ColorSpace,
    },
    SampleImage {
        dst: u32,
        mipmap: Arc<Mipmap>,
        value_type: ValueType,
    },
    Scale {
        dst: u32,
        input: u32,
        factor: f32,
    },
    Mix {
        dst: u32,
        first: u32,
        second: u32,
        amount: Option<u32>,
        constant_amount: f32,
    },
    Procedural {
        dst: u32,
        name: String,
        operands: Vec<u32>,
        parameters: [f32; 4],
        value_type: ValueType,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypedTextureProgram {
    pub instructions: Vec<Instruction>,
    pub slot_types: Vec<ValueType>,
    pub result: u32,
}

impl TypedTextureProgram {
    pub fn compile(root: &Arc<TextureNode>) -> Result<Self, PbrtError> {
        Compiler::default().compile(root)
    }
}

#[derive(Default)]
struct Compiler {
    instructions: Vec<Instruction>,
    slot_types: Vec<ValueType>,
    slots_by_node: HashMap<usize, u32>,
    visiting: Vec<usize>,
}

impl Compiler {
    fn compile(mut self, root: &Arc<TextureNode>) -> Result<TypedTextureProgram, PbrtError> {
        let result = self.emit(root)?;
        Ok(TypedTextureProgram {
            instructions: self.instructions,
            slot_types: self.slot_types,
            result,
        })
    }

    fn emit(&mut self, node: &Arc<TextureNode>) -> Result<u32, PbrtError> {
        let key = Arc::as_ptr(node) as usize;
        if self.visiting.contains(&key) {
            return Err(PbrtError::error("Texture graph contains a cycle."));
        }
        if let Some(&slot) = self.slots_by_node.get(&key) {
            return Ok(slot);
        }
        self.visiting.push(key);
        let mut operands = Vec::with_capacity(node.children.len());
        for child in &node.children {
            operands.push(self.emit(child)?);
        }
        let texture = texture_component(node)?;
        let value_type = value_type(texture.kind, &texture.params);
        let dst = u32::try_from(self.slot_types.len())
            .map_err(|_| PbrtError::error("Texture program slot table exceeds u32."))?;
        let instruction = instruction(dst, node, texture, value_type, operands)?;
        self.instructions.push(instruction);
        self.slot_types.push(value_type);
        self.slots_by_node.insert(key, dst);
        self.visiting.pop();
        Ok(dst)
    }
}

fn texture_component(node: &TextureNode) -> Result<&crate::gpu::node::Texture, PbrtError> {
    let mut texture = None;
    for component in &node.components {
        if let TextureComponent::Texture(value) = component {
            if texture.replace(value).is_some() {
                return Err(PbrtError::error(&format!(
                    "Texture node \"{}\" has multiple texture components.",
                    node.name
                )));
            }
        }
    }
    texture.ok_or_else(|| {
        PbrtError::error(&format!(
            "Texture node \"{}\" has no texture component.",
            node.name
        ))
    })
}

fn value_type(kind: TextureKind, params: &ParameterDictionary) -> ValueType {
    match kind {
        TextureKind::Float => ValueType::Float,
        TextureKind::Spectrum => ValueType::LinearRgb(parse_color_space(
            &params.get_one_string("colorspace", "sRGB"),
        )),
    }
}

fn parse_color_space(value: &str) -> ColorSpace {
    match value {
        "ACES2065-1" => ColorSpace::Aces2065,
        "DCI-P3" => ColorSpace::DciP3,
        "Rec2020" => ColorSpace::Rec2020,
        "sRGB" | "srgb" => ColorSpace::Srgb,
        _ => ColorSpace::Unknown,
    }
}

fn instruction(
    dst: u32,
    node: &TextureNode,
    texture: &crate::gpu::node::Texture,
    value_type: ValueType,
    operands: Vec<u32>,
) -> Result<Instruction, PbrtError> {
    match texture.name.as_str() {
        "constant" => match texture.kind {
            TextureKind::Float => Ok(Instruction::ConstantFloat {
                dst,
                value: texture.params.get_one_float("value", 0.0) as f32,
            }),
            TextureKind::Spectrum => Ok(Instruction::ConstantRgb {
                dst,
                value: texture
                    .params
                    .get_one_spectrum("value", &Spectrum::from(0.0))
                    .to_rgb()
                    .map(|value| value as f32),
                color_space: value_type_color_space(value_type),
            }),
        },
        "imagemap" => Ok(Instruction::SampleImage {
            dst,
            mipmap: texture.mipmap.clone().ok_or_else(|| {
                PbrtError::error(&format!("Image texture \"{}\" has no mipmap.", node.name))
            })?,
            value_type,
        }),
        "scale" => {
            if operands.len() != 1 {
                return Err(PbrtError::error(&format!(
                    "Texture node \"{}\" scale expects one child.",
                    node.name
                )));
            }
            Ok(Instruction::Scale {
                dst,
                input: operands[0],
                factor: texture
                    .params
                    .get_one_float("scale", texture.params.get_one_float("value", 1.0))
                    as f32,
            })
        }
        "mix" => {
            if !(2..=3).contains(&operands.len()) {
                return Err(PbrtError::error(&format!(
                    "Texture node \"{}\" mix expects two or three children.",
                    node.name
                )));
            }
            Ok(Instruction::Mix {
                dst,
                first: operands[0],
                second: operands[1],
                amount: operands.get(2).copied(),
                constant_amount: texture.params.get_one_float("amount", 0.5) as f32,
            })
        }
        name => Ok(Instruction::Procedural {
            dst,
            name: name.to_string(),
            operands,
            parameters: [
                texture.params.get_one_float("roughness", 0.5) as f32,
                texture.params.get_one_int("octaves", 8) as f32,
                texture.params.get_one_float("scale", 1.0) as f32,
                texture.params.get_one_float("variation", 0.2) as f32,
            ],
            value_type,
        }),
    }
}

fn value_type_color_space(value_type: ValueType) -> ColorSpace {
    match value_type {
        ValueType::Float => ColorSpace::Unknown,
        ValueType::LinearRgb(color_space) => color_space,
    }
}
