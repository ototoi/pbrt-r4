//! Typed post-order texture programs owned by Flat IR.

use super::image::ColorSpace;
use super::mapping::TextureMapping;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ValueType {
    Float,
    LinearRgb(ColorSpace),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProceduralOperation {
    Checkerboard,
    DirectionMix,
    Dots,
    Fbm,
    Wrinkled,
    Windy,
    Bilerp,
    Marble,
}

impl ProceduralOperation {
    pub fn name(self) -> &'static str {
        match self {
            Self::Checkerboard => "checkerboard",
            Self::DirectionMix => "directionmix",
            Self::Dots => "dots",
            Self::Fbm => "fbm",
            Self::Wrinkled => "wrinkled",
            Self::Windy => "windy",
            Self::Bilerp => "bilerp",
            Self::Marble => "marble",
        }
    }
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
        image_view: u32,
        mapping: Option<TextureMapping>,
        value_type: ValueType,
    },
    Scale {
        dst: u32,
        input: u32,
        scale: Option<u32>,
        constant_scale: f32,
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
        operation: ProceduralOperation,
        operands: Vec<u32>,
        parameters: [f32; 4],
        mapping: Option<TextureMapping>,
        value_type: ValueType,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct TypedTextureProgram {
    pub instructions: Vec<Instruction>,
    pub slot_types: Vec<ValueType>,
    /// Last instruction index that reads each slot. The result slot is kept
    /// live through the end of the program for backend consumers.
    pub slot_last_use: Vec<u32>,
    pub result: u32,
}
