//! Typed post-order texture programs owned by Flat IR.

use std::collections::HashMap;
use std::sync::Arc;

use crate::gpu::node::{TextureComponent, TextureKind, TextureMapping, TextureNode};
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;
use crate::util::spectrum::Spectrum;

use super::image::{
    ColorSpace, ImageDecoder, ImageFilterMode, ImageValueType, ImageView, ImageWrapMode, Mipmap,
};
use super::optimize::optimize_texture_program;

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
    fn parse(name: &str) -> Option<Self> {
        match name {
            "directionmix" => Some(Self::DirectionMix),
            "dots" => Some(Self::Dots),
            "fbm" => Some(Self::Fbm),
            "wrinkled" => Some(Self::Wrinkled),
            "windy" => Some(Self::Windy),
            "bilerp" => Some(Self::Bilerp),
            "marble" => Some(Self::Marble),
            name if name.contains("checkerboard") => Some(Self::Checkerboard),
            _ => None,
        }
    }

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

pub struct ProgramCompilation {
    pub program: TypedTextureProgram,
    pub mipmaps: Vec<Arc<Mipmap>>,
    pub image_views: Vec<ImageView>,
}

pub fn compile_texture_program(
    root: &Arc<TextureNode>,
    image_decoder: &mut ImageDecoder,
) -> Result<ProgramCompilation, PbrtError> {
    Compiler::new(image_decoder).compile(root)
}

struct Compiler<'a> {
    instructions: Vec<Instruction>,
    slot_types: Vec<ValueType>,
    slots_by_node: HashMap<usize, u32>,
    visiting: Vec<usize>,
    image_views: Vec<ImageView>,
    image_views_by_key: HashMap<ImageViewKey, u32>,
    image_decoder: &'a mut ImageDecoder,
    mipmaps: Vec<Arc<Mipmap>>,
    mipmaps_by_identity: HashMap<usize, u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ImageViewKey {
    mipmap: u32,
    value_type: ImageValueType,
    swrap: ImageWrapMode,
    twrap: ImageWrapMode,
    filter: ImageFilterMode,
    scale: u32,
    invert: bool,
}

impl<'a> Compiler<'a> {
    fn new(image_decoder: &'a mut ImageDecoder) -> Self {
        Self {
            instructions: Vec::new(),
            slot_types: Vec::new(),
            slots_by_node: HashMap::new(),
            visiting: Vec::new(),
            image_views: Vec::new(),
            image_views_by_key: HashMap::new(),
            image_decoder,
            mipmaps: Vec::new(),
            mipmaps_by_identity: HashMap::new(),
        }
    }

    fn compile(mut self, root: &Arc<TextureNode>) -> Result<ProgramCompilation, PbrtError> {
        let result = self.emit(root)?;
        optimize_texture_program(
            self.instructions,
            self.slot_types,
            self.mipmaps,
            self.image_views,
            result,
        )
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
        let value_type = if texture.name == "scale" {
            let input = operands.first().copied().ok_or_else(|| {
                PbrtError::error(&format!(
                    "Texture node \"{}\" scale expects at least one child.",
                    node.name
                ))
            })?;
            let input_type = self
                .slot_types
                .get(input as usize)
                .copied()
                .ok_or_else(|| PbrtError::error("Texture scale has an invalid input slot."))?;
            match (texture.kind, input_type) {
                (TextureKind::Float, ValueType::Float)
                | (TextureKind::Spectrum, ValueType::LinearRgb(_)) => input_type,
                _ => {
                    let result_kind = match texture.kind {
                        TextureKind::Float => "float",
                        TextureKind::Spectrum => "spectrum",
                    };
                    return Err(PbrtError::error(&format!(
                        "Texture scale has incompatible value types: result kind {result_kind}, input {input_type:?}."
                    )));
                }
            }
        } else {
            value_type(texture.kind, &texture.params)
        };
        let dst = u32::try_from(self.slot_types.len())
            .map_err(|_| PbrtError::error("Texture program slot table exceeds u32."))?;
        let image_view = if texture.name == "imagemap" {
            let path = texture.image_path().ok_or_else(|| {
                PbrtError::error(&format!("Image texture \"{}\" has no filename.", node.name))
            })?;
            let default_encoding = if path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
            {
                "sRGB"
            } else {
                "linear"
            };
            let encoding = texture.params.get_one_string("encoding", default_encoding);
            let mipmap = self.image_decoder.decode(&path, &encoding)?;
            let mipmap = self.intern_mipmap(mipmap)?;
            Some(self.intern_image_view(image_view(texture, value_type, mipmap))?)
        } else {
            None
        };
        let instruction = instruction(dst, node, texture, value_type, operands, image_view)?;
        self.instructions.push(instruction);
        self.slot_types.push(value_type);
        self.slots_by_node.insert(key, dst);
        self.visiting.pop();
        Ok(dst)
    }

    fn intern_image_view(&mut self, view: ImageView) -> Result<u32, PbrtError> {
        let key = ImageViewKey {
            mipmap: view.mipmap,
            value_type: view.value_type,
            swrap: view.swrap,
            twrap: view.twrap,
            filter: view.filter,
            scale: view.scale.to_bits(),
            invert: view.invert,
        };
        if let Some(&index) = self.image_views_by_key.get(&key) {
            return Ok(index);
        }
        let index = u32::try_from(self.image_views.len())
            .map_err(|_| PbrtError::error("Texture program image view table exceeds u32."))?;
        self.image_views.push(view);
        self.image_views_by_key.insert(key, index);
        Ok(index)
    }

    fn intern_mipmap(&mut self, mipmap: Arc<Mipmap>) -> Result<u32, PbrtError> {
        let identity = Arc::as_ptr(&mipmap) as usize;
        if let Some(&index) = self.mipmaps_by_identity.get(&identity) {
            return Ok(index);
        }
        let index = u32::try_from(self.mipmaps.len())
            .map_err(|_| PbrtError::error("Texture mipmap table exceeds u32."))?;
        self.mipmaps.push(mipmap);
        self.mipmaps_by_identity.insert(identity, index);
        Ok(index)
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
    image_view: Option<u32>,
) -> Result<Instruction, PbrtError> {
    match texture.name.as_str() {
        "constant" => match texture.kind {
            TextureKind::Float => Ok(Instruction::ConstantFloat {
                dst,
                value: texture.params.get_one_float("value", 1.0) as f32,
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
            image_view: image_view.ok_or_else(|| {
                PbrtError::error("Image texture instruction is missing its image view.")
            })?,
            mapping: texture_mapping(node),
            value_type,
        }),
        "scale" => {
            if !(1..=2).contains(&operands.len()) {
                return Err(PbrtError::error(&format!(
                    "Texture node \"{}\" scale expects one or two children.",
                    node.name
                )));
            }
            Ok(Instruction::Scale {
                dst,
                input: operands[0],
                scale: operands.get(1).copied(),
                constant_scale: texture
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
        name => {
            let operation = ProceduralOperation::parse(name).ok_or_else(|| {
                PbrtError::error(&format!(
                    "Flat texture operation \"{name}\" is not implemented."
                ))
            })?;
            let expected_operands = match operation {
                ProceduralOperation::Bilerp => 4,
                ProceduralOperation::Checkerboard
                | ProceduralOperation::DirectionMix
                | ProceduralOperation::Dots => 2,
                ProceduralOperation::Fbm
                | ProceduralOperation::Wrinkled
                | ProceduralOperation::Windy
                | ProceduralOperation::Marble => 0,
            };
            if operands.len() != expected_operands {
                return Err(PbrtError::error(&format!(
                    "Texture node \"{}\" {} expects {expected_operands} children, got {}.",
                    node.name,
                    operation.name(),
                    operands.len()
                )));
            }
            Ok(Instruction::Procedural {
                dst,
                operation,
                operands,
                parameters: [
                    texture.params.get_one_float("roughness", 0.5) as f32,
                    texture.params.get_one_int("octaves", 8) as f32,
                    texture.params.get_one_float("scale", 1.0) as f32,
                    texture.params.get_one_float("variation", 0.2) as f32,
                ],
                mapping: texture_mapping(node),
                value_type,
            })
        }
    }
}

fn texture_mapping(node: &TextureNode) -> Option<TextureMapping> {
    node.components
        .iter()
        .find_map(|component| match component {
            TextureComponent::Mapping(mapping) => Some(mapping.clone()),
            TextureComponent::Texture(_) => None,
        })
}

fn image_view(
    texture: &crate::gpu::node::Texture,
    value_type: ValueType,
    mipmap: u32,
) -> ImageView {
    let wrap = |name: &str| match name {
        "clamp" => ImageWrapMode::Clamp,
        "black" => ImageWrapMode::Black,
        _ => ImageWrapMode::Repeat,
    };
    let wrap_mode = texture.params.get_one_string("wrap", "repeat");
    let filter = match texture.params.get_one_string("filter", "bilinear") {
        value if value == "point" || value == "nearest" => ImageFilterMode::Nearest,
        value if value == "trilinear" => ImageFilterMode::Trilinear,
        _ => ImageFilterMode::Bilinear,
    };
    ImageView {
        mipmap,
        value_type: match value_type {
            ValueType::Float => ImageValueType::Float,
            ValueType::LinearRgb(_) => ImageValueType::LinearRgb,
        },
        swrap: wrap(&texture.params.get_one_string("swrap", &wrap_mode)),
        twrap: wrap(&texture.params.get_one_string("twrap", &wrap_mode)),
        filter,
        scale: texture.params.get_one_float("scale", 1.0) as f32,
        invert: texture.params.get_one_bool("invert", false),
    }
}

fn value_type_color_space(value_type: ValueType) -> ColorSpace {
    match value_type {
        ValueType::Float => ColorSpace::Unknown,
        ValueType::LinearRgb(color_space) => color_space,
    }
}
