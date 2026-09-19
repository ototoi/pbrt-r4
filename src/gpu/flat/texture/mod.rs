//! Executable and storage-oriented Flat IR texture representation.

pub mod compile;
pub mod evaluate;
pub mod image;
mod optimize;
pub mod program;

pub use compile::{compile_texture_library, TextureLibrary, TextureRoot, TextureRootSpec};
pub use evaluate::{evaluate_texture_program, evaluate_texture_program_at, TextureValue};
pub use image::{
    project_float_mipmap, validate_mipmap, ColorSpace, ImageCompiler, ImageDecoder,
    ImageFilterMode, ImageOptimizationPolicy, ImageValueType, ImageView, ImageWrapMode, Mipmap,
    MipmapEncoding, MipmapLevel, MipmapLevelData,
};
pub use program::{
    Instruction as TextureInstruction, TypedTextureProgram, ValueType as TextureValueType,
};
