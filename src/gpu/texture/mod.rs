//! Backend-independent texture compilation data.

pub mod compile;
pub mod evaluate;
pub mod image;
pub mod typed_program;

pub use compile::{compile_texture_library, TextureLibrary, TextureRoot, TextureRootSpec};
pub use evaluate::{evaluate_texture_program, evaluate_texture_program_at, TextureValue};
pub use image::{
    project_float_mipmap, ColorSpace, ImageCompiler, ImageFilterMode, ImageOptimizationPolicy,
    ImageValueType, ImageView, ImageWrapMode, Mipmap, MipmapEncoding, MipmapLevel, MipmapLevelData,
};
pub use typed_program::{
    Instruction as TextureInstruction, TypedTextureProgram, ValueType as TextureValueType,
};
