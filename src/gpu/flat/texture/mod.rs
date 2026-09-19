//! Executable and storage-oriented Flat IR texture representation.

mod compile;
mod evaluate;
mod image;
mod optimize;
mod program;

pub use compile::{compile_texture_library, TextureLibrary, TextureRoot, TextureRootSpec};
pub use evaluate::{
    evaluate_texture_root, evaluate_texture_root_at, evaluate_texture_root_with_context,
    TextureEvaluationContext, TextureValue,
};
pub use image::{
    project_float_mipmap, project_linear_rgb_mipmap, validate_mipmap, ColorSpace, ImageCompiler,
    ImageDecoder, ImageFilterMode, ImageOptimizationPolicy, ImageValueType, ImageView,
    ImageWrapMode, Mipmap, MipmapEncoding, MipmapLevel, MipmapLevelData,
};
pub use program::{
    Instruction as TextureInstruction, ProceduralOperation, TypedTextureProgram,
    ValueType as TextureValueType,
};
