//! Executable and storage-oriented Flat IR texture representation.

mod evaluate;
mod image;
mod library;
mod mapping;
mod program;

pub use evaluate::{
    evaluate_texture_root, evaluate_texture_root_at, evaluate_texture_root_with_context,
    TextureEvaluationContext, TextureValue,
};
pub use image::{
    build_linear_rgb_mipmap, project_float_mipmap, project_linear_rgb_mipmap, validate_mipmap,
    ColorSpace, ImageCompiler, ImageDecoder, ImageFilterMode, ImageOptimizationPolicy,
    ImageValueType, ImageView, ImageWrapMode, Mipmap, MipmapEncoding, MipmapLevel, MipmapLevelData,
};
pub use library::{TextureLibrary, TextureRoot};
pub use mapping::{TextureMapping, UvMapping};
pub use program::{
    Instruction as TextureInstruction, ProceduralOperation, TypedTextureProgram,
    ValueType as TextureValueType,
};
