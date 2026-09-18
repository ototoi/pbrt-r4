//! Backend-independent texture compilation data.

pub mod compile;
pub mod image;
pub mod program;

pub use compile::{compile_texture_library, TextureLibrary, TextureRoot, TextureRootSpec};
pub use image::{
    project_float_mipmap, ColorSpace, ImageCompiler, ImageFilterMode, ImageOptimizationPolicy,
    ImageValueType, ImageView, ImageWrapMode, Mipmap, MipmapEncoding, MipmapLevel, MipmapLevelData,
};
