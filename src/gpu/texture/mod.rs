//! Backend-independent texture compilation data.

pub mod image;
pub mod program;

pub use image::{
    project_float_mipmap, ColorSpace, ImageFilterMode, ImageValueType, ImageView, ImageWrapMode,
    Mipmap, MipmapEncoding, MipmapLevel, MipmapLevelData,
};
