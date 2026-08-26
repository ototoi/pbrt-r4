use super::{Float, ImageId, Index, Matrix4x4, SpectrumId, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub enum SpectrumResource {
    Constant {
        value: Float,
    },
    PiecewiseLinear {
        wavelengths_nm: Box<[Float]>,
        values: Box<[Float]>,
    },
    RgbAlbedo {
        coefficients: [Float; 3],
    },
    RgbUnbounded {
        coefficients: [Float; 3],
    },
    RgbIlluminant {
        coefficients: [Float; 3],
        illuminant: SpectrumId,
    },
    Blackbody {
        temperature_kelvin: Float,
        scale: Float,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageChannels {
    R,
    Rg,
    Rgb,
    Rgba,
}

impl ImageChannels {
    pub fn count(self) -> usize {
        match self {
            Self::R => 1,
            Self::Rg => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TexelStorage {
    U8(Box<[u8]>),
    F16(Box<[u16]>),
    F32(Box<[Float]>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorEncoding {
    Linear,
    Srgb,
    Gamma { exponent: Float },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MipLevel {
    pub resolution: [Index; 2],
    pub texel_offset: u64,
    pub texel_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageResource {
    pub resolution: [Index; 2],
    pub channels: ImageChannels,
    pub storage: TexelStorage,
    pub mip_levels: Box<[MipLevel]>,
    pub color_encoding: ColorEncoding,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImageWrapMode {
    Black,
    Clamp,
    Repeat,
    OctahedralSphere,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImageFilter {
    Point,
    Bilinear,
    Trilinear,
    Ewa { max_anisotropy: Float },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextureMapping {
    Uv {
        su: Float,
        sv: Float,
        du: Float,
        dv: Float,
    },
    Spherical {
        texture_from_render: Matrix4x4,
    },
    Cylindrical {
        texture_from_render: Matrix4x4,
    },
    Planar {
        texture_from_render: Matrix4x4,
        vs: Vector3,
        vt: Vector3,
        ds: Float,
        dt: Float,
    },
    Transform3D {
        texture_from_render: Matrix4x4,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FloatTexture {
    Constant {
        value: Float,
    },
    Image {
        image: ImageId,
        mapping: super::TextureMappingId,
        scale: Float,
        invert: bool,
        swrap: ImageWrapMode,
        twrap: ImageWrapMode,
        filter: ImageFilter,
        channel: FloatImageChannel,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatImageChannel {
    Channel0,
    Alpha,
    RgbAverage,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpectrumTexture {
    Constant {
        value: SpectrumId,
    },
    Image {
        image: ImageId,
        mapping: super::TextureMappingId,
        scale: Float,
        invert: bool,
        swrap: ImageWrapMode,
        twrap: ImageWrapMode,
        filter: ImageFilter,
        spectrum_type: SpectrumType,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpectrumType {
    Albedo,
    Unbounded,
    Illuminant,
}
