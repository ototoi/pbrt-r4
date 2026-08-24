use super::{GpuFloat, GpuIndex, GpuMatrix4x4, GpuVector3, ImageId, SpectrumId};

#[derive(Clone, Debug, PartialEq)]
pub enum GpuSpectrumResource {
    Constant {
        value: GpuFloat,
    },
    PiecewiseLinear {
        wavelengths_nm: Box<[GpuFloat]>,
        values: Box<[GpuFloat]>,
    },
    RgbAlbedo {
        coefficients: [GpuFloat; 3],
    },
    RgbUnbounded {
        coefficients: [GpuFloat; 3],
    },
    RgbIlluminant {
        coefficients: [GpuFloat; 3],
        illuminant: SpectrumId,
    },
    Blackbody {
        temperature_kelvin: GpuFloat,
        scale: GpuFloat,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuImageChannels {
    R,
    Rg,
    Rgb,
    Rgba,
}

impl GpuImageChannels {
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
pub enum GpuTexelStorage {
    U8(Box<[u8]>),
    F16(Box<[u16]>),
    F32(Box<[GpuFloat]>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuColorEncoding {
    Linear,
    Srgb,
    Gamma { exponent: GpuFloat },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMipLevel {
    pub resolution: [GpuIndex; 2],
    pub texel_offset: u64,
    pub texel_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuImageResource {
    pub resolution: [GpuIndex; 2],
    pub channels: GpuImageChannels,
    pub storage: GpuTexelStorage,
    pub mip_levels: Box<[GpuMipLevel]>,
    pub color_encoding: GpuColorEncoding,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuImageWrapMode {
    Black,
    Clamp,
    Repeat,
    OctahedralSphere,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuImageFilter {
    Point,
    Bilinear,
    Trilinear,
    Ewa { max_anisotropy: GpuFloat },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuTextureMapping {
    Uv {
        su: GpuFloat,
        sv: GpuFloat,
        du: GpuFloat,
        dv: GpuFloat,
    },
    Spherical {
        texture_from_render: GpuMatrix4x4,
    },
    Cylindrical {
        texture_from_render: GpuMatrix4x4,
    },
    Planar {
        texture_from_render: GpuMatrix4x4,
        vs: GpuVector3,
        vt: GpuVector3,
        ds: GpuFloat,
        dt: GpuFloat,
    },
    Transform3D {
        texture_from_render: GpuMatrix4x4,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuFloatTexture {
    Constant {
        value: GpuFloat,
    },
    Image {
        image: ImageId,
        mapping: super::TextureMappingId,
        scale: GpuFloat,
        invert: bool,
        swrap: GpuImageWrapMode,
        twrap: GpuImageWrapMode,
        filter: GpuImageFilter,
        channel: GpuFloatImageChannel,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuFloatImageChannel {
    Channel0,
    Alpha,
    RgbAverage,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuSpectrumTexture {
    Constant {
        value: SpectrumId,
    },
    Image {
        image: ImageId,
        mapping: super::TextureMappingId,
        scale: GpuFloat,
        invert: bool,
        swrap: GpuImageWrapMode,
        twrap: GpuImageWrapMode,
        filter: GpuImageFilter,
        spectrum_type: GpuSpectrumType,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuSpectrumType {
    Albedo,
    Unbounded,
    Illuminant,
}
