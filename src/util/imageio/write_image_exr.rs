//! Write OpenEXR images with pbrt-v4-compatible metadata.
//!
//! pbrt-v4's `RGBFilm` / `GBufferFilm` outputs have the following
//! structure:
//!   - `dataWindow`: the cropped pixel region (`output_bounds`) in
//!     **whole-image coordinates**. e.g. a 1920×1440 image cropped to
//!     0.4–0.6 yields dataWindow (768, 576)–(1151, 863).
//!   - `displayWindow`: the full image region (`total_resolution`).
//!   - compression = ZIP (= `Encoding::SMALL_LOSSLESS`), scanline,
//!     INCREASING_Y.
//!   - `RGBFilm` writes FLOAT samples, `GBufferFilm` writes HALF.
//!   - extra attributes like `renderTimeSeconds`, `samplesPerPixel`.
//!
//! Matching these conventions lets standard tools (macOS EXR viewer,
//! oiio, etc.) read r4 output identically to v4. Before this, r4
//! wrote tiled + RLE + crop-relative dataWindow EXRs that confused
//! auto-exposure pipelines and shifted the perceived colour.

use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::image::ImageMetadata;

use exr::meta::attribute::{AttributeValue, IntegerBounds, Text};
use exr::prelude::f16;
use exr::prelude::{
    AnyChannel, AnyChannels, Encoding, FlatSamples, Image, Layer, LayerAttributes, SmallVec, Vec2,
    WritableImage,
};

/// EXR pixel-sample format. `RGBFilm` uses FLOAT, `GBufferFilm` uses
/// HALF, matching pbrt-v4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExrPixelType {
    Float,
    Half,
}

/// Optional metadata written alongside the channel data.
#[derive(Default, Clone)]
pub struct ExrExtraMeta<'a> {
    pub render_time_seconds: Option<f32>,
    pub samples_per_pixel: Option<i32>,
    pub mse: Option<f32>,
    pub world_to_camera: Option<[Float; 16]>,
    pub world_to_ndc: Option<[Float; 16]>,
    /// `("name", "value")` string attributes (used by e.g. spectral EXR).
    pub string_attrs: &'a [(&'a str, &'a str)],
}

fn samples_to_flat(samples: &[f32], pixel_type: ExrPixelType) -> FlatSamples {
    match pixel_type {
        ExrPixelType::Float => FlatSamples::F32(samples.to_vec()),
        ExrPixelType::Half => {
            let v: Vec<f16> = samples.iter().map(|x| f16::from_f32(*x)).collect();
            FlatSamples::F16(v)
        }
    }
}

fn matrix_to_attr(m: &[Float; 16]) -> AttributeValue {
    let mut arr = [0.0_f32; 16];
    for i in 0..16 {
        arr[i] = m[i] as f32;
    }
    AttributeValue::Matrix4x4(arr)
}

/// Internal worker: write a channel group to EXR with v4-compatible
/// metadata.
fn write_channels_inner(
    name: &str,
    output_bounds: &Bounds2i,
    total_resolution: &Point2i,
    pixel_type: ExrPixelType,
    channels: &[(&str, Vec<f32>)],
    meta: &ExrExtraMeta,
) -> std::result::Result<(), PbrtError> {
    let resolution = output_bounds.diagonal();
    let expected_len = (resolution.x * resolution.y) as usize;

    let mut any_channels: SmallVec<[AnyChannel<FlatSamples>; 4]> = SmallVec::new();
    for (channel_name, samples) in channels.iter() {
        if samples.len() != expected_len {
            return Err(PbrtError::error(&format!(
                "EXR channel \"{}\" has {} samples; expected {}.",
                channel_name,
                samples.len(),
                expected_len
            )));
        }
        any_channels.push(AnyChannel::new(
            *channel_name,
            samples_to_flat(samples, pixel_type),
        ));
    }

    // Layer attributes
    let mut layer_attrs = LayerAttributes::default();
    // dataWindow position: the crop region's offset within the full image.
    layer_attrs.layer_position = Vec2(output_bounds.min.x, output_bounds.min.y);

    if let Some(t) = meta.render_time_seconds {
        layer_attrs
            .other
            .insert(Text::from("renderTimeSeconds"), AttributeValue::F32(t));
    }
    if let Some(spp) = meta.samples_per_pixel {
        layer_attrs
            .other
            .insert(Text::from("samplesPerPixel"), AttributeValue::I32(spp));
    }
    if let Some(mse) = meta.mse {
        layer_attrs
            .other
            .insert(Text::from("MSE"), AttributeValue::F32(mse));
    }
    if let Some(m) = meta.world_to_camera.as_ref() {
        layer_attrs
            .other
            .insert(Text::from("worldToCamera"), matrix_to_attr(m));
    }
    if let Some(m) = meta.world_to_ndc.as_ref() {
        layer_attrs
            .other
            .insert(Text::from("worldToNDC"), matrix_to_attr(m));
    }
    for (key, value) in meta.string_attrs {
        layer_attrs
            .other
            .insert(Text::from(*key), AttributeValue::Text(Text::from(*value)));
    }

    // displayWindow covers the whole image: (0, 0)..(total_res.x - 1, total_res.y - 1).
    let display_window = IntegerBounds::new(
        Vec2(0, 0),
        Vec2(total_resolution.x as usize, total_resolution.y as usize),
    );

    let layer = Layer::new(
        (resolution.x as usize, resolution.y as usize),
        layer_attrs,
        Encoding::SMALL_LOSSLESS, // ZIP16 + scanline + INCREASING_Y (v4 compatible)
        AnyChannels::sort(any_channels),
    );
    let mut image = Image::from_layer(layer);
    image.attributes.display_window = display_window;

    image
        .write()
        .to_file(name)
        .map_err(|e| PbrtError::error(&e.to_string()))
}

/// RGBFilm path: 3 float channels (R/G/B), FLOAT samples, v4-compatible
/// metadata.
pub fn write_image_exr(
    name: &str,
    rgb: &[Float],
    output_bounds: &Bounds2i,
    total_resolution: &Point2i,
) -> std::result::Result<(), PbrtError> {
    write_image_exr_with_meta(
        name,
        rgb,
        output_bounds,
        total_resolution,
        &ExrExtraMeta::default(),
    )
}

/// RGBFilm path with extra metadata.
pub fn write_image_exr_with_meta(
    name: &str,
    rgb: &[Float],
    output_bounds: &Bounds2i,
    total_resolution: &Point2i,
    meta: &ExrExtraMeta,
) -> std::result::Result<(), PbrtError> {
    let resolution = output_bounds.diagonal();
    let pixels = (resolution.x * resolution.y) as usize;
    let mut r = vec![0.0f32; pixels];
    let mut g = vec![0.0f32; pixels];
    let mut b = vec![0.0f32; pixels];
    for i in 0..pixels {
        r[i] = rgb[3 * i + 0] as f32;
        g[i] = rgb[3 * i + 1] as f32;
        b[i] = rgb[3 * i + 2] as f32;
    }
    write_channels_inner(
        name,
        output_bounds,
        total_resolution,
        ExrPixelType::Float,
        &[("R", r), ("G", g), ("B", b)],
        meta,
    )
}

/// Arbitrary channel set + FLOAT samples + v4-compatible metadata.
/// Used by SpectralFilm and other non-HALF paths.
pub fn write_image_exr_channels(
    name: &str,
    output_bounds: &Bounds2i,
    total_resolution: &Point2i,
    channels: &[(&str, Vec<f32>)],
) -> std::result::Result<(), PbrtError> {
    write_channels_inner(
        name,
        output_bounds,
        total_resolution,
        ExrPixelType::Float,
        channels,
        &ExrExtraMeta::default(),
    )
}

/// Arbitrary channel set + HALF samples + v4-compatible metadata.
/// Used by `GBufferFilm` because pbrt-v4 writes that film as HALF.
pub fn write_image_exr_channels_half(
    name: &str,
    output_bounds: &Bounds2i,
    total_resolution: &Point2i,
    channels: &[(&str, Vec<f32>)],
    meta: &ExrExtraMeta,
) -> std::result::Result<(), PbrtError> {
    write_channels_inner(
        name,
        output_bounds,
        total_resolution,
        ExrPixelType::Half,
        channels,
        meta,
    )
}

/// Arbitrary channel set + FLOAT samples + string attributes (used for
/// SpectralFilm's `spectralLayoutVersion` and similar tags).
pub fn write_image_exr_channels_with_attrs(
    name: &str,
    output_bounds: &Bounds2i,
    total_resolution: &Point2i,
    channels: &[(&str, Vec<f32>)],
    attrs: &[(&str, &str)],
) -> std::result::Result<(), PbrtError> {
    let meta = ExrExtraMeta {
        string_attrs: attrs,
        ..Default::default()
    };
    write_channels_inner(
        name,
        output_bounds,
        total_resolution,
        ExrPixelType::Float,
        channels,
        &meta,
    )
}

/// Write arbitrary float channels using the metadata carried by `Image`.
/// String attributes are copied so the caller's metadata can remain borrowed.
pub fn write_image_exr_channels_with_metadata(
    name: &str,
    output_bounds: &Bounds2i,
    total_resolution: &Point2i,
    channels: &[(&str, Vec<f32>)],
    metadata: &ImageMetadata,
) -> std::result::Result<(), PbrtError> {
    write_image_exr_channels_with_metadata_as(
        name,
        output_bounds,
        total_resolution,
        channels,
        metadata,
        ExrPixelType::Float,
    )
}

pub fn write_image_exr_channels_with_metadata_as(
    name: &str,
    output_bounds: &Bounds2i,
    total_resolution: &Point2i,
    channels: &[(&str, Vec<f32>)],
    metadata: &ImageMetadata,
    pixel_type: ExrPixelType,
) -> std::result::Result<(), PbrtError> {
    let owned_attrs: Vec<(String, String)> = metadata
        .strings
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let attrs: Vec<(&str, &str)> = owned_attrs
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect();
    let meta = ExrExtraMeta {
        render_time_seconds: metadata.render_time_seconds.map(|value| value as f32),
        samples_per_pixel: metadata.samples_per_pixel,
        mse: metadata.mse.map(|value| value as f32),
        string_attrs: &attrs,
        ..Default::default()
    };
    write_channels_inner(
        name,
        output_bounds,
        total_resolution,
        pixel_type,
        channels,
        &meta,
    )
}
