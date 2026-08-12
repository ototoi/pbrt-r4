//! Dedicated EXR reader built directly on the `exr` crate.
//!
//! Going through `image::open` only handles **3-channel RGB** EXRs;
//! Y (luma), RGBA, or any other channel layout is rejected with
//! `"image does not contain non-deep rgb channels"`. Watercolor's
//! `Spot_Floor_02x11.exr` (Y-only) hit exactly that case.
//!
//! Here we:
//!  - read RGB if R/G/B are all present;
//!  - otherwise fall back to Y as grayscale `(Y, Y, Y)`;
//!  - return an explicit "unsupported" error for any other layout.
//!
//! Deep EXR files (`deepscanline` / `deeptile` `type`) are rejected
//! implicitly via `.no_deep_data()`.

use super::read_image::{RawImage, RawImageData};
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::image::ImageMetadata;
use crate::util::spectrum::*;

use exr::meta::attribute::{AttributeValue, Text};
use exr::prelude::{FlatSamples, ReadChannels, ReadLayers};

use std::path::Path;

/// Read an EXR and return `(Vec<RGBSpectrum>, Point2i)`. Supports R/G/B
/// or Y-only inputs. Colour-space metadata is dropped; use
/// `read_image_exr_with_metadata` when metadata is needed.
pub fn read_image_exr(path: &Path) -> Result<(Vec<RGBSpectrum>, Point2i), PbrtError> {
    let (texels, resolution, _cs) = read_image_exr_with_metadata(path)?;
    Ok((texels, resolution))
}

/// pbrt-v4 `Image::Read` (util/image.cpp) wraps the pixel data and
/// metadata into `ImageAndMetadata`; `ImageMetadata::GetColorSpace`
/// then resolves the chromaticities to a named `RGBColorSpace*` (or
/// falls back to sRGB on miss). r4 collapses that into one call: the
/// returned colour space is what pbrt-v4 would have selected.
pub fn read_image_exr_with_metadata(
    path: &Path,
) -> Result<(Vec<RGBSpectrum>, Point2i, &'static RGBColorSpace), PbrtError> {
    let (width, height, channels, metadata) = read_exr_channels(path)?;
    let total = width * height;
    let (r, g, b) = pick_rgb_or_y(channels, path)?;
    let mut spcs = vec![RGBSpectrum::zero(); total];
    for i in 0..total {
        spcs[i] = RGBSpectrum::rgb_from_rgb(&[r[i] as Float, g[i] as Float, b[i] as Float]);
    }
    // pbrt-v4 `ImageMetadata::GetColorSpace` (util/image.cpp:35-40):
    // fall back to sRGB whenever the EXR is silent on colour space or
    // its chromaticities don't match a known table.
    let color_space = metadata.color_space.unwrap_or(&SRGB);
    Ok((
        spcs,
        Point2i::from((width as i32, height as i32)),
        color_space,
    ))
}

/// Raw-image variant that preserves the channel count: returns 1 ch
/// for Y-only inputs and preserves RGB/RGBA or arbitrary named channels.
/// The result can feed
/// `scalarize_raw_image` for float-texture pipelines.
pub fn read_raw_image_exr(path: &Path) -> Result<RawImage, PbrtError> {
    Ok(read_raw_image_exr_with_channels(path)?.0)
}

/// Read an EXR while preserving the channel names from its layer.
pub fn read_raw_image_exr_with_channels(path: &Path) -> Result<(RawImage, Vec<String>), PbrtError> {
    let (raw, names, _metadata) = read_raw_image_exr_with_channels_and_metadata(path)?;
    Ok((raw, names))
}

/// Read an EXR while preserving channels and the metadata needed by image
/// comparison and conversion tools.
pub fn read_raw_image_exr_with_channels_and_metadata(
    path: &Path,
) -> Result<(RawImage, Vec<String>, ImageMetadata), PbrtError> {
    let (width, height, channels, _metadata) = read_exr_channels(path)?;
    let total = width * height;
    let channel_names = channels.names.clone();
    // For Y-only inputs (luminance / displacement / alpha) we keep a
    // single channel so downstream consumers like `scalarize_raw_image`
    // don't waste work on duplicated lanes.
    let (data, ch_count, channel_names) =
        if channels.r.is_none() && channels.g.is_none() && channels.b.is_none() {
            if let Some(y) = channels.y {
                (y, 1usize, vec!["Y".to_string()])
            } else if let Some(a) = channels.a {
                (a, 1usize, vec!["A".to_string()])
            } else if !channels.other.is_empty() {
                let count = channels.other.len();
                let mut data = vec![0.0 as Float; count * total];
                for (channel, samples) in channels.other.iter().enumerate() {
                    for pixel in 0..total {
                        data[pixel * count + channel] = samples[pixel];
                    }
                }
                (data, count, channel_names)
            } else {
                return Err(PbrtError::error(&format!(
                    "EXR \"{}\" has no R/G/B or Y channels.",
                    path.display()
                )));
            }
        } else if let (Some(r), Some(g), Some(b), Some(a)) = (
            channels.r.clone(),
            channels.g.clone(),
            channels.b.clone(),
            channels.a.clone(),
        ) {
            let mut data = vec![0.0 as Float; 4 * total];
            for i in 0..total {
                data[4 * i] = r[i];
                data[4 * i + 1] = g[i];
                data[4 * i + 2] = b[i];
                data[4 * i + 3] = a[i];
            }
            (
                data,
                4,
                vec![
                    "R".to_string(),
                    "G".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                ],
            )
        } else {
            // R/G/B (ignoring any incidental Y) → 3-channel interleaved.
            let (r, g, b) = pick_rgb_or_y(channels, path)?;
            let mut data = vec![0.0 as Float; 3 * total];
            for i in 0..total {
                data[3 * i + 0] = r[i] as Float;
                data[3 * i + 1] = g[i] as Float;
                data[3 * i + 2] = b[i] as Float;
            }
            (
                data,
                3,
                vec!["R".to_string(), "G".to_string(), "B".to_string()],
            )
        };
    Ok((
        RawImage {
            data: RawImageData::F32(data),
            resolution: Point2i::from((width as i32, height as i32)),
            channels: ch_count,
        },
        channel_names,
        _metadata,
    ))
}

struct ExrChannels {
    r: Option<Vec<Float>>,
    g: Option<Vec<Float>>,
    b: Option<Vec<Float>>,
    a: Option<Vec<Float>>,
    y: Option<Vec<Float>>,
    names: Vec<String>,
    other: Vec<Vec<Float>>,
}

/// Chromaticities attribute extracted from an EXR header, used to
/// match the image against a known `RGBColorSpace`. `None` indicates
/// the EXR did not carry the attribute (caller should default to
/// sRGB, matching pbrt-v4 `ImageMetadata::GetColorSpace`).
#[derive(Clone, Copy, Debug)]
pub struct ExrChromaticities {
    pub red: [Float; 2],
    pub green: [Float; 2],
    pub blue: [Float; 2],
    pub white: [Float; 2],
}

fn read_exr_channels(path: &Path) -> Result<(usize, usize, ExrChannels, ImageMetadata), PbrtError> {
    let image = exr::prelude::read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .first_valid_layer()
        .all_attributes()
        .from_file(path)
        .map_err(|e| {
            PbrtError::error(&format!(
                "Failed to decode EXR \"{}\": {}",
                path.display(),
                e
            ))
        })?;

    let chromaticities = image.attributes.chromaticities.map(|c| ExrChromaticities {
        red: [c.red.x() as Float, c.red.y() as Float],
        green: [c.green.x() as Float, c.green.y() as Float],
        blue: [c.blue.x() as Float, c.blue.y() as Float],
        white: [c.white.x() as Float, c.white.y() as Float],
    });

    let metadata = exr_metadata(
        &image.attributes,
        &image.layer_data.attributes,
        image.layer_data.size,
        chromaticities,
    );
    let layer = image.layer_data;
    let width = layer.size.0;
    let height = layer.size.1;
    let total = width * height;

    let mut r = None;
    let mut g = None;
    let mut b = None;
    let mut a = None;
    let mut y = None;
    let mut names = Vec::new();
    let mut other = Vec::new();
    for chan in layer.channel_data.list.iter() {
        let name = chan.name.to_string();
        names.push(name.clone());
        let samples = flat_samples_to_floats(&chan.sample_data, total);
        match name.as_str() {
            "R" => r = Some(samples),
            "G" => g = Some(samples),
            "B" => b = Some(samples),
            "A" => a = Some(samples),
            "Y" => y = Some(samples),
            _ => other.push(samples),
        }
    }
    Ok((
        width,
        height,
        ExrChannels {
            r,
            g,
            b,
            a,
            y,
            names,
            other,
        },
        metadata,
    ))
}

fn exr_metadata(
    image_attributes: &exr::meta::header::ImageAttributes,
    layer_attributes: &exr::meta::header::LayerAttributes,
    layer_size: exr::prelude::Vec2<usize>,
    chromaticities: Option<ExrChromaticities>,
) -> ImageMetadata {
    let attribute = |name: &str| {
        layer_attributes
            .other
            .get(&Text::from(name))
            .or_else(|| image_attributes.other.get(&Text::from(name)))
    };
    let mut metadata = ImageMetadata {
        color_space: chromaticities
            .and_then(|c| lookup_color_space(c.red, c.green, c.blue, c.white)),
        ..ImageMetadata::default()
    };
    if let Some(AttributeValue::F32(value)) = attribute("renderTimeSeconds") {
        metadata.render_time_seconds = Some(*value as Float);
    }
    if let Some(AttributeValue::I32(value)) = attribute("samplesPerPixel") {
        metadata.samples_per_pixel = Some(*value);
    }
    if let Some(AttributeValue::F32(value)) = attribute("MSE") {
        metadata.mse = Some(*value as Float);
    }
    for (name, value) in image_attributes
        .other
        .iter()
        .chain(layer_attributes.other.iter())
    {
        if let AttributeValue::Text(value) = value {
            metadata.strings.insert(name.to_string(), value.to_string());
        }
    }
    let display = image_attributes.display_window;
    metadata.full_resolution = Some(Point2i::new(
        display.size.x() as i32,
        display.size.y() as i32,
    ));
    let data = layer_attributes.layer_position;
    metadata.pixel_bounds = Some(Bounds2i::from((
        (data.x(), data.y()),
        (
            data.x() + layer_size.0 as i32,
            data.y() + layer_size.1 as i32,
        ),
    )));
    metadata
}

fn pick_rgb_or_y(
    channels: ExrChannels,
    path: &Path,
) -> Result<(Vec<Float>, Vec<Float>, Vec<Float>), PbrtError> {
    if let (Some(r), Some(g), Some(b)) =
        (channels.r.clone(), channels.g.clone(), channels.b.clone())
    {
        return Ok((r, g, b));
    }
    if let Some(y) = channels.y {
        return Ok((y.clone(), y.clone(), y));
    }
    Err(PbrtError::error(&format!(
        "EXR \"{}\" has no R/G/B or Y channels.",
        path.display()
    )))
}

fn flat_samples_to_floats(samples: &FlatSamples, expected: usize) -> Vec<Float> {
    match samples {
        FlatSamples::F16(v) => {
            let mut out = vec![0.0 as Float; expected.min(v.len())];
            for (i, h) in v.iter().take(out.len()).enumerate() {
                out[i] = h.to_f32() as Float;
            }
            out
        }
        FlatSamples::F32(v) => v
            .iter()
            .take(expected)
            .map(|&x| x as Float)
            .collect::<Vec<Float>>(),
        FlatSamples::U32(v) => v
            .iter()
            .take(expected)
            .map(|&x| x as Float)
            .collect::<Vec<Float>>(),
    }
}
