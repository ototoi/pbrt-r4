use crate::util::base::*;
use crate::util::error::PbrtError;
use crate::util::geometry::Bounds2i;
use crate::util::imageio::write_image_exr::{
    write_image_exr_channels_with_metadata_as, ExrPixelType,
};
use crate::util::spectrum::rgb_to_spectrum::{RGBColorSpace, SRGB};
use crate::util::spectrum::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelFormat {
    U256,
    Half,
    Float,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageWrapMode {
    Black,
    Clamp,
    Repeat,
    OctahedralSphere,
}

impl PixelFormat {
    pub fn texel_bytes(self) -> usize {
        match self {
            Self::U256 => 1,
            Self::Half => 2,
            Self::Float => 4,
        }
    }
}

#[derive(Clone, Default)]
pub struct ImageMetadata {
    pub render_time_seconds: Option<Float>,
    pub pixel_bounds: Option<Bounds2i>,
    pub full_resolution: Option<Point2i>,
    pub samples_per_pixel: Option<i32>,
    pub mse: Option<Float>,
    pub color_space: Option<&'static RGBColorSpace>,
    pub strings: HashMap<String, String>,
}

/// pbrt-v4 `Image` equivalent for lights/textures that need direct
/// access to image pixels rather than a filtered pyramid.
#[derive(Clone)]
pub struct Image {
    resolution: Point2i,
    texels: Vec<RGBSpectrum>,
    color_space: &'static RGBColorSpace,
    channel_names: Vec<String>,
    channel_data: Vec<Float>,
    format: PixelFormat,
    metadata: ImageMetadata,
}

impl Image {
    pub fn new(resolution: Point2i, texels: Vec<RGBSpectrum>) -> Self {
        let expected = (resolution.x * resolution.y) as usize;
        debug_assert_eq!(texels.len(), expected);
        Self {
            resolution,
            texels,
            color_space: &SRGB,
            channel_names: vec!["R".to_string(), "G".to_string(), "B".to_string()],
            channel_data: Vec::new(),
            format: PixelFormat::Float,
            metadata: ImageMetadata {
                color_space: Some(&SRGB),
                ..ImageMetadata::default()
            },
        }
    }

    pub fn with_color_space(
        resolution: Point2i,
        texels: Vec<RGBSpectrum>,
        color_space: &'static RGBColorSpace,
    ) -> Result<Self, PbrtError> {
        let expected = (resolution.x * resolution.y) as usize;
        if texels.len() != expected {
            return Err(PbrtError::error(&format!(
                "image texel count mismatch: expected {}, got {}",
                expected,
                texels.len()
            )));
        }
        Ok(Self {
            resolution,
            texels,
            color_space,
            channel_names: vec!["R".to_string(), "G".to_string(), "B".to_string()],
            channel_data: Vec::new(),
            format: PixelFormat::Float,
            metadata: ImageMetadata {
                color_space: Some(color_space),
                ..ImageMetadata::default()
            },
        })
    }

    pub fn try_with_color_space(
        resolution: Point2i,
        texels: Vec<RGBSpectrum>,
        color_space: &'static RGBColorSpace,
    ) -> Result<Self, PbrtError> {
        let expected = (resolution.x * resolution.y) as usize;
        if texels.len() != expected {
            return Err(PbrtError::error(&format!(
                "image texel count mismatch: expected {}, got {}",
                expected,
                texels.len()
            )));
        }
        Self::with_color_space(resolution, texels, color_space)
    }

    /// Construct an image with arbitrary float channels, corresponding to
    /// pbrt-v4 `Image(PixelFormat::Float, ...)` and its channel layout.
    pub fn from_channels(
        resolution: Point2i,
        channel_names: Vec<String>,
        channel_data: Vec<Float>,
    ) -> Self {
        Self::from_channels_with_format(resolution, channel_names, channel_data, PixelFormat::Float)
    }

    pub fn from_channels_with_format(
        resolution: Point2i,
        channel_names: Vec<String>,
        channel_data: Vec<Float>,
        format: PixelFormat,
    ) -> Self {
        let expected = (resolution.x * resolution.y) as usize * channel_names.len();
        assert_eq!(channel_data.len(), expected);
        let channel_data = channel_data
            .into_iter()
            .map(|value| match format {
                PixelFormat::U256 => ((value.clamp(0.0, 1.0) * 255.0).round() / 255.0),
                PixelFormat::Half => half::f16::from_f32(value as f32).to_f32() as Float,
                PixelFormat::Float => value,
            })
            .collect();
        Self {
            resolution,
            texels: Vec::new(),
            color_space: &SRGB,
            channel_names,
            channel_data,
            format,
            metadata: ImageMetadata::default(),
        }
    }

    pub fn format(&self) -> PixelFormat {
        self.format
    }

    pub fn metadata(&self) -> &ImageMetadata {
        &self.metadata
    }

    pub fn metadata_mut(&mut self) -> &mut ImageMetadata {
        &mut self.metadata
    }

    /// Return the half-open pixel rectangle selected by `bounds`, matching
    /// pbrt-v4 `Image::Crop`.
    pub fn crop(&self, bounds: Bounds2i) -> Result<Self, PbrtError> {
        assert!(bounds.min.x >= 0);
        assert!(bounds.min.y >= 0);
        assert!(bounds.max.x <= self.resolution.x);
        assert!(bounds.max.y <= self.resolution.y);
        assert!(bounds.min.x <= bounds.max.x);
        assert!(bounds.min.y <= bounds.max.y);

        let resolution = Point2i::new(bounds.max.x - bounds.min.x, bounds.max.y - bounds.min.y);
        let mut image = if self.channel_data.is_empty() {
            let mut texels = Vec::with_capacity((resolution.x * resolution.y) as usize);
            for y in bounds.min.y..bounds.max.y {
                for x in bounds.min.x..bounds.max.x {
                    texels.push(self.texels[(y * self.resolution.x + x) as usize]);
                }
            }
            Self::with_color_space(resolution, texels, self.color_space)?
        } else {
            let mut data =
                Vec::with_capacity((resolution.x * resolution.y) as usize * self.n_channels());
            for y in bounds.min.y..bounds.max.y {
                for x in bounds.min.x..bounds.max.x {
                    let pixel = (y * self.resolution.x + x) as usize;
                    for channel in 0..self.n_channels() {
                        data.push(self.channel(pixel, channel));
                    }
                }
            }
            Self {
                resolution,
                texels: Vec::new(),
                color_space: self.color_space,
                channel_names: self.channel_names.clone(),
                channel_data: data,
                format: self.format,
                metadata: self.metadata.clone(),
            }
        };
        image.metadata.pixel_bounds = Some(bounds);
        image.metadata.full_resolution = self.metadata.full_resolution.or(Some(self.resolution));
        Ok(image)
    }

    /// Select and reorder channels, matching v4 `Image::SelectChannels`.
    pub fn select_channels(&self, channels: &[usize]) -> Self {
        assert!(!channels.is_empty());
        for &channel in channels {
            assert!(channel < self.n_channels());
        }
        let pixels = (self.resolution.x * self.resolution.y) as usize;
        let data: Vec<Float> = (0..pixels)
            .flat_map(|pixel| {
                channels
                    .iter()
                    .map(move |&channel| self.channel(pixel, channel))
            })
            .collect();
        let names = channels
            .iter()
            .map(|&channel| self.channel_names[channel].clone())
            .collect();
        Self {
            resolution: self.resolution,
            texels: Vec::new(),
            color_space: self.color_space,
            channel_names: names,
            channel_data: data,
            format: self.format,
            metadata: self.metadata.clone(),
        }
    }

    /// Write the image through the v4-compatible EXR channel path.
    pub fn write_exr(
        &self,
        name: &str,
        output_bounds: &Bounds2i,
        total_resolution: &Point2i,
    ) -> Result<(), PbrtError> {
        let pixel_type = match self.format {
            PixelFormat::Float => ExrPixelType::Float,
            PixelFormat::Half => ExrPixelType::Half,
            PixelFormat::U256 => {
                return Err(PbrtError::error(
                    "U256 images cannot be written through the EXR float/half path.",
                ));
            }
        };
        let pixels = (self.resolution.x * self.resolution.y) as usize;
        let channels: Vec<(&str, Vec<f32>)> = self
            .channel_names
            .iter()
            .enumerate()
            .map(|(channel, name)| {
                let values = (0..pixels)
                    .map(|pixel| self.channel(pixel, channel) as f32)
                    .collect();
                (name.as_str(), values)
            })
            .collect();
        write_image_exr_channels_with_metadata_as(
            name,
            output_bounds,
            total_resolution,
            &channels,
            &self.metadata,
            pixel_type,
        )
    }

    pub fn resolution(&self) -> Point2i {
        self.resolution
    }

    pub fn width(&self) -> i32 {
        self.resolution.x
    }

    pub fn height(&self) -> i32 {
        self.resolution.y
    }

    pub fn lookup_nearest(&self, uv: &Point2f) -> RGBSpectrum {
        let x = ((uv.x * self.resolution.x as Float) as i32).clamp(0, self.resolution.x - 1);
        let y = ((uv.y * self.resolution.y as Float) as i32).clamp(0, self.resolution.y - 1);
        self.texels[(y * self.resolution.x + x) as usize]
    }

    pub fn color_space(&self) -> &'static RGBColorSpace {
        self.color_space
    }

    pub fn channel_names(&self) -> &[String] {
        &self.channel_names
    }

    pub fn n_channels(&self) -> usize {
        self.channel_names.len()
    }

    /// Return a channel value using v4's pixel-major channel layout.
    pub fn channel(&self, pixel: usize, channel: usize) -> Float {
        assert!(channel < self.n_channels());
        if self.channel_data.is_empty() {
            self.texels[pixel].to_rgb()[channel]
        } else {
            self.channel_data[pixel * self.n_channels() + channel]
        }
    }

    pub fn lookup_nearest_channel(&self, uv: &Point2f, channel: usize) -> Float {
        self.lookup_nearest_channel_with_wrap(
            uv,
            channel,
            ImageWrapMode::Clamp,
            ImageWrapMode::Clamp,
        )
    }

    pub fn lookup_nearest_channel_with_wrap(
        &self,
        uv: &Point2f,
        channel: usize,
        wrap_x: ImageWrapMode,
        wrap_y: ImageWrapMode,
    ) -> Float {
        let x = (uv.x * self.resolution.x as Float) as i32;
        let y = (uv.y * self.resolution.y as Float) as i32;
        match remap_pixel(x, y, self.resolution, wrap_x, wrap_y) {
            Some((x, y)) => self.channel((y * self.resolution.x + x) as usize, channel),
            None => 0.0,
        }
    }

    /// Bilinearly interpolate one channel with clamp wrapping, matching the
    /// default behavior of v4 `Image::BilerpChannel`.
    pub fn bilerp_channel(&self, uv: &Point2f, channel: usize) -> Float {
        self.bilerp_channel_with_wrap(uv, channel, ImageWrapMode::Clamp, ImageWrapMode::Clamp)
    }

    pub fn bilerp_channel_with_wrap(
        &self,
        uv: &Point2f,
        channel: usize,
        wrap_x: ImageWrapMode,
        wrap_y: ImageWrapMode,
    ) -> Float {
        assert!(channel < self.n_channels());
        let x = uv.x * self.resolution.x as Float - 0.5;
        let y = uv.y * self.resolution.y as Float - 0.5;
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let dx = x - x.floor();
        let dy = y - y.floor();
        let sample = |px: i32, py: i32| match remap_pixel(px, py, self.resolution, wrap_x, wrap_y) {
            Some((px, py)) => self.channel((py * self.resolution.x + px) as usize, channel),
            None => 0.0,
        };
        let v00 = sample(x0, y0);
        let v10 = sample(x0 + 1, y0);
        let v01 = sample(x0, y0 + 1);
        let v11 = sample(x0 + 1, y0 + 1);
        (1.0 - dx) * (1.0 - dy) * v00
            + dx * (1.0 - dy) * v10
            + (1.0 - dx) * dy * v01
            + dx * dy * v11
    }

    pub fn average_channels(&self, channels: &[usize]) -> Vec<Float> {
        assert!(!channels.is_empty());
        let pixels = (self.resolution.x * self.resolution.y) as usize;
        channels
            .iter()
            .map(|&channel| {
                let sum: Float = (0..pixels).map(|pixel| self.channel(pixel, channel)).sum();
                sum / pixels as Float
            })
            .collect()
    }

    pub fn mse_channels(&self, reference: &Self, channels: &[usize]) -> Vec<Float> {
        self.assert_compatible(reference);
        let pixels = (self.resolution.x * self.resolution.y) as usize;
        channels
            .iter()
            .map(|&channel| {
                (0..pixels)
                    .map(|pixel| {
                        let delta =
                            self.channel(pixel, channel) - reference.channel(pixel, channel);
                        delta * delta
                    })
                    .sum::<Float>()
                    / pixels as Float
            })
            .collect()
    }

    pub fn mae_channels(&self, reference: &Self, channels: &[usize]) -> Vec<Float> {
        self.assert_compatible(reference);
        let pixels = (self.resolution.x * self.resolution.y) as usize;
        channels
            .iter()
            .map(|&channel| {
                (0..pixels)
                    .map(|pixel| {
                        (self.channel(pixel, channel) - reference.channel(pixel, channel)).abs()
                    })
                    .filter(|error| error.is_finite())
                    .sum::<Float>()
                    / pixels as Float
            })
            .collect()
    }

    pub fn mrse_channels(&self, reference: &Self, channels: &[usize]) -> Vec<Float> {
        self.assert_compatible(reference);
        let pixels = (self.resolution.x * self.resolution.y) as usize;
        channels
            .iter()
            .map(|&channel| {
                (0..pixels)
                    .map(|pixel| {
                        let delta =
                            self.channel(pixel, channel) - reference.channel(pixel, channel);
                        let denominator = reference.channel(pixel, channel) + 0.01;
                        delta * delta / (denominator * denominator)
                    })
                    .filter(|error| error.is_finite())
                    .sum::<Float>()
                    / pixels as Float
            })
            .collect()
    }

    pub fn has_any_nan_pixels(&self) -> bool {
        self.channel_data.iter().any(|value| value.is_nan())
            || self
                .texels
                .iter()
                .any(|texel| texel.to_rgb().iter().any(|value| value.is_nan()))
    }

    pub fn has_any_infinite_pixels(&self) -> bool {
        self.channel_data.iter().any(|value| value.is_infinite())
            || self
                .texels
                .iter()
                .any(|texel| texel.to_rgb().iter().any(|value| value.is_infinite()))
    }

    fn assert_compatible(&self, reference: &Self) {
        assert_eq!(self.resolution, reference.resolution);
        assert_eq!(self.n_channels(), reference.n_channels());
    }

    pub fn texels(&self) -> &[RGBSpectrum] {
        &self.texels
    }

    /// Per-channel mean squared error against an image with the same layout.
    /// This is the RGB equivalent of pbrt-v4 `Image::MSE` for the current
    /// direct-access image representation.
    pub fn mse(&self, reference: &Self) -> RGBSpectrum {
        assert_eq!(self.resolution, reference.resolution);
        let mut sum = [0.0; 3];
        for (pixel, reference_pixel) in self.texels.iter().zip(&reference.texels) {
            let a = pixel.to_rgb();
            let b = reference_pixel.to_rgb();
            for c in 0..3 {
                let delta = a[c] - b[c];
                sum[c] += delta * delta;
            }
        }
        let scale = 1.0 / self.texels.len() as Float;
        RGBSpectrum::new(sum[0] * scale, sum[1] * scale, sum[2] * scale)
    }

    /// Per-channel mean absolute error, matching pbrt-v4 `Image::MAE`.
    pub fn mae(&self, reference: &Self) -> RGBSpectrum {
        assert_eq!(self.resolution, reference.resolution);
        let mut sum = [0.0; 3];
        for (pixel, reference_pixel) in self.texels.iter().zip(&reference.texels) {
            let a = pixel.to_rgb();
            let b = reference_pixel.to_rgb();
            for c in 0..3 {
                let error = (a[c] - b[c]).abs();
                if error.is_finite() {
                    sum[c] += error;
                }
            }
        }
        let scale = 1.0 / self.texels.len() as Float;
        RGBSpectrum::new(sum[0] * scale, sum[1] * scale, sum[2] * scale)
    }

    /// Per-channel mean relative squared error, matching pbrt-v4 `Image::MRSE`.
    pub fn mrse(&self, reference: &Self) -> RGBSpectrum {
        assert_eq!(self.resolution, reference.resolution);
        let mut sum = [0.0; 3];
        for (pixel, reference_pixel) in self.texels.iter().zip(&reference.texels) {
            let a = pixel.to_rgb();
            let b = reference_pixel.to_rgb();
            for c in 0..3 {
                let error = (a[c] - b[c]).powi(2) / (b[c] + 0.01).powi(2);
                if error.is_finite() {
                    sum[c] += error;
                }
            }
        }
        let scale = 1.0 / self.texels.len() as Float;
        RGBSpectrum::new(sum[0] * scale, sum[1] * scale, sum[2] * scale)
    }

    /// Per-channel average, matching pbrt-v4 `Image::Average`.
    pub fn average(&self) -> RGBSpectrum {
        let mut sum = [0.0; 3];
        for pixel in &self.texels {
            let rgb = pixel.to_rgb();
            for c in 0..3 {
                sum[c] += rgb[c];
            }
        }
        let scale = 1.0 / self.texels.len() as Float;
        RGBSpectrum::new(sum[0] * scale, sum[1] * scale, sum[2] * scale)
    }
}

fn remap_pixel(
    mut x: i32,
    mut y: i32,
    resolution: Point2i,
    wrap_x: ImageWrapMode,
    wrap_y: ImageWrapMode,
) -> Option<(i32, i32)> {
    if wrap_x == ImageWrapMode::OctahedralSphere || wrap_y == ImageWrapMode::OctahedralSphere {
        assert_eq!(wrap_x, ImageWrapMode::OctahedralSphere);
        assert_eq!(wrap_y, ImageWrapMode::OctahedralSphere);
        if x < 0 {
            x = -x;
            y = resolution.y - 1 - y;
        } else if x >= resolution.x {
            x = 2 * resolution.x - 1 - x;
            y = resolution.y - 1 - y;
        }
        if y < 0 {
            x = resolution.x - 1 - x;
            y = -y;
        } else if y >= resolution.y {
            x = resolution.x - 1 - x;
            y = 2 * resolution.y - 1 - y;
        }
        if resolution.x == 1 {
            x = 0;
        }
        if resolution.y == 1 {
            y = 0;
        }
        return Some((x, y));
    }
    for (coordinate, size, mode) in [
        (&mut x, resolution.x, wrap_x),
        (&mut y, resolution.y, wrap_y),
    ] {
        if *coordinate >= 0 && *coordinate < size {
            continue;
        }
        match mode {
            ImageWrapMode::Black => return None,
            ImageWrapMode::Clamp => *coordinate = (*coordinate).clamp(0, size - 1),
            ImageWrapMode::Repeat => {
                *coordinate = (*coordinate % size + size) % size;
            }
            ImageWrapMode::OctahedralSphere => unreachable!(),
        }
    }
    Some((x, y))
}
