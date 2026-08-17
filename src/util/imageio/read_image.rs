use super::color_encoding::ColorEncoding;
use super::read_image_pfm::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::spectrum::*;

use image::*;
use std::path::Path;

type Rgb16Image = ImageBuffer<Rgb<u16>, Vec<u16>>;

pub struct RawImage {
    pub data: RawImageData,
    pub resolution: Point2i,
    pub channels: usize,
}

pub enum RawImageData {
    F32(Vec<Float>),
    F16(Vec<half::f16>),
    U8 {
        data: Vec<u8>,
        encoding: ColorEncoding,
    },
}

impl RawImage {
    /// Return conventional channel names for formats supported by the
    /// `image` crate. EXR readers retain their named-channel metadata
    /// separately because EXR may contain arbitrary channel names.
    pub fn channel_names(&self) -> Vec<String> {
        match self.channels {
            1 => vec!["Y".to_string()],
            2 => vec!["Y".to_string(), "A".to_string()],
            3 => vec!["R".to_string(), "G".to_string(), "B".to_string()],
            4 => vec![
                "R".to_string(),
                "G".to_string(),
                "B".to_string(),
                "A".to_string(),
            ],
            count => (0..count).map(|i| format!("C{}", i)).collect(),
        }
    }

    pub fn channel(&self, pixel: usize, channel: usize) -> Float {
        let index = self.channels * pixel + channel;
        match &self.data {
            RawImageData::F32(data) => data[index],
            RawImageData::F16(data) => data[index].to_f32() as Float,
            RawImageData::U8 { data, encoding } => {
                let v = data[index] as Float / 255.0;
                encoding.to_linear(v)
            }
        }
    }

    pub fn data_f32(&self) -> Vec<Float> {
        let pixels = (self.resolution.x * self.resolution.y) as usize;
        let mut data = vec![0.0; pixels * self.channels];
        for i in 0..pixels {
            for c in 0..self.channels {
                data[self.channels * i + c] = self.channel(i, c);
            }
        }
        data
    }
}

fn convert_from_luma8(
    img: &image::GrayImage,
    encoding: ColorEncoding,
) -> (Vec<RGBSpectrum>, Point2i) {
    let (width, height) = img.dimensions();
    let mut spcs = vec![RGBSpectrum::zero(); (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = img[(x, y)];
            let mut r = pixel[0] as Float / 255.0;

            r = encoding.to_linear(r);

            spcs[index] = RGBSpectrum::rgb_from_rgb(&[r, r, r]);
        }
    }
    return (spcs, Point2i::from((width as i32, height as i32)));
}

fn convert_raw_from_luma8(img: &image::GrayImage, encoding: ColorEncoding) -> RawImage {
    let (width, height) = img.dimensions();
    RawImage {
        data: RawImageData::U8 {
            data: img.as_raw().clone(),
            encoding,
        },
        resolution: Point2i::from((width as i32, height as i32)),
        channels: 1,
    }
}

fn convert_from_lumaa8(
    img: &image::GrayAlphaImage,
    encoding: ColorEncoding,
) -> (Vec<RGBSpectrum>, Point2i) {
    let (width, height) = img.dimensions();
    let mut spcs = vec![RGBSpectrum::zero(); (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = img[(x, y)];
            let mut r = pixel[0] as Float / 255.0;
            let _a = pixel[1] as Float / 255.0; // Ignore alpha channel

            r = encoding.to_linear(r);

            spcs[index] = RGBSpectrum::rgb_from_rgb(&[r, r, r]);
        }
    }
    return (spcs, Point2i::from((width as i32, height as i32)));
}

fn convert_raw_from_lumaa8(img: &image::GrayAlphaImage, encoding: ColorEncoding) -> RawImage {
    let (width, height) = img.dimensions();
    RawImage {
        data: RawImageData::U8 {
            data: img.as_raw().clone(),
            encoding,
        },
        resolution: Point2i::from((width as i32, height as i32)),
        channels: 2,
    }
}

fn convert_from_rgb8(
    img: &image::RgbImage,
    encoding: ColorEncoding,
) -> (Vec<RGBSpectrum>, Point2i) {
    let (width, height) = img.dimensions();
    let mut spcs = vec![RGBSpectrum::zero(); (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = img[(x, y)];
            let mut r = pixel[0] as Float / 255.0;
            let mut g = pixel[1] as Float / 255.0;
            let mut b = pixel[2] as Float / 255.0;

            r = encoding.to_linear(r);
            g = encoding.to_linear(g);
            b = encoding.to_linear(b);

            spcs[index] = RGBSpectrum::rgb_from_rgb(&[r, g, b]);
        }
    }
    return (spcs, Point2i::from((width as i32, height as i32)));
}

fn convert_raw_from_rgb8(img: &image::RgbImage, encoding: ColorEncoding) -> RawImage {
    let (width, height) = img.dimensions();
    RawImage {
        data: RawImageData::U8 {
            data: img.as_raw().clone(),
            encoding,
        },
        resolution: Point2i::from((width as i32, height as i32)),
        channels: 3,
    }
}

fn convert_from_rgba8(
    img: &image::RgbaImage,
    encoding: ColorEncoding,
) -> (Vec<RGBSpectrum>, Point2i) {
    let (width, height) = img.dimensions();
    let mut spcs = vec![RGBSpectrum::zero(); (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = img[(x, y)];
            let mut r = pixel[0] as Float / 255.0;
            let mut g = pixel[1] as Float / 255.0;
            let mut b = pixel[2] as Float / 255.0;

            r = encoding.to_linear(r);
            g = encoding.to_linear(g);
            b = encoding.to_linear(b);

            spcs[index] = RGBSpectrum::rgb_from_rgb(&[r, g, b]);
        }
    }
    return (spcs, Point2i::from((width as i32, height as i32)));
}

fn convert_raw_from_rgba8(img: &image::RgbaImage, encoding: ColorEncoding) -> RawImage {
    let (width, height) = img.dimensions();
    RawImage {
        data: RawImageData::U8 {
            data: img.as_raw().clone(),
            encoding,
        },
        resolution: Point2i::from((width as i32, height as i32)),
        channels: 4,
    }
}

fn convert_from_rgb16(img: &Rgb16Image) -> (Vec<RGBSpectrum>, Point2i) {
    let (width, height) = img.dimensions();
    let mut spcs = vec![RGBSpectrum::zero(); (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = img.get_pixel(x, y);
            let r = pixel[0] as Float / 65535.0;
            let g = pixel[1] as Float / 65535.0;
            let b = pixel[2] as Float / 65535.0;
            spcs[index] = RGBSpectrum::new(r, g, b);
        }
    }
    return (spcs, Point2i::from((width as i32, height as i32)));
}

fn convert_raw_from_rgb16(img: &Rgb16Image) -> RawImage {
    let (width, height) = img.dimensions();
    let mut data = vec![half::f16::ZERO; (3 * width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = img.get_pixel(x, y);
            data[3 * index] = half::f16::from_f32(pixel[0] as f32 / 65535.0);
            data[3 * index + 1] = half::f16::from_f32(pixel[1] as f32 / 65535.0);
            data[3 * index + 2] = half::f16::from_f32(pixel[2] as f32 / 65535.0);
        }
    }
    RawImage {
        data: RawImageData::F16(data),
        resolution: Point2i::from((width as i32, height as i32)),
        channels: 3,
    }
}

fn convert_from_rgb32f(img: &image::Rgb32FImage) -> (Vec<RGBSpectrum>, Point2i) {
    let (width, height) = img.dimensions();
    let mut spcs = vec![RGBSpectrum::zero(); (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = img.get_pixel(x, y);
            let r = pixel[0] as Float;
            let g = pixel[1] as Float;
            let b = pixel[2] as Float;
            spcs[index] = RGBSpectrum::new(r, g, b);
        }
    }
    return (spcs, Point2i::from((width as i32, height as i32)));
}

fn convert_raw_from_rgb32f(img: &image::Rgb32FImage) -> RawImage {
    let (width, height) = img.dimensions();
    let mut data = vec![0.0; (3 * width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = img.get_pixel(x, y);
            data[3 * index] = pixel[0] as Float;
            data[3 * index + 1] = pixel[1] as Float;
            data[3 * index + 2] = pixel[2] as Float;
        }
    }
    RawImage {
        data: RawImageData::F32(data),
        resolution: Point2i::from((width as i32, height as i32)),
        channels: 3,
    }
}

fn convert_from_rgba32f(img: &image::Rgba32FImage) -> (Vec<RGBSpectrum>, Point2i) {
    let (width, height) = img.dimensions();
    let mut spcs = vec![RGBSpectrum::zero(); (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = img.get_pixel(x, y);
            let r = pixel[0] as Float;
            let g = pixel[1] as Float;
            let b = pixel[2] as Float;
            spcs[index] = RGBSpectrum::new(r, g, b);
        }
    }
    return (spcs, Point2i::from((width as i32, height as i32)));
}

fn convert_raw_from_rgba32f(img: &image::Rgba32FImage) -> RawImage {
    let (width, height) = img.dimensions();
    let mut data = vec![0.0; (4 * width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = img.get_pixel(x, y);
            data[4 * index] = pixel[0] as Float;
            data[4 * index + 1] = pixel[1] as Float;
            data[4 * index + 2] = pixel[2] as Float;
            data[4 * index + 3] = pixel[3] as Float;
        }
    }
    RawImage {
        data: RawImageData::F32(data),
        resolution: Point2i::from((width as i32, height as i32)),
        channels: 4,
    }
}

// use crate::image::*;
pub fn read_image_common(
    path: &Path,
    encoding: ColorEncoding,
) -> Result<(Vec<RGBSpectrum>, Point2i), PbrtError> {
    // The `image` crate only accepts 3-channel RGB for EXR and rejects
    // Y / RGBA / multi-layer files. Dispatch to the dedicated reader.
    if has_extension(path, "exr") {
        return super::read_image_exr::read_image_exr(path);
    }
    let r: Result<DynamicImage, ImageError> = image::open(path);
    match r {
        Ok(dimg) => match dimg {
            DynamicImage::ImageLuma8(img) => {
                return Ok(convert_from_luma8(&img, encoding));
            }
            DynamicImage::ImageLumaA8(img) => {
                return Ok(convert_from_lumaa8(&img, encoding));
            }
            DynamicImage::ImageRgb8(img) => {
                return Ok(convert_from_rgb8(&img, encoding));
            }
            DynamicImage::ImageRgba8(img) => {
                return Ok(convert_from_rgba8(&img, encoding));
            }
            DynamicImage::ImageRgb16(img) => {
                return Ok(convert_from_rgb16(&img));
            }
            DynamicImage::ImageRgb32F(img) => {
                return Ok(convert_from_rgb32f(&img));
            }
            DynamicImage::ImageRgba32F(img) => {
                return Ok(convert_from_rgba32f(&img));
            }
            _ => {
                let msg = format!("This file is not supported: {}", path.to_string_lossy());
                return Err(PbrtError::from(msg));
            }
        },
        Err(e) => {
            return Err(PbrtError::from(e.to_string()));
        }
    };
}

fn has_extension(path: &Path, ext: &str) -> bool {
    return path.extension().unwrap_or_default() == ext;
}

pub fn read_image_with_encoding(
    name: &str,
    encoding: ColorEncoding,
) -> Result<(Vec<RGBSpectrum>, Point2i), PbrtError> {
    let path = Path::new(name);
    if !path.exists() {
        return Err(PbrtError::from(format!("File not found: {}", name)));
    }
    if has_extension(path, "pfm") {
        return read_image_pfm(name);
    } else {
        return read_image_common(path, encoding);
    }
}

pub fn read_raw_image_with_encoding(
    name: &str,
    encoding: ColorEncoding,
) -> Result<RawImage, PbrtError> {
    let path = Path::new(name);
    if !path.exists() {
        return Err(PbrtError::from(format!("File not found: {}", name)));
    }
    if has_extension(path, "exr") {
        return super::read_image_exr::read_raw_image_exr(path);
    }
    if has_extension(path, "pfm") {
        let (spectra, resolution) = read_image_pfm(name)?;
        let mut data = vec![0.0; 3 * spectra.len()];
        for (index, spectrum) in spectra.iter().enumerate() {
            let rgb = spectrum.to_rgb();
            data[3 * index] = rgb[0];
            data[3 * index + 1] = rgb[1];
            data[3 * index + 2] = rgb[2];
        }
        return Ok(RawImage {
            data: RawImageData::F32(data),
            resolution,
            channels: 3,
        });
    }

    let image = image::open(path)?;
    let raw = match image {
        DynamicImage::ImageLuma8(img) => convert_raw_from_luma8(&img, encoding),
        DynamicImage::ImageLumaA8(img) => convert_raw_from_lumaa8(&img, encoding),
        DynamicImage::ImageRgb8(img) => convert_raw_from_rgb8(&img, encoding),
        DynamicImage::ImageRgba8(img) => convert_raw_from_rgba8(&img, encoding),
        DynamicImage::ImageRgb16(img) => convert_raw_from_rgb16(&img),
        DynamicImage::ImageRgb32F(img) => convert_raw_from_rgb32f(&img),
        DynamicImage::ImageRgba32F(img) => convert_raw_from_rgba32f(&img),
        _ => {
            return Err(PbrtError::from(format!(
                "This file is not supported: {}",
                name
            )));
        }
    };
    Ok(raw)
}

pub fn read_image(name: &str) -> Result<(Vec<RGBSpectrum>, Point2i), PbrtError> {
    read_image_with_encoding(name, ColorEncoding::Linear)
}

fn convert_f32_from_rgba32f(img: &image::Rgb32FImage) -> (Vec<f32>, Point2i) {
    let (width, height) = img.dimensions();
    let mut values = vec![0.0; (3 * width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = img.get_pixel(x, y);
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];
            values[3 * index + 0] = r;
            values[3 * index + 1] = g;
            values[3 * index + 2] = b;
        }
    }
    return (values, Point2i::from((width as i32, height as i32)));
}

pub fn read_cache_image(name: &str) -> Result<(Vec<f32>, Point2i), PbrtError> {
    let r: Result<DynamicImage, ImageError> = image::open(name);
    match r {
        Ok(dimg) => match dimg {
            DynamicImage::ImageRgb32F(img) => {
                return Ok(convert_f32_from_rgba32f(&img));
            }
            _ => {
                return Err(PbrtError::from("This file is not supported."));
            }
        },
        Err(e) => {
            return Err(PbrtError::from(e.to_string()));
        }
    };
}
