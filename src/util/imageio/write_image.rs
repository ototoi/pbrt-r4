use super::write_image_exr::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;

use image::*;
use std::path::Path;

impl From<image::ImageError> for PbrtError {
    fn from(value: image::ImageError) -> Self {
        let msg = value.to_string();
        return PbrtError::error(&msg);
    }
}

fn to_byte(v: Float) -> u8 {
    Float::clamp(255.0 * gamma_correct(v), 0.0, 255.0) as u8
}

pub fn write_image_bytes(
    name: &str,
    rgb: &[Float],
    output_bounds: &Bounds2i,
    _total_resolution: &Point2i,
) -> Result<(), PbrtError> {
    let resolution = output_bounds.diagonal();
    let mut byte_img: Vec<u8> = vec![0; (resolution.x * resolution.y * 3) as usize];

    {
        let x0: u32 = output_bounds.min.x as u32;
        let x1: u32 = output_bounds.max.x as u32;
        let y0: u32 = output_bounds.min.y as u32;
        let y1: u32 = output_bounds.max.y as u32;

        let width = x1 - x0;

        for y in y0..y1 {
            let yy = y - y0;
            for x in x0..x1 {
                let xx = x - x0;
                let index: usize = (yy * width + xx) as usize;
                byte_img[3 * index + 0] = to_byte(rgb[3 * index + 0]);
                byte_img[3 * index + 1] = to_byte(rgb[3 * index + 1]);
                byte_img[3 * index + 2] = to_byte(rgb[3 * index + 2]);
            }
        }
    }
    if Path::new(name)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("qoi"))
    {
        let encoded = qoi::encode_to_vec(&byte_img, resolution.x as u32, resolution.y as u32)
            .map_err(|e| PbrtError::error(&e.to_string()))?;
        std::fs::write(name, encoded).map_err(|e| PbrtError::error(&e.to_string()))?;
        return Ok(());
    }
    let img = RgbImage::from_vec(resolution.x as u32, resolution.y as u32, byte_img).unwrap();
    match img.save(name) {
        Ok(()) => {
            return Ok(());
        }
        Err(e) => {
            return Err(PbrtError::from(e));
        }
    }
}

pub fn write_image(
    name: &str,
    rgb: &[Float],
    output_bounds: &Bounds2i,
    total_resolution: &Point2i,
) -> Result<(), PbrtError> {
    if let Some(ext) = Path::new(name).extension() {
        if let Some(s) = ext.to_str() {
            match s {
                "exr" => {
                    return write_image_exr(name, rgb, output_bounds, total_resolution);
                }
                _ => return write_image_bytes(name, rgb, output_bounds, total_resolution),
            }
        }
    }
    return Err(PbrtError::error("write_image: error"));
}

pub fn write_cache_image(
    name: &str,
    values: &[f32],
    resolution: &Point2i,
) -> Result<(), PbrtError> {
    let channels = values.len() / (resolution.x * resolution.y) as usize;
    let float_img = if channels == 3 {
        values.to_vec()
    } else {
        let mut v = vec![0.0; values.len() * 3];
        for i in 0..values.len() {
            v[3 * i + 0] = values[i];
            v[3 * i + 1] = values[i];
            v[3 * i + 2] = values[i];
        }
        v
    };
    let img = Rgb32FImage::from_vec(resolution.x as u32, resolution.y as u32, float_img).unwrap();
    match img.save(name) {
        Ok(()) => {
            return Ok(());
        }
        Err(e) => {
            return Err(PbrtError::from(e));
        }
    }
}
