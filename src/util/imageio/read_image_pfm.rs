use crate::util::base::*;
use crate::util::error::*;
use crate::util::spectrum::*;

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use log::info;

// v4 util/image.cpp readWord()
fn is_header_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn skip_header_whitespace(input: &[u8], pos: &mut usize) {
    while *pos < input.len() && is_header_whitespace(input[*pos]) {
        *pos += 1;
    }
}

fn read_header_word<'a>(input: &'a [u8], pos: &mut usize) -> Result<&'a [u8], PbrtError> {
    skip_header_whitespace(input, pos);
    let start = *pos;
    while *pos < input.len() && !is_header_whitespace(input[*pos]) {
        *pos += 1;
    }
    let end = *pos;
    if start == end {
        return Err(PbrtError::error("Premature end of PFM header."));
    }
    if *pos < input.len() {
        if input[*pos] == b'\r' && input.get(*pos + 1) == Some(&b'\n') {
            *pos += 2;
        } else {
            *pos += 1;
        }
    }
    Ok(&input[start..end])
}

fn parse_positive_dimension(word: &[u8], name: &str) -> Result<usize, PbrtError> {
    let text =
        std::str::from_utf8(word).map_err(|_| PbrtError::error("Invalid PFM header text."))?;
    let value = text
        .parse::<u64>()
        .map_err(|_| PbrtError::error(&format!("Unable to decode PFM {} \"{}\".", name, text)))?;
    if value == 0 || value > i32::MAX as u64 {
        return Err(PbrtError::error(&format!(
            "PFM {} \"{}\" is outside the valid positive range.",
            name, text
        )));
    }
    Ok(value as usize)
}

fn parse_scale(word: &[u8]) -> Result<f32, PbrtError> {
    let text =
        std::str::from_utf8(word).map_err(|_| PbrtError::error("Invalid PFM header text."))?;
    text.parse::<f32>()
        .map_err(|_| PbrtError::error(&format!("Unable to decode PFM scale \"{}\".", text)))
}

fn read_image_pfm_core(input: &[u8]) -> Result<(Vec<RGBSpectrum>, Point2i), PbrtError> {
    let mut pos = 0;
    let cc = read_header_word(input, &mut pos)?;
    let n_channels = match cc {
        b"Pf" => 1usize,
        b"PF" => 3usize,
        _ => {
            return Err(PbrtError::error(&format!(
                "Unable to decode PFM file type \"{}\".",
                String::from_utf8_lossy(cc)
            )));
        }
    };

    let width = parse_positive_dimension(read_header_word(input, &mut pos)?, "width")?;
    let height = parse_positive_dimension(read_header_word(input, &mut pos)?, "height")?;
    let scale = parse_scale(read_header_word(input, &mut pos)?)?;

    let pixels = (width as u64)
        .checked_mul(height as u64)
        .ok_or_else(|| PbrtError::error("PFM dimensions overflowed."))?;
    let n_floats = pixels
        .checked_mul(n_channels as u64)
        .ok_or_else(|| PbrtError::error("PFM channel count overflowed."))?;
    let payload_bytes = n_floats
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .ok_or_else(|| PbrtError::error("PFM payload size overflowed."))?;
    let payload_bytes = usize::try_from(payload_bytes)
        .map_err(|_| PbrtError::error("PFM payload size is too large."))?;
    if input
        .len()
        .checked_sub(pos)
        .map_or(true, |available| available < payload_bytes)
    {
        return Err(PbrtError::error("Premature end of file in PFM file."));
    }

    let file_little_endian = scale < 0.0;
    let mut data = vec![0.0f32; n_floats as usize];
    {
        let mut offset = pos;
        // Flip in Y, as P*M has the origin at the lower left.
        for y in 0..height {
            let yy = height - y - 1;
            for x in 0..width {
                let index = (yy * width + x) as usize;
                for c in 0..n_channels {
                    let bytes: [u8; 4] = input[offset..offset + 4].try_into().unwrap();
                    offset += 4;
                    data[n_channels * index + c] = if file_little_endian {
                        f32::from_le_bytes(bytes)
                    } else {
                        f32::from_be_bytes(bytes)
                    };
                }
            }
        }
    }

    let scale = scale.abs();
    if scale != 1.0 {
        for f in &mut data {
            *f *= scale;
        }
    }

    let mut rgb = Vec::with_capacity(width * height);
    if n_channels == 1 {
        for f in data {
            rgb.push(RGBSpectrum::rgb_from_rgb(&[f as Float; 3]));
        }
    } else {
        for chunk in data.chunks_exact(3) {
            rgb.push(RGBSpectrum::rgb_from_rgb(&[
                chunk[0] as Float,
                chunk[1] as Float,
                chunk[2] as Float,
            ]));
        }
    }
    Ok((rgb, Point2i::from((width as i32, height as i32))))
}

pub fn read_image_pfm(name: &str) -> Result<(Vec<RGBSpectrum>, Point2i), PbrtError> {
    let path = Path::new(name);
    let mut reader = BufReader::new(File::open(path)?);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;

    match read_image_pfm_core(&bytes) {
        Ok((pixels, resolution)) => {
            info!(
                "Read PFM image {} ({}x{}) resolution",
                name, resolution.x, resolution.y
            );
            Ok((pixels, resolution))
        }
        Err(_) => Err(PbrtError::error(&format!(
            "Error reading PFM file \"{}\"",
            path.display()
        ))),
    }
}
