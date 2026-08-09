use crate::util::base::*;
use crate::util::error::*;
use crate::util::spectrum::*;

use nom::character::complete::{alphanumeric1, multispace0};
use nom::error::*;
use nom::number::complete::*;
use nom::sequence;
use nom::IResult;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use log::info;

fn read_word(s: &[u8]) -> IResult<&[u8], &[u8]> {
    let (s, word) = sequence::delimited(multispace0, alphanumeric1, multispace0)(s)?;
    return Ok((s, word));
}

fn read_value(s: &[u8]) -> IResult<&[u8], &[u8]> {
    let (s, word) = sequence::delimited(multispace0, recognize_float, multispace0)(s)?;
    return Ok((s, word));
}

fn read_image_pfm_core(input: &[u8]) -> IResult<&[u8], (Vec<RGBSpectrum>, Point2i)> {
    // read either "Pf" or "PF"
    let (input, cc) = read_word(input)?;
    let cc =
        std::str::from_utf8(cc).map_err(|_| nom::Err::Error(Error::new(input, ErrorKind::Fail)))?;

    let n_channels;
    if cc == "Pf" {
        n_channels = 1;
    } else if cc == "PF" {
        n_channels = 3;
    } else {
        return Err(nom::Err::Error(Error::new(input, ErrorKind::Fail)));
    }

    // read the rest of the header
    // read width
    let (input, width) = read_value(input)?;
    let width = std::str::from_utf8(width)
        .map_err(|_| nom::Err::Error(Error::new(input, ErrorKind::Fail)))?
        .parse::<usize>()
        .map_err(|_| nom::Err::Error(Error::new(input, ErrorKind::Fail)))?;

    // read height
    let (input, height) = read_value(input)?;
    let height = std::str::from_utf8(height)
        .map_err(|_| nom::Err::Error(Error::new(input, ErrorKind::Fail)))?
        .parse::<usize>()
        .map_err(|_| nom::Err::Error(Error::new(input, ErrorKind::Fail)))?;

    // read scale
    let (input, scale) = read_value(input)?;
    let scale = std::str::from_utf8(scale)
        .map_err(|_| nom::Err::Error(Error::new(input, ErrorKind::Fail)))?
        .parse::<f32>()
        .map_err(|_| nom::Err::Error(Error::new(input, ErrorKind::Fail)))?;

    // apply endian conversian and scale if appropriate
    let file_little_endian = scale < 0.0;
    // read the data
    let n_floats = n_channels * width * height;
    let mut data = vec![0.0; n_floats];
    {
        let mut input = input;
        // Flip in Y, as P*M has the origin at the lower left.
        for y in 0..height {
            let yy = height - y - 1;
            for x in 0..width {
                let index = (yy * width + x) as usize;
                for c in 0..n_channels {
                    let (inp, f) = if file_little_endian {
                        le_f32(input)?
                    } else {
                        be_f32(input)?
                    };
                    data[n_channels * index + c] = f;
                    input = inp;
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

    // create RGBs...
    let mut rgb = Vec::with_capacity(width as usize * height as usize);
    if n_channels == 1 {
        for f in data {
            let f = f as Float;
            rgb.push(RGBSpectrum::rgb_from_rgb(&[f, f, f]));
        }
    } else {
        for i in 0..(width * height) as usize {
            let r = data[i * 3] as Float;
            let g = data[i * 3 + 1] as Float;
            let b = data[i * 3 + 2] as Float;
            rgb.push(RGBSpectrum::rgb_from_rgb(&[r, g, b]));
        }
    }
    assert!(rgb.len() == (width * height) as usize);
    return Ok((input, (rgb, Point2i::from((width as i32, height as i32)))));
}

pub fn read_image_pfm(name: &str) -> Result<(Vec<RGBSpectrum>, Point2i), PbrtError> {
    let path = Path::new(name);
    let mut reader = BufReader::new(File::open(path)?);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;

    match read_image_pfm_core(&bytes) {
        Ok((_, v)) => {
            let resolution = v.1;
            info!(
                "Read PFM image {} ({}x{}) resolution",
                name, resolution.x, resolution.y
            );
            return Ok(v);
        }
        Err(_) => {
            let msg = format!("Error reading PFM file \"{}\"", path.display());
            return Err(PbrtError::error(&msg));
        }
    }
}
