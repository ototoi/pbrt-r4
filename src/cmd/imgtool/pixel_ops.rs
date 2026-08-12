use super::{
    falsecolor_table::FALSE_COLOR_VALUES, load_image, output_exr_region, output_exr_region_format,
    ImgToolError,
};
use pbrt_r4::util::base::{inverse_gamma_correct, Float};
use pbrt_r4::util::geometry::{Bounds2i, Vector2};
use pbrt_r4::util::image::PixelFormat;
use pbrt_r4::util::imageio::write_image;
use std::ffi::OsString;
use std::path::Path;

pub fn scalenormalmap(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut input = None;
    let mut output = None;
    let mut scale = 1.0;
    let mut index = 0;

    while index < args.len() {
        match args[index].to_str() {
            Some("--scale") | Some("-scale") => {
                index += 1;
                scale = args
                    .get(index)
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| ImgToolError::with_help("scalenormalmap: --scale missing"))?
                    .parse()
                    .map_err(|_| ImgToolError::new("scalenormalmap: invalid --scale"))?;
            }
            Some("--outfile") | Some("-outfile") => {
                index += 1;
                output = Some(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| {
                            ImgToolError::with_help("scalenormalmap: --outfile missing")
                        })?
                        .to_string(),
                );
            }
            Some(value) if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "scalenormalmap: unknown option {value}"
                )))
            }
            Some(value) if input.is_none() => input = Some(value.to_string()),
            _ => {
                return Err(ImgToolError::new(
                    "scalenormalmap: multiple input filenames",
                ))
            }
        }
        index += 1;
    }

    let input = input.ok_or_else(|| ImgToolError::with_help("scalenormalmap: input missing"))?;
    let output =
        output.ok_or_else(|| ImgToolError::with_help("scalenormalmap: --outfile missing"))?;
    let image = load_image(Path::new(&input))?;
    let channels = ["R", "G", "B"]
        .map(|name| {
            image
                .channel_names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| {
                    ImgToolError::new("scalenormalmap: image needs R, G, and B channels")
                })
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;

    let pixels = (image.raw.resolution.x * image.raw.resolution.y) as usize;
    let mut data = Vec::with_capacity(pixels * 3);
    for pixel in 0..pixels {
        let mut normal = [0.0; 3];
        for (component, &channel) in channels.iter().enumerate() {
            normal[component] = 2.0 * image.raw.channel(pixel, channel) - 1.0;
        }
        normal[0] *= scale;
        normal[1] *= scale;
        normal[2] = (1.0 - normal[0] * normal[0] - normal[1] * normal[1])
            .max(0.0)
            .sqrt();
        data.extend(normal.map(|value: Float| (value + 1.0) / 2.0));
    }

    output_exr_region(
        &output,
        &image.metadata,
        image.raw.resolution,
        &["R".to_string(), "G".to_string(), "B".to_string()],
        data,
    )
}

pub fn falsecolor(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut input = None;
    let mut output = None;
    let mut max_value = None;
    let mut plusminus = false;
    let mut ramp = false;
    let mut index = 0;

    while index < args.len() {
        match args[index].to_str() {
            Some("--maxvalue") | Some("-maxvalue") => {
                index += 1;
                max_value = Some(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| ImgToolError::with_help("falsecolor: --maxvalue missing"))?
                        .parse()
                        .map_err(|_| ImgToolError::new("falsecolor: invalid --maxvalue"))?,
                );
            }
            Some("--outfile") | Some("-outfile") => {
                index += 1;
                output = Some(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| ImgToolError::with_help("falsecolor: --outfile missing"))?
                        .to_string(),
                );
            }
            Some("--plusminus") | Some("-plusminus") => plusminus = true,
            Some("--ramp") | Some("-ramp") => ramp = true,
            Some(value) if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "falsecolor: unknown option {value}"
                )))
            }
            Some(value) if input.is_none() => input = Some(value.to_string()),
            _ => return Err(ImgToolError::new("falsecolor: multiple input filenames")),
        }
        index += 1;
    }

    let output = output.ok_or_else(|| ImgToolError::with_help("falsecolor: --outfile missing"))?;
    let (resolution, metadata, values) = if ramp {
        let resolution = Vector2::new(10, 300);
        let values: Vec<Float> = (0..resolution.y)
            .flat_map(|y| {
                std::iter::repeat_n(
                    (resolution.y - 1 - y) as Float / (resolution.y - 1) as Float,
                    resolution.x as usize,
                )
            })
            .collect();
        (resolution, Default::default(), values)
    } else {
        let input = input.ok_or_else(|| ImgToolError::with_help("falsecolor: input missing"))?;
        let image = load_image(Path::new(&input))?;
        let resolution = image.raw.resolution;
        let channels = image.raw.channels;
        let mut values = Vec::with_capacity((resolution.x * resolution.y) as usize);
        for y in 0..resolution.y {
            for x in 0..resolution.x {
                let pixel = (y * resolution.x + x) as usize;
                values.push(
                    (0..channels)
                        .map(|channel| image.raw.channel(pixel, channel))
                        .sum::<Float>()
                        / channels as Float,
                );
            }
        }
        (resolution, Default::default(), values)
    };

    let max_value = max_value.unwrap_or_else(|| {
        values
            .iter()
            .map(|value| value.abs())
            .fold(Float::NEG_INFINITY, Float::max)
    });
    if !max_value.is_finite() || max_value == 0.0 {
        return Err(ImgToolError::new(
            "falsecolor: maxvalue must be finite and non-zero",
        ));
    }

    let mut output_data = Vec::with_capacity(values.len() * 3);
    for value in values {
        let mut relative = value / max_value;
        let rgb = if plusminus {
            if relative > 0.0 {
                [0.0, relative, 0.0]
            } else {
                relative = relative.abs();
                [relative, 0.0, 0.0]
            }
        } else {
            relative = relative.clamp(0.0, 1.0);
            let index = ((relative * FALSE_COLOR_VALUES.len() as Float) as usize)
                .min(FALSE_COLOR_VALUES.len() - 1);
            FALSE_COLOR_VALUES[index]
        };
        output_data.extend(rgb.map(inverse_gamma_correct));
    }

    if output
        .rsplit_once('.')
        .is_some_and(|(_, extension)| extension.eq_ignore_ascii_case("exr"))
    {
        output_exr_region_format(
            &output,
            &metadata,
            resolution,
            &["R".to_string(), "G".to_string(), "B".to_string()],
            output_data,
            PixelFormat::Half,
        )
    } else {
        let bounds = Bounds2i::from(((0, 0), (resolution.x, resolution.y)));
        write_image(&output, &output_data, &bounds, &resolution).map_err(ImgToolError::from)
    }
}
