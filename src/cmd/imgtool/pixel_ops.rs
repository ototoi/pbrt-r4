use super::{load_image, output_exr_region, ImgToolError};
use pbrt_r4::util::base::Float;
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
