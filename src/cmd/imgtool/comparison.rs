use super::{assert_compatible, load_image, output_exr_region, parse_crop, ImgToolError};
use pbrt_r4::util::base::Float;
use pbrt_r4::util::geometry::Vector2;
use std::ffi::OsString;
use std::fs;
use std::path::Path;

pub fn error(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut base = None;
    let mut reference = None;
    let mut crop = None;
    let mut metric = "MSE".to_string();
    let mut output = None;
    let mut index = 0;

    while index < args.len() {
        match args[index].to_str() {
            Some("--reference") | Some("-reference") => {
                index += 1;
                reference = Some(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| ImgToolError::with_help("error: --reference missing"))?
                        .to_string(),
                );
            }
            Some("--crop") | Some("-crop") => {
                index += 1;
                crop = Some(parse_crop(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| ImgToolError::with_help("error: --crop missing"))?,
                )?);
            }
            Some("--metric") | Some("-metric") => {
                index += 1;
                metric = args
                    .get(index)
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| ImgToolError::with_help("error: --metric missing"))?
                    .to_string();
            }
            Some("--errorfile") | Some("-errorfile") => {
                index += 1;
                output = Some(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| ImgToolError::with_help("error: --errorfile missing"))?
                        .to_string(),
                );
            }
            Some(value) if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "error: unknown option {value}"
                )))
            }
            Some(value) if base.is_none() => base = Some(value.to_string()),
            _ => return Err(ImgToolError::new("error: multiple base filenames provided")),
        }
        index += 1;
    }

    let base = base.ok_or_else(|| ImgToolError::with_help("error: base filename missing"))?;
    let reference =
        reference.ok_or_else(|| ImgToolError::with_help("error: --reference missing"))?;
    if !matches!(metric.as_str(), "MSE" | "MAE" | "MRSE") {
        return Err(ImgToolError::new("error: metric must be MSE, MAE, or MRSE"));
    }

    let base_path = Path::new(&base);
    let directory = base_path.parent().unwrap_or_else(|| Path::new("."));
    let prefix = base_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ImgToolError::new("error: base filename is not valid UTF-8"))?;
    let mut filenames: Vec<_> = fs::read_dir(directory)
        .map_err(|error| ImgToolError::new(error.to_string()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix))
        })
        .collect();
    filenames.sort();
    if filenames.is_empty() {
        return Err(ImgToolError::new("error: no matching files"));
    }

    let reference_image = load_image(Path::new(&reference))?;
    let [x0, x1, y0, y1] = crop.unwrap_or([
        0,
        reference_image.raw.resolution.x,
        0,
        reference_image.raw.resolution.y,
    ]);
    if x0 < 0
        || y0 < 0
        || x0 >= x1
        || y0 >= y1
        || x1 > reference_image.raw.resolution.x
        || y1 > reference_image.raw.resolution.y
    {
        return Err(ImgToolError::new("error: crop is outside reference bounds"));
    }
    let resolution = Vector2::new(x1 - x0, y1 - y0);
    let mut sum = 0.0;
    let mut error_image = vec![0.0; (resolution.x * resolution.y) as usize];

    for filename in &filenames {
        let image = load_image(filename)?;
        assert_compatible(&reference_image, &image)?;
        for y in y0..y1 {
            for x in x0..x1 {
                let pixel = (y * image.raw.resolution.x + x) as usize;
                let mut pixel_error = 0.0;
                for channel in 0..image.raw.channels {
                    let delta = image.raw.channel(pixel, channel)
                        - reference_image.raw.channel(pixel, channel);
                    let value = match metric.as_str() {
                        "MAE" => delta.abs(),
                        "MRSE" => {
                            delta * delta
                                / (reference_image.raw.channel(pixel, channel) + 0.01).powi(2)
                        }
                        _ => delta * delta,
                    };
                    if value.is_finite() {
                        pixel_error += value;
                    }
                }
                let offset = ((y - y0) * resolution.x + x - x0) as usize;
                error_image[offset] += pixel_error / image.raw.channels as Float;
                sum += pixel_error / image.raw.channels as Float;
            }
        }
    }

    let pixel_count = (resolution.x * resolution.y) as Float;
    let image_count = filenames.len() as Float;
    for value in &mut error_image {
        *value /= image_count;
    }
    println!(
        "{} estimate = {:.9}",
        metric,
        sum / (image_count * pixel_count)
    );

    if let Some(output) = output {
        output_exr_region(
            &output,
            &reference_image.metadata,
            resolution,
            &["Error".to_string()],
            error_image,
        )?;
    }
    Ok(())
}
