use super::{load_image, ImgToolError};
use pbrt_r4::util::base::Float;
use std::ffi::OsString;
use std::path::Path;

pub fn makeemitters(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut input = None;
    let mut downsample = 1_i32;
    let mut index = 0;
    while index < args.len() {
        match args[index].to_str() {
            Some("--downsample") | Some("-downsample") => {
                index += 1;
                downsample = args
                    .get(index)
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| ImgToolError::with_help("makeemitters: --downsample missing"))?
                    .parse()
                    .map_err(|_| ImgToolError::new("makeemitters: invalid --downsample"))?;
            }
            Some(value) if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "makeemitters: unknown option {value}"
                )))
            }
            Some(value) if input.is_none() => input = Some(value.to_string()),
            _ => return Err(ImgToolError::new("makeemitters: multiple input filenames")),
        }
        index += 1;
    }
    let input = input.ok_or_else(|| ImgToolError::with_help("makeemitters: input missing"))?;
    if downsample <= 0 {
        return Err(ImgToolError::new(
            "makeemitters: downsample must be positive",
        ));
    }
    let image = load_image(Path::new(&input))?;
    let channels = ["R", "G", "B"]
        .map(|name| {
            image
                .channel_names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| ImgToolError::new("makeemitters: image needs R, G, and B channels"))
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let resolution = image.raw.resolution;
    let aspect = resolution.x as Float / resolution.y as Float;
    println!("AttributeBegin");
    println!("Material \"matte\" \"rgb Kd\" [0 0 0]");
    for y in (0..resolution.y).step_by(downsample as usize) {
        for x in (0..resolution.x).step_by(downsample as usize) {
            let mut sum = [0.0; 3];
            for dy in 0..downsample {
                for dx in 0..downsample {
                    let px = x + dx;
                    let py = y + dy;
                    if px >= resolution.x || py >= resolution.y {
                        continue;
                    }
                    let pixel = (py * resolution.x + px) as usize;
                    for component in 0..3 {
                        sum[component] += image.raw.channel(pixel, channels[component]);
                    }
                }
            }
            let divisor = (downsample * downsample) as Float;
            for value in &mut sum {
                *value /= divisor;
            }
            println!(
                "AreaLightSource \"diffuse\" \"rgb L\" [ {:.6} {:.6} {:.6} ]",
                sum[0], sum[1], sum[2]
            );
            let x0 = aspect * (1.0 - x as Float / resolution.x as Float) - aspect / 2.0;
            let x1 = aspect
                * (1.0 - (x + downsample).min(resolution.x) as Float / resolution.x as Float)
                - aspect / 2.0;
            let y0 = 1.0 - y as Float / resolution.y as Float;
            let y1 = 1.0 - (y + downsample).min(resolution.y) as Float / resolution.y as Float;
            println!(
                "Shape \"bilinear\" \"point3 P\" [ {:.6} {:.6} 0 {:.6} {:.6} 0 {:.6} {:.6} 0 {:.6} {:.6} 0 ]",
                x0, y0, x1, y0, x0, y1, x1, y1
            );
        }
    }
    println!("AttributeEnd");
    Ok(())
}
