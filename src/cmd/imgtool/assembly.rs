use super::{assert_compatible, load_image, output_exr_region, ImgToolError};
use pbrt_r4::util::base::Float;
use pbrt_r4::util::geometry::Vector2;
use std::ffi::OsString;
use std::path::Path;

fn parse_common_files(
    args: &[OsString],
    command: &str,
) -> Result<(String, Vec<String>), ImgToolError> {
    let mut output = None;
    let mut inputs = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].to_str() {
            Some("--outfile") | Some("-outfile") => {
                index += 1;
                output = Some(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| {
                            ImgToolError::with_help(format!("{command}: --outfile missing"))
                        })?
                        .to_string(),
                );
            }
            Some(value) if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "{command}: unknown option {value}"
                )))
            }
            Some(value) => inputs.push(value.to_string()),
            None => return Err(ImgToolError::new(format!("{command}: invalid filename"))),
        }
        index += 1;
    }
    Ok((
        output.ok_or_else(|| ImgToolError::with_help(format!("{command}: --outfile missing")))?,
        inputs,
    ))
}

pub fn assemble(args: &[OsString]) -> Result<(), ImgToolError> {
    let (output, inputs) = parse_common_files(args, "assemble")?;
    if inputs.is_empty() {
        return Err(ImgToolError::with_help("assemble: no input files"));
    }
    if !Path::new(&output)
        .extension()
        .is_some_and(|ext| ext == "exr")
    {
        return Err(ImgToolError::new("assemble: output must be an EXR file"));
    }

    let mut full_resolution = None;
    let mut channel_names = None;
    let mut metadata = None;
    let mut color_space_name = None;
    let mut data = Vec::new();
    let mut seen = Vec::new();
    let mut seen_multiple = 0;

    for filename in inputs {
        if !Path::new(&filename)
            .extension()
            .is_some_and(|ext| ext == "exr")
        {
            return Err(ImgToolError::new(
                "assemble: only EXR inputs have pixel bounds metadata",
            ));
        }
        let image = load_image(Path::new(&filename))?;
        let Some(full) = image.metadata.full_resolution else {
            eprintln!("assemble: {filename} has no full resolution; skipping");
            continue;
        };
        let Some(bounds) = image.metadata.pixel_bounds else {
            eprintln!("assemble: {filename} has no pixel bounds; skipping");
            continue;
        };
        if bounds.diagonal() != image.raw.resolution {
            eprintln!("assemble: {filename} pixel bounds do not match image resolution; skipping");
            continue;
        }

        if full_resolution.is_none() {
            full_resolution = Some(full);
            channel_names = Some(image.channel_names.clone());
            data = vec![0.0; (full.x * full.y) as usize * image.raw.channels];
            seen = vec![false; (full.x * full.y) as usize];
            metadata = Some(image.metadata.clone());
            color_space_name = image
                .metadata
                .color_space
                .map(|color_space| color_space.name);
        } else if full_resolution != Some(full) {
            eprintln!("assemble: {filename} has an incompatible full resolution; skipping");
            continue;
        } else if channel_names
            .as_ref()
            .is_some_and(|names| names.len() != image.raw.channels)
        {
            eprintln!("assemble: {filename} has an incompatible channel count; skipping");
            continue;
        } else if image
            .metadata
            .color_space
            .map(|color_space| color_space.name)
            != color_space_name
        {
            eprintln!("assemble: {filename} has an incompatible color space; skipping");
            continue;
        }

        let full = full_resolution.unwrap();
        if bounds.min.x < 0 || bounds.min.y < 0 || bounds.max.x > full.x || bounds.max.y > full.y {
            eprintln!("assemble: {filename} bounds are outside full resolution; skipping");
            continue;
        }
        for y in 0..image.raw.resolution.y {
            for x in 0..image.raw.resolution.x {
                let full_pixel = ((bounds.min.y + y) * full.x + bounds.min.x + x) as usize;
                if seen[full_pixel] {
                    seen_multiple += 1;
                }
                seen[full_pixel] = true;
                for channel in 0..image.raw.channels {
                    data[full_pixel * image.raw.channels + channel] = image
                        .raw
                        .channel((y * image.raw.resolution.x + x) as usize, channel);
                }
            }
        }
    }

    let Some(full) = full_resolution else {
        return Err(ImgToolError::new(
            "assemble: no valid input tiles were found",
        ));
    };
    let missing = seen.iter().filter(|value| !**value).count();
    if seen_multiple > 0 {
        eprintln!("assemble: {seen_multiple} pixels present in multiple inputs");
    }
    if missing > 0 {
        eprintln!("assemble: {missing} pixels not present in any input");
    }
    let mut output_metadata = metadata.unwrap_or_default();
    output_metadata.full_resolution = Some(full);
    output_metadata.pixel_bounds = Some(pbrt_r4::util::geometry::Bounds2i::from((
        (0, 0),
        (full.x, full.y),
    )));
    output_exr_region(
        &output,
        &output_metadata,
        full,
        &channel_names.unwrap(),
        data,
    )
}

pub fn splitn(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut output = None;
    let mut inputs = Vec::new();
    let mut crop_size = 96_i32;
    let mut crops = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].to_str() {
            Some("--outfile") | Some("-outfile") => {
                index += 1;
                output = Some(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| ImgToolError::with_help("splitn: --outfile missing"))?
                        .to_string(),
                );
            }
            Some("--cropsize") | Some("-cropsize") => {
                index += 1;
                crop_size = args
                    .get(index)
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| ImgToolError::with_help("splitn: --cropsize missing"))?
                    .parse()
                    .map_err(|_| ImgToolError::new("splitn: invalid --cropsize"))?;
            }
            Some("--crop") | Some("-crop") => {
                index += 1;
                let value = args
                    .get(index)
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| ImgToolError::with_help("splitn: --crop missing"))?;
                let position: Vec<i32> = value
                    .split(',')
                    .map(|part| {
                        part.parse()
                            .map_err(|_| ImgToolError::new("splitn: invalid --crop"))
                    })
                    .collect::<Result<_, _>>()?;
                if position.len() != 2 {
                    return Err(ImgToolError::new("splitn: --crop requires x,y"));
                }
                crops.push((position[0], position[1]));
            }
            Some(value) if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "splitn: unknown option {value}"
                )))
            }
            Some(value) => inputs.push(value.to_string()),
            None => return Err(ImgToolError::new("splitn: invalid filename")),
        }
        index += 1;
    }
    let output = output.ok_or_else(|| ImgToolError::with_help("splitn: --outfile missing"))?;
    if inputs.is_empty() {
        return Err(ImgToolError::with_help("splitn: no input files"));
    }
    if crop_size < 1 {
        return Err(ImgToolError::new("splitn: --cropsize must be positive"));
    }
    if crops.len() > 3 {
        return Err(ImgToolError::new(
            "splitn: at most three crops are supported",
        ));
    }
    let mut images = Vec::new();
    for filename in &inputs {
        images.push(load_image(Path::new(filename))?);
    }
    let first = &images[0];
    for image in images.iter().skip(1) {
        assert_compatible(first, image)?;
    }
    let resolution = first.raw.resolution;
    let channels = first.raw.channels;
    let mut data = vec![0.0; (resolution.x * resolution.y) as usize * channels];
    let separator = 6;
    let slope = 15.0;
    for y in 0..resolution.y {
        let mut x_start = 0;
        for (index, image) in images.iter().enumerate() {
            let x_end = if index + 1 == images.len() {
                resolution.x
            } else {
                ((index + 1) as Float / images.len() as Float * resolution.x as Float
                    + (2.0 * y as Float / resolution.y as Float - 1.0) * resolution.x as Float
                        / -slope) as i32
            };
            let copy_end = (x_end - separator / 2).clamp(x_start, resolution.x);
            for x in x_start..copy_end {
                let out_pixel = (y * resolution.x + x) as usize;
                let in_pixel = (y * resolution.x + x) as usize;
                for channel in 0..channels {
                    data[out_pixel * channels + channel] = image.raw.channel(in_pixel, channel);
                }
            }
            if index + 1 != images.len() {
                let black_end = (x_end + separator / 2).clamp(copy_end, resolution.x);
                x_start = black_end;
            } else {
                x_start = resolution.x;
            }
        }
    }
    output_exr_region(
        &output,
        &first.metadata,
        resolution,
        &first.channel_names,
        data,
    )?;

    if !crops.is_empty() {
        let rgb_channels = ["R", "G", "B"]
            .map(|name| {
                first
                    .channel_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| ImgToolError::new("splitn: crops require R, G, and B channels"))
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let border = 5_i32;
        let separator = 6_i32;
        let crop_resolution = Vector2::new(
            (crop_size + 2 * border) * images.len() as i32 + separator * (images.len() as i32 - 1),
            (crop_size + 2 * border) * crops.len() as i32 + separator * (crops.len() as i32 - 1),
        );
        let mut crop_data = vec![1.0; (crop_resolution.x * crop_resolution.y * 3) as usize];
        let edges = [[0.8, 0.15, 0.15], [0.15, 0.8, 0.15], [0.15, 0.15, 0.8]];
        let set_crop_pixel = |data: &mut [Float], x: i32, y: i32, value: [Float; 3]| {
            if x >= 0 && y >= 0 && x < crop_resolution.x && y < crop_resolution.y {
                let offset = ((y * crop_resolution.x + x) * 3) as usize;
                data[offset..offset + 3].copy_from_slice(&value);
            }
        };
        for (crop_index, &(crop_x, crop_y)) in crops.iter().enumerate() {
            let x_origin = |image_index: usize| {
                image_index as i32 * (crop_size + 2 * border + separator) + border
            };
            let y_origin = crop_index as i32 * (crop_size + 2 * border + separator) + border;
            for image_index in 0..images.len() {
                let origin_x = x_origin(image_index);
                for y in 0..crop_size {
                    for x in 0..crop_size {
                        let source_x = crop_x + x;
                        let source_y = crop_y + y;
                        if source_x < 0
                            || source_y < 0
                            || source_x >= resolution.x
                            || source_y >= resolution.y
                        {
                            continue;
                        }
                        let pixel = (source_y * resolution.x + source_x) as usize;
                        let value = [
                            images[image_index].raw.channel(pixel, rgb_channels[0]),
                            images[image_index].raw.channel(pixel, rgb_channels[1]),
                            images[image_index].raw.channel(pixel, rgb_channels[2]),
                        ];
                        set_crop_pixel(&mut crop_data, origin_x + x, y_origin + y, value);
                    }
                }
                for y in (y_origin - border)..(y_origin + crop_size + border) {
                    for x in (origin_x - border)..(origin_x + crop_size + border) {
                        if x < origin_x
                            || x >= origin_x + crop_size
                            || y < y_origin
                            || y >= y_origin + crop_size
                        {
                            set_crop_pixel(&mut crop_data, x, y, edges[crop_index]);
                        }
                    }
                }
            }
        }
        let crop_output = Path::new(&output)
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!(
                "crops-{}",
                Path::new(&output)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("output.exr")
            ));
        let crop_names = ["R".to_string(), "G".to_string(), "B".to_string()];
        output_exr_region(
            crop_output
                .to_str()
                .ok_or_else(|| ImgToolError::new("splitn: crop output path is not valid UTF-8"))?,
            &first.metadata,
            crop_resolution,
            &crop_names,
            crop_data,
        )?;
    }
    Ok(())
}
