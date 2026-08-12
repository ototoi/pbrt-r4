use pbrt_r4::ext::skymodel::HosekSkyModel;
use pbrt_r4::util::base::{Float, Point2f, Point2i};
use pbrt_r4::util::error::PbrtError;
use pbrt_r4::util::geometry::{equal_area_square_to_sphere, spherical_theta, Bounds2i, Vector2};
use pbrt_r4::util::image::{Image, ImageMetadata, PixelFormat};
use pbrt_r4::util::imageio::{
    read_raw_image_exr_with_channels_and_metadata, read_raw_image_gamma_correct,
};
use pbrt_r4::util::imageio::{write_image_bytes, RawImage, RawImageData};
use pbrt_r4::util::math::safe_acos;
use pbrt_r4::util::spectrum::rgb_to_spectrum::{ACES2065_1, SRGB};
use pbrt_r4::util::spectrum::{PiecewiseLinearSpectrum, RGBSpectrum};
use std::ffi::OsString;
use std::fs;
use std::path::Path;

mod assembly;
mod color_ops;
mod comparison;
mod falsecolor_table;
mod pixel_ops;

#[derive(Debug)]
struct ImgToolError {
    message: String,
    show_help: bool,
}

impl ImgToolError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            show_help: false,
        }
    }

    fn with_help(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            show_help: true,
        }
    }
}

impl std::fmt::Display for ImgToolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ImgToolError {}

impl From<PbrtError> for ImgToolError {
    fn from(error: PbrtError) -> Self {
        Self::new(error.to_string())
    }
}

fn help_text() -> &'static str {
    "usage: imgtool <command> [options]\n\n\
where <command> is:\n\n\
cat: Print the pixel values of the specified image.\n\n\
info: Print image resolution, channels, and statistics.\n\n\
convert: Convert an image and apply basic pixel operations.\n\n\
average: Average images matching a filename prefix.\n\n\
diff: Compute per-channel image differences.\n\n\
error: Compute the average error of a set of images.\n\n\
assemble: Assemble EXR image tiles into a full image.\n\n\
splitn: Compose multiple images with diagonal separators.\n\n\
scalenormalmap: Scale the x and y components of a normal map.\n\n\
falsecolor: Convert scalar values to a false-color image.\n\n\
bloom: Add a Gaussian bloom around pixels above a threshold.\n\n\
whitebalance: Apply Bradford white balance to an RGB image.\n\n\
makeequiarea: Convert a lat-long image to an equal-area environment map.\n\n\
makesky: Generate an equi-area environment map using Hosek-Wilkie.\n\n\
help: Print command help.\n\n\
\"imgtool help <command>\" provides detailed information about <command>.\n"
}

fn command_help(command: &str) -> Result<&'static str, ImgToolError> {
    match command {
        "cat" => Ok("usage: imgtool cat [options] <filename>\n\n\
options:\n\
    --csv       Output pixel values as CSV.\n\
    --list      Output a single-channel image as a list.\n\
    --sort      Sort pixels by their channel average.\n"),
        "info" => Ok("usage: imgtool info <filename...>\n"),
        "average" => Ok("usage: imgtool average --outfile <name> <filename base>\n"),
        "diff" => Ok("usage: imgtool diff --reference <name> [--metric MSE|MAE|MRSE] <filename>\n"),
        "error" => Ok("usage: imgtool error [options] <filename base>\n\n\\
options:\n\\
    --reference <name>  Reference image filename.\n\\
    --crop <x0,x1,y0,y1>  Crop images before comparison.\n\\
    --metric <name>     Error metric: MAE, MSE, or MRSE.\n\\
    --errorfile <name>  Output average error image.\n"),
        "assemble" => Ok("usage: imgtool assemble --outfile <name> <filenames...>\n"),
        "splitn" => Ok("usage: imgtool splitn --outfile <name> <filenames...>\n"),
        "scalenormalmap" => Ok("usage: imgtool scalenormalmap [options] <filename>\n\n\\
options:\n\\
    --scale <value>     Scale factor for x and y. Default: 1.\n\\
    --outfile <name>    Output image filename.\n"),
        "falsecolor" => Ok("usage: imgtool falsecolor [options] [filename]\n\n\\
options:\n\\
    --maxvalue <value>   Maximum value for normalization.\n\\
    --plusminus          Show positive values in green and negative in red.\n\\
    --ramp               Generate the v4 10x300 color ramp.\n\\
    --outfile <name>     Output image filename.\n"),
        "bloom" => Ok("usage: imgtool bloom [options] <filename>\n\n\\
options:\n\\
    --level <value>      Threshold. Default: Infinity.\n\\
    --width <value>      Gaussian width. Default: 15.\n\\
    --iterations <n>     Number of blur iterations. Default: 5.\n\\
    --scale <value>      Bloom scale. Default: 0.3.\n\\
    --outfile <name>     Output image filename.\n"),
        "whitebalance" => Ok("usage: imgtool whitebalance [options] <filename>\n\n\\
options:\n\\
    --illuminant <name>  Named source illuminant.\n\\
    --temperature <K>    D-series source temperature.\n\\
    --primaries <x> <y>  Source white chromaticity.\n\\
    --outfile <name>     Output EXR filename.\n"),
        "makeequiarea" => Ok("usage: imgtool makeequiarea [options] <filename>\n\n\\
options:\n\\
    --resolution <n>     Square output resolution. Default: input width.\n\\
    --outfile <name>     Output image filename.\n"),
        "makesky" => Ok("usage: imgtool makesky [options]\n\n\
options:\n\
    --outfile <name>      Output EXR filename.\n\
    --albedo <value>      Ground albedo, 0 through 1. Default: 0.5.\n\
    --turbidity <value>   Atmospheric turbidity, 1.7 through 10. Default: 3.\n\
    --elevation <value>   Solar elevation in degrees, 0 through 90. Default: 10.\n\
    --resolution <value>  Width and height. Default: 2048.\n"),
        "convert" => Ok("usage: imgtool convert [options] <filename>\n\n\
options:\n\
    --outfile <name>          Output image filename.\n\
    --channels <names>        Comma-separated channels to keep.\n\
    --crop <x0,x1,y0,y1>      Crop image to the given bounds.\n\
    --flipy                    Flip the image vertically.\n\
    --scale <value>           Scale pixel values.\n\
    --gamma <value>           Apply a signed power curve.\n\
    --fp16                    Write Float channels as Half in EXR.\n"),
        "help" => Ok("usage: imgtool help [command...]\n"),
        _ => Err(ImgToolError::with_help(format!(
            "imgtool help: command \"{command}\" not known."
        ))),
    }
}

struct LoadedImage {
    raw: RawImage,
    channel_names: Vec<String>,
    metadata: ImageMetadata,
}

fn load_image(path: &Path) -> Result<LoadedImage, ImgToolError> {
    let path_string = path
        .to_str()
        .ok_or_else(|| ImgToolError::new("image path is not valid UTF-8"))?;
    if path.extension().is_some_and(|extension| extension == "exr") {
        let (raw, channel_names, mut metadata) =
            read_raw_image_exr_with_channels_and_metadata(path)?;
        if metadata.color_space.is_none() {
            metadata.color_space = Some(&SRGB);
        }
        Ok(LoadedImage {
            raw,
            channel_names,
            metadata,
        })
    } else {
        let raw = read_raw_image_gamma_correct(path_string, false)?;
        let channel_names = raw.channel_names();
        let metadata = ImageMetadata {
            color_space: Some(&SRGB),
            ..Default::default()
        };
        Ok(LoadedImage {
            raw,
            channel_names,
            metadata,
        })
    }
}

fn channel_value(image: &RawImage, pixel: usize, channel: usize) -> Float {
    image.channel(pixel, channel)
}

fn format_value(value: Float) -> String {
    format!("{value:.6}")
}

fn cat(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut csv = false;
    let mut list = false;
    let mut sort = false;
    let mut filename = None;

    for argument in args {
        match argument.to_str() {
            Some("--csv") | Some("-csv") => csv = !csv,
            Some("--list") | Some("-list") => list = !list,
            Some("--sort") | Some("-sort") => sort = !sort,
            Some(value) if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "cat: unknown option \"{value}\""
                )))
            }
            _ if filename.is_none() => filename = Some(argument),
            _ => return Err(ImgToolError::new("cat: multiple input filenames provided")),
        }
    }

    if sort && csv {
        return Err(ImgToolError::new(
            "imgtool: --sort and --csv don't make sense to use together.",
        ));
    }
    if sort && list {
        return Err(ImgToolError::new(
            "imgtool: --sort and --list don't make sense to use together.",
        ));
    }

    let filename = filename.ok_or_else(|| ImgToolError::with_help("cat: no filename provided"))?;
    let image = load_image(Path::new(filename))?;
    if list && image.channel_names.len() != 1 {
        return Err(ImgToolError::new(
            "cat: --list requires a single-channel image",
        ));
    }

    let pixel_count = (image.raw.resolution.x * image.raw.resolution.y) as usize;
    let mut pixels: Vec<usize> = (0..pixel_count).collect();
    if sort {
        pixels.sort_by(|&a, &b| {
            let a_average: Float = (0..image.raw.channels)
                .map(|channel| channel_value(&image.raw, a, channel))
                .sum::<Float>()
                / image.raw.channels as Float;
            let b_average: Float = (0..image.raw.channels)
                .map(|channel| channel_value(&image.raw, b, channel))
                .sum::<Float>()
                / image.raw.channels as Float;
            a_average.total_cmp(&b_average)
        });
    }

    for pixel in pixels {
        let x = pixel as i32 % image.raw.resolution.x;
        let y = pixel as i32 / image.raw.resolution.x;
        let values: Vec<String> = (0..image.raw.channels)
            .map(|channel| format_value(channel_value(&image.raw, pixel, channel)))
            .collect();
        if list {
            if x == 0 {
                print!("{{");
            }
            print!("{}", values.join(", "));
            if x + 1 == image.raw.resolution.x {
                println!("}}");
            } else {
                print!(", ");
            }
        } else {
            if !csv {
                print!("({x}, {y}): ");
            }
            println!("{}", values.join(","));
        }
    }
    Ok(())
}

fn pixel_format(image: &RawImageData) -> &'static str {
    match image {
        RawImageData::F32(_) => "Float",
        RawImageData::F16(_) => "Half",
        RawImageData::U8 { .. } => "U256",
    }
}

fn info(args: &[OsString]) -> Result<(), ImgToolError> {
    if args.is_empty() {
        return Err(ImgToolError::with_help("info: no filenames provided"));
    }
    for filename in args {
        let path = Path::new(filename);
        let image = load_image(path)?;
        println!("{}:", path.display());
        println!(
            "\tresolution ({}, {})",
            image.raw.resolution.x, image.raw.resolution.y
        );
        println!(
            "\tcolor space: {}",
            image
                .metadata
                .color_space
                .map(|color_space| color_space.name)
                .unwrap_or("unknown")
        );
        if let Some(bounds) = image.metadata.pixel_bounds {
            println!(
                "\tpixel bounds: ({}, {})-({}, {})",
                bounds.min.x, bounds.min.y, bounds.max.x, bounds.max.y
            );
        }
        if let Some(full_resolution) = image.metadata.full_resolution {
            println!(
                "\tfull resolution: ({}, {})",
                full_resolution.x, full_resolution.y
            );
        }
        if let Some(samples_per_pixel) = image.metadata.samples_per_pixel {
            println!("\tsamples per pixel: {samples_per_pixel}");
        }
        println!("\tpixel format: {}", pixel_format(&image.raw.data));
        println!("\tChannels:");
        for channel in 0..image.raw.channels {
            let mut min = Float::INFINITY;
            let mut max = Float::NEG_INFINITY;
            let mut sum = 0.0;
            let mut valid = 0;
            let mut infinite = 0;
            let mut nan = 0;
            for pixel in 0..(image.raw.resolution.x * image.raw.resolution.y) as usize {
                let value = channel_value(&image.raw, pixel, channel);
                if value.is_nan() {
                    nan += 1;
                } else if value.is_infinite() {
                    infinite += 1;
                } else {
                    min = min.min(value);
                    max = max.max(value);
                    sum += value;
                    valid += 1;
                }
            }
            let average = if valid == 0 {
                Float::NAN
            } else {
                sum / valid as Float
            };
            println!(
                "\t    {:>20}: min {:>12.6} max {:>12.6} avg {:>12.6} ({} infinite, {} not-a-number)",
                image.channel_names[channel], min, max, average, infinite, nan
            );
        }
    }
    Ok(())
}

fn parse_crop(value: &str) -> Result<[i32; 4], ImgToolError> {
    let values: Vec<i32> = value
        .split(',')
        .map(|part| {
            part.parse::<i32>()
                .map_err(|_| ImgToolError::new("convert: invalid --crop value"))
        })
        .collect::<Result<_, _>>()?;
    values
        .try_into()
        .map_err(|_| ImgToolError::new("convert: --crop requires x0,x1,y0,y1"))
}

fn write_converted_image(
    path: &Path,
    resolution: Vector2<i32>,
    names: &[String],
    data: &[Float],
    format: PixelFormat,
) -> Result<(), ImgToolError> {
    let filename = path
        .to_str()
        .ok_or_else(|| ImgToolError::new("convert: output path is not valid UTF-8"))?;
    if path.extension().is_some_and(|extension| extension == "exr") {
        let image = Image::from_channels_with_format(
            Point2i::new(resolution.x, resolution.y),
            names.to_vec(),
            data.to_vec(),
            format,
        );
        let bounds = Bounds2i::from(((0, 0), (resolution.x, resolution.y)));
        image.write_exr(filename, &bounds, &Point2i::new(resolution.x, resolution.y))?;
        return Ok(());
    }

    if format != PixelFormat::U256 {
        return Err(ImgToolError::new(
            "convert: non-EXR output requires an 8-bit image",
        ));
    }
    let rgb = match names.len() {
        1 => data
            .chunks_exact(1)
            .flat_map(|pixel| [pixel[0], pixel[0], pixel[0]])
            .collect(),
        3 => data.to_vec(),
        _ => {
            return Err(ImgToolError::new(
                "convert: non-EXR output requires one or three channels",
            ));
        }
    };
    write_image_bytes(
        filename,
        &rgb,
        &Bounds2i::from(((0, 0), (resolution.x, resolution.y))),
        &Point2i::new(resolution.x, resolution.y),
    )?;
    Ok(())
}

fn convert(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut input = None;
    let mut output = None;
    let mut channels = None;
    let mut crop = None;
    let mut flip_y = false;
    let mut scale = 1.0;
    let mut gamma = 1.0;
    let mut fp16 = false;

    let mut index = 0;
    while index < args.len() {
        let argument = args[index].to_string_lossy();
        let value = |index: &mut usize| -> Result<&str, ImgToolError> {
            *index += 1;
            args.get(*index)
                .and_then(|value| value.to_str())
                .ok_or_else(|| ImgToolError::with_help("convert: option value missing"))
        };
        match argument.as_ref() {
            "--outfile" | "-outfile" => output = Some(value(&mut index)?.to_string()),
            "--channels" | "-channels" => channels = Some(value(&mut index)?.to_string()),
            "--crop" | "-crop" => crop = Some(parse_crop(value(&mut index)?)?),
            "--scale" | "-scale" => {
                scale = value(&mut index)?
                    .parse()
                    .map_err(|_| ImgToolError::new("convert: invalid --scale value"))?
            }
            "--gamma" | "-gamma" => {
                gamma = value(&mut index)?
                    .parse()
                    .map_err(|_| ImgToolError::new("convert: invalid --gamma value"))?
            }
            "--flipy" | "-flipy" => flip_y = !flip_y,
            "--fp16" | "-fp16" => fp16 = true,
            "--bw" | "-bw" | "--clamp" | "-clamp" | "--colorspace" | "-colorspace"
            | "--tonemap" | "-tonemap" => {
                return Err(ImgToolError::new(format!(
                    "convert: option {argument} is not implemented"
                )))
            }
            value if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "convert: unknown option \"{value}\""
                )))
            }
            value if input.is_none() => input = Some(value.to_string()),
            _ => {
                return Err(ImgToolError::new(
                    "convert: multiple input filenames provided",
                ))
            }
        }
        index += 1;
    }

    let input = input.ok_or_else(|| ImgToolError::with_help("convert: input filename missing"))?;
    let output = output.ok_or_else(|| ImgToolError::with_help("convert: --outfile missing"))?;
    if scale == 0.0 || gamma <= 0.0 {
        return Err(ImgToolError::new(
            "convert: --scale must be non-zero and --gamma must be positive",
        ));
    }
    let image = load_image(Path::new(&input))?;
    let selected: Vec<usize> = match channels {
        Some(names) => names
            .split(',')
            .map(|name| {
                image
                    .channel_names
                    .iter()
                    .position(|candidate| candidate == name)
                    .ok_or_else(|| ImgToolError::new(format!("convert: channel {name} missing")))
            })
            .collect::<Result<_, _>>()?,
        None => (0..image.raw.channels).collect(),
    };
    let (x0, x1, y0, y1) = crop.map_or(
        (0, image.raw.resolution.x, 0, image.raw.resolution.y),
        |[x0, x1, y0, y1]| (x0, x1, y0, y1),
    );
    if x0 < 0
        || y0 < 0
        || x0 >= x1
        || y0 >= y1
        || x1 > image.raw.resolution.x
        || y1 > image.raw.resolution.y
    {
        return Err(ImgToolError::new("convert: crop is outside image bounds"));
    }
    let resolution = Vector2::new(x1 - x0, y1 - y0);
    let names: Vec<String> = selected
        .iter()
        .map(|&channel| image.channel_names[channel].clone())
        .collect();
    let mut data = Vec::with_capacity((resolution.x * resolution.y) as usize * selected.len());
    for y in 0..resolution.y {
        let source_y = if flip_y { y1 - 1 - y } else { y0 + y };
        for x in 0..resolution.x {
            let pixel = (source_y * image.raw.resolution.x + x0 + x) as usize;
            for &channel in &selected {
                let mut value = image.raw.channel(pixel, channel) * scale;
                if gamma != 1.0 {
                    value = value.signum() * value.abs().powf(gamma);
                }
                data.push(value);
            }
        }
    }
    let format = if fp16 {
        PixelFormat::Half
    } else if Path::new(&output)
        .extension()
        .is_some_and(|extension| extension == "exr")
    {
        PixelFormat::Float
    } else {
        PixelFormat::U256
    };
    write_converted_image(Path::new(&output), resolution, &names, &data, format)
}

fn image_data(image: &LoadedImage) -> Vec<Float> {
    (0..(image.raw.resolution.x * image.raw.resolution.y) as usize)
        .flat_map(|pixel| {
            (0..image.raw.channels).map(move |channel| image.raw.channel(pixel, channel))
        })
        .collect()
}

fn assert_compatible(first: &LoadedImage, second: &LoadedImage) -> Result<(), ImgToolError> {
    if first.raw.resolution != second.raw.resolution || first.channel_names != second.channel_names
    {
        return Err(ImgToolError::new(
            "imgtool: image resolution or channel layout does not match",
        ));
    }
    if first
        .metadata
        .color_space
        .map(|color_space| color_space.name)
        != second
            .metadata
            .color_space
            .map(|color_space| color_space.name)
    {
        return Err(ImgToolError::new(
            "imgtool: image color spaces do not match",
        ));
    }
    Ok(())
}

fn output_exr(path: &str, image: &LoadedImage, data: Vec<Float>) -> Result<(), ImgToolError> {
    output_exr_region(
        path,
        &image.metadata,
        image.raw.resolution,
        &image.channel_names,
        data,
    )
}

fn output_exr_region(
    path: &str,
    metadata: &ImageMetadata,
    resolution: Vector2<i32>,
    channel_names: &[String],
    data: Vec<Float>,
) -> Result<(), ImgToolError> {
    output_exr_region_format(
        path,
        metadata,
        resolution,
        channel_names,
        data,
        PixelFormat::Float,
    )
}

fn output_exr_region_format(
    path: &str,
    metadata: &ImageMetadata,
    resolution: Vector2<i32>,
    channel_names: &[String],
    data: Vec<Float>,
    format: PixelFormat,
) -> Result<(), ImgToolError> {
    let mut output = Image::from_channels_with_format(
        Point2i::new(resolution.x, resolution.y),
        channel_names.to_vec(),
        data,
        format,
    );
    *output.metadata_mut() = metadata.clone();
    let bounds = Bounds2i::from(((0, 0), (resolution.x, resolution.y)));
    output.write_exr(path, &bounds, &Point2i::new(resolution.x, resolution.y))?;
    Ok(())
}

fn average(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut base = None;
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].to_str() {
            Some("--outfile") | Some("-outfile") => {
                index += 1;
                output = Some(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| ImgToolError::with_help("average: --outfile missing"))?
                        .to_string(),
                );
            }
            Some(value) if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "average: unknown option {value}"
                )))
            }
            Some(value) if base.is_none() => base = Some(value.to_string()),
            _ => {
                return Err(ImgToolError::new(
                    "average: multiple base filenames provided",
                ))
            }
        }
        index += 1;
    }
    let base = base.ok_or_else(|| ImgToolError::with_help("average: base filename missing"))?;
    let output = output.ok_or_else(|| ImgToolError::with_help("average: --outfile missing"))?;
    let path = Path::new(&base);
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    let prefix = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(&base);
    let mut filenames: Vec<_> = fs::read_dir(directory)
        .map_err(|error| ImgToolError::new(error.to_string()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|candidate| {
            candidate
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix))
        })
        .collect();
    filenames.sort();
    if filenames.is_empty() {
        return Err(ImgToolError::new("average: no matching files"));
    }
    let first = load_image(&filenames[0])?;
    let mut sum =
        vec![0.0; (first.raw.resolution.x * first.raw.resolution.y) as usize * first.raw.channels];
    let file_count = filenames.len();
    for filename in filenames {
        let image = load_image(&filename)?;
        assert_compatible(&first, &image)?;
        for (total, value) in sum.iter_mut().zip(image_data(&image)) {
            *total += value;
        }
    }
    for value in &mut sum {
        *value /= file_count as Float;
    }
    output_exr(&output, &first, sum)
}

fn diff(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut input = None;
    let mut reference = None;
    let mut channels = None;
    let mut crop = None;
    let mut diff_tolerance = 0.0;
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
                        .ok_or_else(|| ImgToolError::with_help("diff: --reference missing"))?
                        .to_string(),
                );
            }
            Some("--metric") | Some("-metric") => {
                index += 1;
                metric = args
                    .get(index)
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| ImgToolError::with_help("diff: --metric missing"))?
                    .to_string();
            }
            Some("--channels") | Some("-channels") => {
                index += 1;
                channels = Some(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| ImgToolError::with_help("diff: --channels missing"))?
                        .to_string(),
                );
            }
            Some("--crop") | Some("-crop") => {
                index += 1;
                crop = Some(parse_crop(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| ImgToolError::with_help("diff: --crop missing"))?,
                )?);
            }
            Some("--difftol") | Some("-difftol") => {
                index += 1;
                diff_tolerance = args
                    .get(index)
                    .and_then(|value| value.to_str())
                    .ok_or_else(|| ImgToolError::with_help("diff: --difftol missing"))?
                    .parse()
                    .map_err(|_| ImgToolError::new("diff: invalid --difftol value"))?;
            }
            Some("--outfile") | Some("-outfile") => {
                index += 1;
                output = Some(
                    args.get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| ImgToolError::with_help("diff: --outfile missing"))?
                        .to_string(),
                );
            }
            Some(value) if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "diff: unknown option {value}"
                )))
            }
            Some(value) if input.is_none() => input = Some(value.to_string()),
            _ => return Err(ImgToolError::new("diff: multiple input filenames provided")),
        }
        index += 1;
    }
    let input = input.ok_or_else(|| ImgToolError::with_help("diff: input filename missing"))?;
    let reference =
        reference.ok_or_else(|| ImgToolError::with_help("diff: --reference missing"))?;
    if !matches!(metric.as_str(), "MSE" | "MAE" | "MRSE") {
        if metric == "FLIP" {
            return Err(ImgToolError::new(
                "diff: FLIP metric is not implemented in this build",
            ));
        }
        return Err(ImgToolError::new("diff: metric must be MSE, MAE, or MRSE"));
    }
    let image = load_image(Path::new(&input))?;
    let reference = load_image(Path::new(&reference))?;
    if image
        .metadata
        .color_space
        .map(|color_space| color_space.name)
        != reference
            .metadata
            .color_space
            .map(|color_space| color_space.name)
    {
        return Err(ImgToolError::new(
            "imgtool: image color spaces do not match",
        ));
    }
    let selected: Vec<usize> = channels
        .unwrap_or_else(|| image.channel_names.join(","))
        .split(',')
        .map(|name| {
            image
                .channel_names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| ImgToolError::new(format!("diff: channel {name} missing")))
        })
        .collect::<Result<_, _>>()?;
    let reference_selected: Vec<usize> = selected
        .iter()
        .map(|&channel| {
            reference
                .channel_names
                .iter()
                .position(|name| name == &image.channel_names[channel])
                .ok_or_else(|| {
                    ImgToolError::new(format!(
                        "diff: reference channel {} missing",
                        image.channel_names[channel]
                    ))
                })
        })
        .collect::<Result<_, _>>()?;
    let [x0, x1, y0, y1] = crop.unwrap_or([0, image.raw.resolution.x, 0, image.raw.resolution.y]);
    if x0 < 0
        || y0 < 0
        || x0 >= x1
        || y0 >= y1
        || x1 > image.raw.resolution.x
        || y1 > image.raw.resolution.y
        || x1 > reference.raw.resolution.x
        || y1 > reference.raw.resolution.y
    {
        return Err(ImgToolError::new("diff: crop is outside image bounds"));
    }
    if diff_tolerance < 0.0 {
        return Err(ImgToolError::new("diff: --difftol must not be negative"));
    }
    let resolution = Vector2::new(x1 - x0, y1 - y0);
    let mut errors = vec![0.0; selected.len()];
    let mut output_data =
        Vec::with_capacity((resolution.x * resolution.y) as usize * selected.len());
    for y in y0..y1 {
        for x in x0..x1 {
            let image_pixel = (y * image.raw.resolution.x + x) as usize;
            let reference_pixel = (y * reference.raw.resolution.x + x) as usize;
            for (index, (&channel, &reference_channel)) in
                selected.iter().zip(&reference_selected).enumerate()
            {
                let a = image.raw.channel(image_pixel, channel);
                let b = reference.raw.channel(reference_pixel, reference_channel);
                let delta = a - b;
                let error = match metric.as_str() {
                    "MAE" => delta.abs(),
                    "MRSE" => delta * delta / (b + 0.01).powi(2),
                    _ => delta * delta,
                };
                if error.is_finite() {
                    errors[index] += error;
                }
                output_data.push(error);
            }
        }
    }
    let pixels = (resolution.x * resolution.y) as Float;
    let means: Vec<Float> = errors.iter().map(|error| error / pixels).collect();
    println!("{}:", input);
    for (channel, error) in selected.iter().zip(&means) {
        println!("{}: {:.6}", image.channel_names[*channel], error);
    }
    if let Some(output) = output {
        let names = selected
            .iter()
            .map(|&channel| image.channel_names[channel].clone())
            .collect::<Vec<_>>();
        output_exr_region(&output, &image.metadata, resolution, &names, output_data)?;
    }
    let average = means.iter().sum::<Float>() / means.len() as Float;
    if average * 100.0 > diff_tolerance {
        Err(ImgToolError::new(format!(
            "diff: images differ ({} = {:.6}, tolerance = {:.6}%)",
            metric, average, diff_tolerance
        )))
    } else {
        Ok(())
    }
}

fn makesky(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut outfile = None;
    let mut albedo = 0.5_f64;
    let mut turbidity = 3.0_f64;
    let mut elevation = 10.0_f64;
    let mut resolution = 2048_i32;
    let mut index = 0;
    while index < args.len() {
        let argument = args[index].to_string_lossy();
        let value = |index: &mut usize| -> Result<&str, ImgToolError> {
            *index += 1;
            args.get(*index)
                .and_then(|value| value.to_str())
                .ok_or_else(|| ImgToolError::with_help("makesky: option value missing"))
        };
        match argument.as_ref() {
            "--outfile" | "-outfile" => outfile = Some(value(&mut index)?.to_string()),
            "--albedo" | "-albedo" => {
                albedo = value(&mut index)?
                    .parse()
                    .map_err(|_| ImgToolError::new("makesky: invalid --albedo"))?
            }
            "--turbidity" | "-turbidity" => {
                turbidity = value(&mut index)?
                    .parse()
                    .map_err(|_| ImgToolError::new("makesky: invalid --turbidity"))?
            }
            "--elevation" | "-elevation" => {
                elevation = value(&mut index)?
                    .parse()
                    .map_err(|_| ImgToolError::new("makesky: invalid --elevation"))?
            }
            "--resolution" | "-resolution" => {
                resolution = value(&mut index)?
                    .parse()
                    .map_err(|_| ImgToolError::new("makesky: invalid --resolution"))?
            }
            value if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "makesky: unknown option {value}"
                )))
            }
            _ => return Err(ImgToolError::new("makesky: unexpected argument")),
        }
        index += 1;
    }

    let outfile = outfile.ok_or_else(|| ImgToolError::with_help("makesky: --outfile missing"))?;
    if !(0.0..=1.0).contains(&albedo) {
        return Err(ImgToolError::new("makesky: albedo must be between 0 and 1"));
    }
    if !(1.7..=10.0).contains(&turbidity) {
        return Err(ImgToolError::new(
            "makesky: turbidity must be between 1.7 and 10",
        ));
    }
    if !(0.0..=90.0).contains(&elevation) {
        return Err(ImgToolError::new(
            "makesky: elevation must be between 0 and 90 degrees",
        ));
    }
    if resolution < 1 {
        return Err(ImgToolError::new("makesky: resolution must be at least 1"));
    }
    if !Path::new(&outfile)
        .extension()
        .is_some_and(|extension| extension == "exr")
    {
        return Err(ImgToolError::new("makesky: output must be an EXR file"));
    }

    let elevation_radians = elevation.to_radians();
    let model = HosekSkyModel::new(elevation_radians, turbidity, albedo)?;
    let sun_direction = Vector2::new(elevation_radians.cos(), elevation_radians.sin());
    let wavelengths: Vec<f64> = (0..=12)
        .map(|index| 320.0 + (index as f64 / 12.0) * (720.0 - 320.0))
        .collect();
    let mut texels = Vec::with_capacity((resolution * resolution) as usize);
    for y in 0..resolution {
        for x in 0..resolution {
            let uv = Point2f::new(
                (x as Float + 0.5) / resolution as Float,
                (y as Float + 0.5) / resolution as Float,
            );
            let direction = equal_area_square_to_sphere(&uv);
            if direction.z <= 0.0 {
                texels.push(RGBSpectrum::zero());
                continue;
            }
            let theta = spherical_theta(&direction) as f64;
            let sun_dot =
                direction.y as f64 * sun_direction.x + direction.z as f64 * sun_direction.y;
            let gamma = safe_acos(sun_dot as Float) as f64;
            let values: Vec<Float> = wavelengths
                .iter()
                .map(|&wavelength| model.solar_radiance(theta, gamma, wavelength).unwrap() as Float)
                .collect();
            let spectrum = PiecewiseLinearSpectrum::new(
                wavelengths.iter().map(|&value| value as Float).collect(),
                values,
            );
            let xyz = spectrum.evaluate().to_xyz();
            let rgb = ACES2065_1.xyz_to_rgb(xyz);
            texels.push(RGBSpectrum::new(rgb[0], rgb[1], rgb[2]));
        }
    }
    let mut image =
        Image::with_color_space(Point2i::new(resolution, resolution), texels, &ACES2065_1)?;
    image
        .metadata_mut()
        .strings
        .insert("makesky.albedo".to_string(), albedo.to_string());
    image.metadata_mut().strings.insert(
        "makesky.elevation".to_string(),
        elevation_radians.to_string(),
    );
    image
        .metadata_mut()
        .strings
        .insert("makesky.turbidity".to_string(), turbidity.to_string());
    let bounds = Bounds2i::from(((0, 0), (resolution, resolution)));
    image.write_exr(&outfile, &bounds, &Point2i::new(resolution, resolution))?;
    Ok(())
}

fn run<I, T>(arguments: I) -> Result<(), ImgToolError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let mut arguments = arguments.into_iter().map(Into::into);
    let _program = arguments.next();
    let Some(command) = arguments.next() else {
        eprintln!("{0}", help_text());
        return Ok(());
    };
    let command = command
        .into_string()
        .map_err(|_| ImgToolError::new("command is not valid UTF-8"))?;
    let arguments: Vec<OsString> = arguments.collect();

    match command.as_str() {
        "help" | "--help" | "-help" | "-h" => {
            if arguments.is_empty() {
                eprintln!("{0}", help_text());
            } else {
                for command in arguments {
                    let command = command
                        .into_string()
                        .map_err(|_| ImgToolError::new("command is not valid UTF-8"))?;
                    eprintln!("{}", command_help(&command)?);
                }
            }
            Ok(())
        }
        "cat" => cat(&arguments),
        "info" => info(&arguments),
        "convert" => convert(&arguments),
        "average" => average(&arguments),
        "diff" => diff(&arguments),
        "error" => comparison::error(&arguments),
        "assemble" => assembly::assemble(&arguments),
        "splitn" => assembly::splitn(&arguments),
        "scalenormalmap" => pixel_ops::scalenormalmap(&arguments),
        "falsecolor" => pixel_ops::falsecolor(&arguments),
        "bloom" => pixel_ops::bloom(&arguments),
        "whitebalance" => color_ops::whitebalance(&arguments),
        "makeequiarea" => color_ops::makeequiarea(&arguments),
        "makesky" => makesky(&arguments),
        _ => Err(ImgToolError::with_help(format!(
            "imgtool: command \"{command}\" not known."
        ))),
    }
}

fn main() {
    match run(std::env::args_os()) {
        Ok(()) => std::process::exit(0),
        Err(error) => {
            eprintln!("imgtool: {error}");
            if error.show_help {
                eprintln!("{}", help_text());
            }
            std::process::exit(1);
        }
    }
}
