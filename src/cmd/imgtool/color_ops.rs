use super::{load_image, output_exr_region, ImgToolError};
use pbrt_r4::filters::GaussianFilter;
use pbrt_r4::util::base::{Float, Point2f};
use pbrt_r4::util::geometry::{
    equal_area_square_to_sphere, spherical_phi, spherical_theta, Vector2,
};
use pbrt_r4::util::image::{Image, ImageWrapMode, PixelFormat};
use pbrt_r4::util::rng::RNG;
use pbrt_r4::util::spectrum::d_illuminant::d_illuminant;
use pbrt_r4::util::spectrum::named::lookup_named_spectrum;
use pbrt_r4::util::spectrum::rgb_to_spectrum::RGBColorSpace;
use pbrt_r4::util::spectrum::{spectrum_to_xyz, Spectrum};
use std::ffi::OsString;
use std::path::Path;

const LMS_FROM_XYZ: [[Float; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];
const XYZ_FROM_LMS: [[Float; 3]; 3] = [
    [0.986993, -0.147054, 0.159963],
    [0.432305, 0.51836, 0.0492912],
    [-0.00852866, 0.0400428, 0.968487],
];

pub fn whitebalance(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut input = None;
    let mut output = None;
    let mut illuminant = None;
    let mut temperature = None;
    let mut primaries = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].to_str() {
            Some("--outfile") | Some("-outfile") => {
                index += 1;
                output = Some(option_value(
                    args,
                    index,
                    "whitebalance: --outfile missing",
                )?);
            }
            Some("--illuminant") | Some("-illuminant") => {
                index += 1;
                illuminant = Some(option_value(
                    args,
                    index,
                    "whitebalance: --illuminant missing",
                )?);
            }
            Some("--temperature") | Some("-temperature") => {
                index += 1;
                temperature = Some(parse_value(
                    args,
                    index,
                    "whitebalance: invalid --temperature",
                )?);
            }
            Some("--primaries") | Some("-primaries") => {
                let x = parse_value(args, index + 1, "whitebalance: invalid --primaries")?;
                let y = parse_value(args, index + 2, "whitebalance: invalid --primaries")?;
                primaries = Some([x, y]);
                index += 2;
            }
            Some(value) if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "whitebalance: unknown option {value}"
                )))
            }
            Some(value) if input.is_none() => input = Some(value.to_string()),
            _ => return Err(ImgToolError::new("whitebalance: multiple input filenames")),
        }
        index += 1;
    }

    let input = input.ok_or_else(|| ImgToolError::with_help("whitebalance: input missing"))?;
    let output =
        output.ok_or_else(|| ImgToolError::with_help("whitebalance: --outfile missing"))?;
    if illuminant.is_some() as u8 + temperature.is_some() as u8 + primaries.is_some() as u8 != 1 {
        return Err(ImgToolError::with_help(
            "whitebalance: exactly one of --illuminant, --temperature, or --primaries is required",
        ));
    }

    let image = load_image(Path::new(&input))?;
    let channels = ["R", "G", "B"]
        .map(|name| {
            image
                .channel_names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| ImgToolError::new("whitebalance: image needs R, G, and B channels"))
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let color_space = image
        .metadata
        .color_space
        .ok_or_else(|| ImgToolError::new("whitebalance: input color space is missing"))?;
    let source_white = if let Some(name) = illuminant {
        let spectrum = lookup_named_spectrum(&format!("stdillum-{name}"))
            .ok_or_else(|| ImgToolError::new("whitebalance: illuminant unknown"))?;
        chromaticity(&spectrum)?
    } else if let Some(kelvin) = temperature {
        chromaticity(&Spectrum::PiecewiseLinear(d_illuminant(kelvin)))?
    } else {
        primaries.expect("validated whitebalance source")
    };
    let matrix = white_balance_matrix(source_white, color_space.w, color_space);
    let pixels = (image.raw.resolution.x * image.raw.resolution.y) as usize;
    let mut data = Vec::with_capacity(pixels * image.raw.channels);
    for pixel in 0..pixels {
        let rgb = [
            image.raw.channel(pixel, channels[0]),
            image.raw.channel(pixel, channels[1]),
            image.raw.channel(pixel, channels[2]),
        ];
        let balanced = mul_matrix_vector(&matrix, &rgb);
        for channel in 0..image.raw.channels {
            data.push(match channels.iter().position(|&value| value == channel) {
                Some(0) => balanced[0],
                Some(1) => balanced[1],
                Some(2) => balanced[2],
                Some(_) => image.raw.channel(pixel, channel),
                None => image.raw.channel(pixel, channel),
            });
        }
    }
    output_exr_region(
        &output,
        &image.metadata,
        image.raw.resolution,
        &image.channel_names,
        data,
    )
}

fn option_value(args: &[OsString], index: usize, message: &str) -> Result<String, ImgToolError> {
    args.get(index)
        .and_then(|value| value.to_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| ImgToolError::with_help(message))
}

fn parse_value(args: &[OsString], index: usize, message: &str) -> Result<Float, ImgToolError> {
    option_value(args, index, message)?
        .parse()
        .map_err(|_| ImgToolError::new(message))
}

fn chromaticity(spectrum: &Spectrum) -> Result<[Float; 2], ImgToolError> {
    let xyz = spectrum_to_xyz(spectrum);
    let sum = xyz.iter().sum::<Float>();
    if sum <= 0.0 || !sum.is_finite() {
        return Err(ImgToolError::new("whitebalance: invalid illuminant"));
    }
    Ok([xyz[0] / sum, xyz[1] / sum])
}

fn xyz_from_xy(xy: [Float; 2]) -> [Float; 3] {
    [xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]]
}

fn rgb_to_xyz_matrix(color_space: &RGBColorSpace) -> [[Float; 3]; 3] {
    let primaries = [
        xyz_from_xy(color_space.r),
        xyz_from_xy(color_space.g),
        xyz_from_xy(color_space.b),
    ];
    let matrix = [
        [primaries[0][0], primaries[1][0], primaries[2][0]],
        [primaries[0][1], primaries[1][1], primaries[2][1]],
        [primaries[0][2], primaries[1][2], primaries[2][2]],
    ];
    let scale = mul_matrix_vector(&invert_3x3(matrix), &xyz_from_xy(color_space.w));
    [
        [
            matrix[0][0] * scale[0],
            matrix[0][1] * scale[1],
            matrix[0][2] * scale[2],
        ],
        [
            matrix[1][0] * scale[0],
            matrix[1][1] * scale[1],
            matrix[1][2] * scale[2],
        ],
        [
            matrix[2][0] * scale[0],
            matrix[2][1] * scale[1],
            matrix[2][2] * scale[2],
        ],
    ]
}

fn white_balance_matrix(
    source_white: [Float; 2],
    target_white: [Float; 2],
    color_space: &RGBColorSpace,
) -> [[Float; 3]; 3] {
    let source_lms = mul_matrix_vector(&LMS_FROM_XYZ, &xyz_from_xy(source_white));
    let target_lms = mul_matrix_vector(&LMS_FROM_XYZ, &xyz_from_xy(target_white));
    let adaptation = [
        [target_lms[0] / source_lms[0], 0.0, 0.0],
        [0.0, target_lms[1] / source_lms[1], 0.0],
        [0.0, 0.0, target_lms[2] / source_lms[2]],
    ];
    let rgb_to_xyz = rgb_to_xyz_matrix(color_space);
    let xyz_to_rgb = invert_3x3(rgb_to_xyz);
    multiply_matrix(
        &xyz_to_rgb,
        &multiply_matrix(&XYZ_FROM_LMS, &multiply_matrix(&adaptation, &LMS_FROM_XYZ)),
    )
}

fn multiply_matrix(a: &[[Float; 3]; 3], b: &[[Float; 3]; 3]) -> [[Float; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            result[row][column] = (0..3).map(|i| a[row][i] * b[i][column]).sum();
        }
    }
    result
}

fn mul_matrix_vector(matrix: &[[Float; 3]; 3], vector: &[Float; 3]) -> [Float; 3] {
    matrix.map(|row| row.iter().zip(vector).map(|(a, b)| a * b).sum())
}

fn invert_3x3(m: [[Float; 3]; 3]) -> [[Float; 3]; 3] {
    let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let inv = 1.0 / determinant;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ]
}

pub fn makeequiarea(args: &[OsString]) -> Result<(), ImgToolError> {
    let mut input = None;
    let mut output = None;
    let mut resolution = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].to_str() {
            Some("--outfile") | Some("-outfile") => {
                index += 1;
                output = Some(option_value(
                    args,
                    index,
                    "makeequiarea: --outfile missing",
                )?);
            }
            Some("--resolution") | Some("-resolution") => {
                index += 1;
                resolution =
                    Some(parse_value(args, index, "makeequiarea: invalid --resolution")? as i32);
            }
            Some(value) if value.starts_with('-') => {
                return Err(ImgToolError::with_help(format!(
                    "makeequiarea: unknown option {value}"
                )))
            }
            Some(value) if input.is_none() => input = Some(value.to_string()),
            _ => return Err(ImgToolError::new("makeequiarea: multiple input filenames")),
        }
        index += 1;
    }
    let input = input.ok_or_else(|| ImgToolError::with_help("makeequiarea: input missing"))?;
    let output =
        output.ok_or_else(|| ImgToolError::with_help("makeequiarea: --outfile missing"))?;
    let image = load_image(Path::new(&input))?;
    if image.raw.resolution.x != 2 * image.raw.resolution.y {
        eprintln!(
            "{}: resolution ({}, {}) doesn't have a 2:1 aspect ratio.",
            input, image.raw.resolution.x, image.raw.resolution.y
        );
    }
    let resolution = resolution.unwrap_or(image.raw.resolution.x);
    if resolution < 1 {
        return Err(ImgToolError::new(
            "makeequiarea: resolution must be positive",
        ));
    }
    let mut source_data = Vec::with_capacity(
        (image.raw.resolution.x * image.raw.resolution.y * image.raw.channels as i32) as usize,
    );
    for pixel in 0..(image.raw.resolution.x * image.raw.resolution.y) as usize {
        for channel in 0..image.raw.channels {
            source_data.push(image.raw.channel(pixel, channel));
        }
    }
    let source = Image::from_channels_with_format(
        image.raw.resolution,
        image.channel_names.clone(),
        source_data,
        PixelFormat::Float,
    );
    let filter = GaussianFilter::new(&Vector2::new(1.5, 1.5), 2.0);
    let mut output_data =
        vec![0.0; (resolution * resolution * image.raw.channels as i32 * 1) as usize];
    for v in 0..resolution {
        let mut rng = RNG::new_sequence(v as u64);
        for u in 0..resolution {
            let mut sum = vec![0.0 as Float; image.raw.channels];
            let mut weight_sum = 0.0 as Float;
            for dv in 0..6 {
                for du in 0..6 {
                    let sample = Point2f::new(
                        (du as Float + rng.uniform_float()) / 6.0,
                        (dv as Float + rng.uniform_float()) / 6.0,
                    );
                    let filter_sample = filter.sample(&sample);
                    let square = wrap_equal_area_square(Point2f::new(
                        (u as Float + 0.5 + filter_sample.p.x) / resolution as Float,
                        (v as Float + 0.5 + filter_sample.p.y) / resolution as Float,
                    ));
                    let direction = equal_area_square_to_sphere(&square);
                    let uv = Point2f::new(
                        spherical_phi(&direction) / (2.0 * std::f32::consts::PI as Float),
                        spherical_theta(&direction) / (std::f32::consts::PI as Float),
                    );
                    for channel in 0..image.raw.channels {
                        sum[channel] += filter_sample.weight
                            * source.bilerp_channel_with_wrap(
                                &uv,
                                channel,
                                ImageWrapMode::Repeat,
                                ImageWrapMode::Clamp,
                            );
                    }
                    weight_sum += filter_sample.weight;
                }
            }
            let pixel = (v * resolution + u) as usize * image.raw.channels;
            for channel in 0..image.raw.channels {
                output_data[pixel + channel] = (sum[channel] / weight_sum).max(0.0);
            }
        }
    }
    output_exr_region(
        &output,
        &image.metadata,
        Vector2::new(resolution, resolution),
        &image.channel_names,
        output_data,
    )
}

fn wrap_equal_area_square(mut uv: Point2f) -> Point2f {
    if uv.x < 0.0 {
        uv.x = -uv.x;
        uv.y = 1.0 - uv.y;
    } else if uv.x > 1.0 {
        uv.x = 2.0 - uv.x;
        uv.y = 1.0 - uv.y;
    }
    if uv.y < 0.0 {
        uv.x = 1.0 - uv.x;
        uv.y = -uv.y;
    } else if uv.y > 1.0 {
        uv.x = 1.0 - uv.x;
        uv.y = 2.0 - uv.y;
    }
    uv
}
