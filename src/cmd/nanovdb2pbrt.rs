use nanovdb_rs::{GridClass, GridType, NvdbFile};
use std::env;
use std::fmt::{Display, Formatter};
use std::io::{self, Write};
use std::path::Path;
use std::process::ExitCode;

#[derive(Debug)]
struct CliError {
    message: String,
    show_usage: bool,
}

impl CliError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            show_usage: false,
        }
    }

    fn usage(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            show_usage: true,
        }
    }
}

impl Display for CliError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CliError {}

const USAGE: &str = "usage: nanovdb2pbrt [<options>] <filename.nvdb>\n\nOptions:\n  --grid <name>        Name of grid to extract. Default: \"density\"\n";

struct Arguments {
    filename: String,
    grid: String,
    _downsample: i32,
}

fn parse_arguments<I, T>(arguments: I) -> Result<Option<Arguments>, CliError>
where
    I: IntoIterator<Item = T>,
    T: Into<String>,
{
    let mut filename = None;
    let mut grid = String::from("density");
    let mut downsample = 0;
    let mut arguments = arguments.into_iter().map(Into::into).peekable();

    while let Some(argument) = arguments.next() {
        if argument == "--help" || argument == "-help" || argument == "-h" {
            return Ok(None);
        }
        if let Some(value) = argument.strip_prefix("--grid=") {
            grid = parse_nonempty(value, "--grid")?;
        } else if let Some(value) = argument.strip_prefix("-grid=") {
            grid = parse_nonempty(value, "--grid")?;
        } else if argument == "--grid" || argument == "-grid" {
            grid = next_option_value(&mut arguments, "--grid", false)?;
        } else if let Some(value) = argument.strip_prefix("--downsample=") {
            downsample = parse_i32(value, "--downsample")?;
        } else if let Some(value) = argument.strip_prefix("-downsample=") {
            downsample = parse_i32(value, "--downsample")?;
        } else if argument == "--downsample" || argument == "-downsample" {
            downsample = parse_i32(
                &next_option_value(&mut arguments, "--downsample", true)?,
                "--downsample",
            )?;
        } else if argument.starts_with('-') {
            return Err(CliError::usage(format!("unknown option: {argument}")));
        } else if filename.is_some() {
            return Err(CliError::usage("multiple input filenames provided"));
        } else {
            filename = Some(argument);
        }
    }

    let Some(filename) = filename else {
        return Err(CliError::usage("must specify a nanovdb filename"));
    };
    Ok(Some(Arguments {
        filename,
        grid,
        _downsample: downsample,
    }))
}

fn next_option_value<I>(
    arguments: &mut std::iter::Peekable<I>,
    name: &str,
    allow_leading_dash: bool,
) -> Result<String, CliError>
where
    I: Iterator<Item = String>,
{
    arguments
        .next()
        .filter(|value| allow_leading_dash || !value.starts_with('-'))
        .ok_or_else(|| CliError::usage(format!("missing value for {name}")))
}

fn parse_nonempty(value: &str, name: &str) -> Result<String, CliError> {
    if value.is_empty() {
        Err(CliError::usage(format!("missing value for {name}")))
    } else {
        Ok(value.to_owned())
    }
}

fn parse_i32(value: &str, name: &str) -> Result<i32, CliError> {
    value
        .parse()
        .map_err(|_| CliError::usage(format!("invalid value for {name}: {value}")))
}

fn convert(arguments: &Arguments) -> Result<String, CliError> {
    let file = NvdbFile::open(Path::new(&arguments.filename))
        .map_err(|error| CliError::new(format!("{}: {error}", arguments.filename)))?;
    let grid = file
        .grids()
        .iter()
        .find(|grid| grid.name() == arguments.grid && grid.value_type() == GridType::Float)
        .ok_or_else(|| {
            CliError::new(format!(
                "{}: didn't find \"{}\" Float grid",
                arguments.filename, arguments.grid
            ))
        })?;

    if !matches!(
        grid.grid_metadata().grid_class,
        GridClass::FogVolume | GridClass::Unknown
    ) {
        return Err(CliError::new(format!(
            "{}: \"{}\" isn't a FogVolume grid",
            arguments.filename, arguments.grid
        )));
    }
    let (index_min, index_max) = grid.index_bbox();
    if index_min.iter().zip(index_max).any(|(&min, max)| min > max) {
        return Err(CliError::new(format!(
            "{}: \"{}\" has an empty index bounding box",
            arguments.filename, arguments.grid
        )));
    }
    let (world_min, world_max) = grid.world_bbox();
    let world_values = [
        world_min.x,
        world_min.y,
        world_min.z,
        world_max.x,
        world_max.y,
        world_max.z,
    ];
    if world_values.iter().any(|value| !value.is_finite()) {
        return Err(CliError::new(format!(
            "{}: \"{}\" has a non-finite world bounding box",
            arguments.filename, arguments.grid
        )));
    }

    let upper = index_max.map(|value| {
        value.checked_add(1).ok_or_else(|| {
            CliError::new(format!(
                "{}: index bounding box is too large",
                arguments.filename
            ))
        })
    });
    let upper = upper.into_iter().collect::<Result<Vec<_>, _>>()?;
    let dimensions = upper
        .iter()
        .zip(index_min)
        .map(|(&max, min)| i64::from(max) - i64::from(min) + 1)
        .collect::<Vec<_>>();
    if dimensions
        .iter()
        .any(|&dimension| dimension <= 0 || dimension > i64::from(i32::MAX))
    {
        return Err(CliError::new(format!(
            "{}: index bounding box dimensions are invalid",
            arguments.filename
        )));
    }
    let expected = dimensions.iter().try_fold(1usize, |product, &dimension| {
        product.checked_mul(dimension as usize).ok_or_else(|| {
            CliError::new(format!("{}: dense grid is too large", arguments.filename))
        })
    })?;

    let mut accessor = grid
        .float_read_accessor()
        .ok_or_else(|| CliError::new("failed to create Float grid accessor"))?;
    let mut values = Vec::with_capacity(expected);
    for z in index_min[2]..=upper[2] {
        for y in index_min[1]..=upper[1] {
            for x in index_min[0]..=upper[0] {
                values.push(accessor.get_value([x, y, z]));
            }
        }
    }
    if values.len() != expected {
        return Err(CliError::new(format!(
            "internal error: voxel count does not match dimensions",
        )));
    }

    let mut output = String::new();
    output.push_str(&format!(
        "\"integer nx\" {} \"integer ny\" {}  \"integer nz\" {}\n",
        dimensions[0], dimensions[1], dimensions[2]
    ));
    output.push_str(&format!(
        "\t\"point3 p0\" [ {:.6} {:.6} {:.6} ] \"point3 p1\" [ {:.6} {:.6} {:.6} ]\n",
        world_min.x as f32,
        world_min.y as f32,
        world_min.z as f32,
        world_max.x as f32,
        world_max.y as f32,
        world_max.z as f32
    ));
    output.push_str(&format!("\t\"float {}\" [\n", arguments.grid));
    for (index, value) in values.iter().enumerate() {
        if *value == 0.0 {
            output.push_str("0 ");
        } else {
            output.push_str(&format!("{value:.6} "));
        }
        if index % 20 == 19 {
            output.push('\n');
        }
    }
    output.push_str("]\n");
    Ok(output)
}

fn run<I, T>(arguments: I) -> Result<Option<String>, CliError>
where
    I: IntoIterator<Item = T>,
    T: Into<String>,
{
    let Some(arguments) = parse_arguments(arguments)? else {
        return Ok(Some(USAGE.to_owned()));
    };
    convert(&arguments).map(Some)
}

fn main() -> ExitCode {
    match run(env::args().skip(1)) {
        Ok(Some(output)) => match io::stdout().write_all(output.as_bytes()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("nanovdb2pbrt: failed to write stdout: {error}");
                ExitCode::from(1)
            }
        },
        Ok(None) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("nanovdb2pbrt: {}", error.message);
            if error.show_usage {
                eprintln!("\n{USAGE}");
            }
            ExitCode::from(1)
        }
    }
}
