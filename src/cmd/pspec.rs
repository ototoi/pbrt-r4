use pbrt_r4::samplers::{
    owen_scrambled_radical_inverse, owen_sobol_sample, IndependentSampler, PMJ02BNSampler,
    PaddedSobolSampler, RandomizeStrategy, StratifiedSampler, ZSobolSampler,
};
use pbrt_r4::util::base::{Float, Point2f, Point2i};
use pbrt_r4::util::geometry::Bounds2i;
use pbrt_r4::util::image::{Image, PixelFormat};
use pbrt_r4::util::lowdiscrepancy::{radical_inverse, scrambled_radical_inverse, DigitPermutation};
use pbrt_r4::util::rng::{mix_bits, RNG};
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Write};

const DEFAULT_NPOINTS: usize = 1024;
const DEFAULT_NSETS: usize = 4;
const DEFAULT_RESOLUTION: usize = 1500;

const USAGE: &str = "usage: pspec <sampler> [options]\n\n\
samplers: cwd.pts grid halton halton.owen halton.permutedigits independent lhs\n\
          pmj02bn sobol sobol.fastowen sobol.owen sobol.permutedigits sobol.z\n\
          stdin.binary stdin.dat stratified\n\n\
options:\n\
  --npoints <n>    Number of points per set (default: 1024)\n\
  --nsets <n>      Number of sets (default: 4)\n\
  --outbase <name> Output basename\n\
  --resolution <n> Power spectrum resolution (default: 1500)\n";

#[derive(Debug)]
struct CliError {
    message: String,
    usage: bool,
}

impl CliError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            usage: false,
        }
    }

    fn usage(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            usage: true,
        }
    }
}

#[derive(Debug)]
struct Args {
    sampler: String,
    npoints: usize,
    nsets: usize,
    outbase: Option<String>,
    resolution: usize,
}

fn parse_value<I>(args: &mut I, option: &str) -> Result<String, CliError>
where
    I: Iterator<Item = String>,
{
    args.next()
        .filter(|value| !value.starts_with('-'))
        .ok_or_else(|| CliError::usage(format!("missing value for {option}")))
}

fn parse_positive(value: &str, option: &str) -> Result<usize, CliError> {
    let value = value
        .parse::<usize>()
        .map_err(|_| CliError::usage(format!("invalid value for {option}: {value}")))?;
    if value == 0 {
        return Err(CliError::usage(format!(
            "{option} must be greater than zero"
        )));
    }
    Ok(value)
}

fn option_value<I>(inline: Option<&str>, args: &mut I, option: &str) -> Result<String, CliError>
where
    I: Iterator<Item = String>,
{
    match inline {
        Some(value) => Ok(value.to_owned()),
        None => parse_value(args, option),
    }
}

fn parse_args<I>(mut args: I) -> Result<Args, CliError>
where
    I: Iterator<Item = String>,
{
    let sampler = args
        .next()
        .ok_or_else(|| CliError::usage("must specify a sampler"))?;
    if sampler == "--help" || sampler == "-h" || sampler == "-help" {
        return Err(CliError::usage(String::new()));
    }

    let mut parsed = Args {
        sampler,
        npoints: DEFAULT_NPOINTS,
        nsets: DEFAULT_NSETS,
        outbase: None,
        resolution: DEFAULT_RESOLUTION,
    };
    while let Some(argument) = args.next() {
        let (option, inline) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(name, value)| {
                (name, Some(value))
            });
        match option {
            "--npoints" => {
                let value = option_value(inline, &mut args, "--npoints")?;
                parsed.npoints = parse_positive(&value, "--npoints")?;
            }
            "--nsets" => {
                let value = option_value(inline, &mut args, "--nsets")?;
                parsed.nsets = parse_positive(&value, "--nsets")?;
            }
            "--resolution" => {
                let value = option_value(inline, &mut args, "--resolution")?;
                parsed.resolution = parse_positive(&value, "--resolution")?;
            }
            "--outbase" => {
                parsed.outbase = Some(option_value(inline, &mut args, "--outbase")?);
            }
            "--help" | "-h" | "-help" => return Err(CliError::usage(String::new())),
            _ if argument.starts_with('-') => {
                return Err(CliError::usage(format!("unknown option: {argument}")))
            }
            _ => return Err(CliError::usage(format!("unknown argument: {argument}"))),
        }
    }
    Ok(parsed)
}

fn seed_for_set(set: usize) -> u32 {
    mix_bits(set as u64) as u32
}

fn rng_for_set(set: usize) -> RNG {
    let mut rng = RNG::new();
    rng.set_sequence_with_seed(seed_for_set(set) as u64, set as u64);
    rng
}

enum GeneratedSampler {
    Independent(IndependentSampler),
    Stratified(StratifiedSampler),
    PaddedSobol(PaddedSobolSampler),
    Pmj(PMJ02BNSampler),
}

impl GeneratedSampler {
    fn sample(&mut self, index: usize) -> Point2f {
        match self {
            Self::Independent(s) => {
                s.start_pixel_sample(index as u32, 0);
                s.get_2d()
            }
            Self::Stratified(s) => s.get_2d(),
            Self::PaddedSobol(s) => {
                s.start_pixel_sample(index as u32, 0);
                s.get_2d()
            }
            Self::Pmj(s) => {
                s.start_pixel_sample(index as u32, 0);
                s.get_2d()
            }
        }
    }
}

fn generated_samples(name: &str, npoints: usize, set: usize) -> Result<Vec<Point2f>, CliError> {
    let seed = seed_for_set(set);
    let mut samples = Vec::with_capacity(npoints);
    match name {
        "grid" => {
            let side = (npoints as f64).sqrt() as usize;
            for y in 0..side {
                for x in 0..side {
                    samples.push(Point2f::new(
                        x as Float / side as Float,
                        y as Float / side as Float,
                    ));
                }
            }
        }
        "lhs" => {
            let mut rng = rng_for_set(set);
            for i in 0..npoints {
                samples.push(Point2f::new(
                    (i as Float + rng.uniform_float()) / npoints as Float,
                    (i as Float + rng.uniform_float()) / npoints as Float,
                ));
            }
            for i in 0..npoints {
                let other = i + rng.uniform_uint32_threshold((npoints - i) as u32) as usize;
                samples.swap(i, other);
            }
        }
        "halton" => {
            for i in 0..npoints {
                samples.push(Point2f::new(
                    radical_inverse(0, i as u64),
                    radical_inverse(1, i as u64),
                ));
            }
        }
        "halton.permutedigits" => {
            let mut rng = rng_for_set(set);
            let perm2 = DigitPermutation::new(2, rng.uniform_uint32());
            let perm3 = DigitPermutation::new(3, rng.uniform_uint32());
            for i in 0..npoints {
                samples.push(Point2f::new(
                    scrambled_radical_inverse(0, i as u64, &perm2),
                    scrambled_radical_inverse(1, i as u64, &perm3),
                ));
            }
        }
        "independent" => {
            let mut sampler =
                GeneratedSampler::Independent(IndependentSampler::new(npoints as u32, seed));
            for i in 0..npoints {
                samples.push(sampler.sample(i));
            }
        }
        "stratified" => {
            let side = (npoints as f64).sqrt() as u32;
            let effective = side * side;
            let mut stratified = StratifiedSampler::new(side, side, true, seed, 2);
            stratified.start_pixel(&Point2i::zero());
            let mut sampler = GeneratedSampler::Stratified(stratified);
            for i in 0..effective as usize {
                samples.push(sampler.sample(i));
            }
        }
        "pmj02bn" => {
            let mut sampler =
                GeneratedSampler::Pmj(PMJ02BNSampler::new(npoints as u32, seed as i32));
            for i in 0..npoints {
                samples.push(sampler.sample(i));
            }
        }
        "sobol" | "sobol.fastowen" | "sobol.owen" | "sobol.permutedigits" => {
            let randomize = match name {
                "sobol" => RandomizeStrategy::None,
                "sobol.fastowen" => RandomizeStrategy::FastOwen,
                "sobol.owen" => RandomizeStrategy::Owen,
                _ => RandomizeStrategy::PermuteDigits,
            };
            let mut sampler = GeneratedSampler::PaddedSobol(PaddedSobolSampler::new(
                npoints as u32,
                randomize,
                seed,
            ));
            for i in 0..npoints {
                samples.push(sampler.sample(i));
            }
        }
        "sobol.z" => {
            if !npoints.is_power_of_two() {
                return Err(CliError::new("Must use power of 2 points for \"sobol.z\"."));
            }
            let side = 1usize << (npoints.trailing_zeros() / 2);
            let spp = npoints / (side * side);
            let mut sampler = ZSobolSampler::new(
                spp as u32,
                Point2i::new(side as i32, side as i32),
                RandomizeStrategy::Owen,
                seed,
            );
            sampler.start_pixel(&Point2i::zero());
            for y in 0..side {
                for x in 0..side {
                    sampler.start_pixel(&Point2i::new(x as i32, y as i32));
                    for sample in 0..spp {
                        sampler.start_pixel_sample(sample as u32, 0);
                        let u = sampler.get_2d();
                        samples.push(Point2f::new(
                            (x as Float + u.x) / side as Float,
                            (y as Float + u.y) / side as Float,
                        ));
                    }
                }
            }
        }
        "halton.owen" => {
            let mut rng = rng_for_set(set);
            let scramble_x = rng.uniform_uint32();
            let scramble_y = rng.uniform_uint32();
            for i in 0..npoints {
                samples.push(Point2f::new(
                    owen_sobol_sample(i as i64, 0, scramble_x),
                    owen_scrambled_radical_inverse(1, i as u64, scramble_y),
                ));
            }
        }
        _ => return Err(CliError::usage(format!("sampler unknown: {name}"))),
    }
    Ok(samples)
}

fn read_binary_set<R: Read>(
    reader: &mut R,
    npoints: usize,
) -> Result<Option<Vec<Point2f>>, CliError> {
    let mut points = Vec::with_capacity(npoints);
    for index in 0..npoints {
        let mut bytes = [0u8; 8];
        match reader.read_exact(&mut bytes) {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
                if index > 1 {
                    return Err(CliError::new(format!(
                        "Partial point set provided in standard input: have {index} points at EOF."
                    )));
                }
                return Ok(None);
            }
            Err(error) => return Err(CliError::new(format!("stdin.binary: {error}"))),
        }
        points.push(Point2f::new(
            f32::from_le_bytes(bytes[0..4].try_into().unwrap()) as Float,
            f32::from_le_bytes(bytes[4..8].try_into().unwrap()) as Float,
        ));
    }
    Ok(Some(points))
}

fn read_text_sets(
    input: &str,
    npoints: usize,
    nsets: usize,
) -> Result<Vec<Vec<Point2f>>, CliError> {
    let mut tokens = input.split_whitespace();
    let mut sets = Vec::new();
    while sets.len() < nsets {
        let mut points = Vec::with_capacity(npoints);
        while points.len() < npoints {
            let Some(x) = tokens.next() else {
                if points.len() > 1 {
                    return Err(CliError::new(format!(
                        "Partial point set provided in standard input: have {} points at EOF.",
                        points.len()
                    )));
                }
                return Ok(sets);
            };
            if x == "#" {
                if points.len() > 1 {
                    return Err(CliError::new("partial point set ended by '#'"));
                }
                return Ok(sets);
            }
            let Some(y) = tokens.next() else {
                return Err(CliError::new("stdin.dat: expected two values per point"));
            };
            if y == "#" {
                return Err(CliError::new("stdin.dat: expected two values per point"));
            }
            let x = x
                .parse::<Float>()
                .map_err(|_| CliError::new(format!("stdin.dat: invalid x value {x}")))?;
            let y = y
                .parse::<Float>()
                .map_err(|_| CliError::new(format!("stdin.dat: invalid y value {y}")))?;
            points.push(Point2f::new(x, y));
        }
        sets.push(points);
        while let Some(token) = tokens.next() {
            if token == "#" {
                break;
            }
        }
    }
    Ok(sets)
}

fn read_pts_files(npoints: usize, max_sets: usize) -> Result<Vec<Vec<Point2f>>, CliError> {
    let mut files = Vec::new();
    for entry in fs::read_dir(".").map_err(|error| CliError::new(format!("cwd.pts: {error}")))? {
        let entry = entry.map_err(|error| CliError::new(format!("cwd.pts: {error}")))?;
        if entry
            .file_type()
            .map_err(|error| CliError::new(error.to_string()))?
            .is_file()
            && entry.file_name().to_string_lossy().starts_with("pts-")
        {
            files.push(entry.path());
        }
    }
    if files.is_empty() {
        return Err(CliError::new("No *.dat files found in current directory."));
    }
    let mut sets = Vec::new();
    for path in files {
        let input = fs::read_to_string(&path)
            .map_err(|error| CliError::new(format!("{}: {error}", path.display())))?;
        let mut points = Vec::with_capacity(npoints);
        let mut values = input.split_whitespace();
        while points.len() < npoints {
            let Some(x) = values.next() else { break };
            let Some(y) = values.next() else { break };
            points.push(Point2f::new(
                x.parse()
                    .map_err(|_| CliError::new(format!("{}: invalid x", path.display())))?,
                y.parse()
                    .map_err(|_| CliError::new(format!("{}: invalid y", path.display())))?,
            ));
        }
        if points.len() < npoints {
            eprintln!("warning: {}: premature EOF; ignoring file", path.display());
            continue;
        }
        sets.push(points);
        if sets.len() == max_sets {
            break;
        }
    }
    Ok(sets)
}

fn power_spectrum(points: &[Point2f], resolution: usize, output: &mut [Float]) {
    let center = (resolution / 2) as Float;
    for y in 0..resolution {
        for x in 0..resolution {
            let wx = x as Float - center;
            let wy = y as Float - center;
            let mut real = 0.0 as Float;
            let mut imag = 0.0 as Float;
            for point in points {
                let phase = -(std::f64::consts::TAU as Float) * (wx * point.x + wy * point.y);
                real += phase.cos();
                imag += phase.sin();
            }
            output[y * resolution + x] += real * real + imag * imag;
        }
    }
}

fn write_outputs(
    base: &str,
    resolution: usize,
    power: &[Float],
    npoints: usize,
    nsets: usize,
) -> Result<(), CliError> {
    let scale = 1.0 as Float / (npoints * nsets) as Float;
    let pixels = power.iter().map(|value| *value * scale).collect::<Vec<_>>();
    let resolution_i = Point2i::new(resolution as i32, resolution as i32);
    let image = Image::from_channels_with_format(
        resolution_i,
        vec!["power".to_owned()],
        pixels.clone(),
        PixelFormat::Float,
    );
    let bounds = Bounds2i::new(&Point2i::zero(), &resolution_i);
    image
        .write_exr(&format!("{base}.exr"), &bounds, &resolution_i)
        .map_err(|error| CliError::new(error.to_string()))?;

    let mut sum = vec![0.0 as Float; resolution / 2];
    let mut count = vec![0usize; resolution / 2];
    let center = resolution / 2;
    for y in 0..resolution {
        for x in 0..resolution {
            if x == center && y == center {
                continue;
            }
            let dx = x.abs_diff(center);
            let dy = y.abs_diff(center);
            let bucket = (((dx * dx + dy * dy) as f64).sqrt()) as usize;
            if bucket < sum.len() {
                sum[bucket] += pixels[y * resolution + x];
                count[bucket] += 1;
            }
        }
    }
    let mut file = File::create(format!("{base}.txt"))
        .map_err(|error| CliError::new(format!("{base}.txt: {error}")))?;
    for bucket in 1..resolution / 2 {
        if count[bucket] == 0 {
            return Err(CliError::new(format!("empty radial bucket {bucket}")));
        }
        writeln!(file, "{bucket} {}", sum[bucket] / count[bucket] as Float)
            .map_err(|error| CliError::new(format!("{base}.txt: {error}")))?;
    }
    Ok(())
}

fn run(args: Args) -> Result<(), CliError> {
    let resolution = if args.resolution % 2 == 0 {
        args.resolution + 1
    } else {
        args.resolution
    };
    let mut power = vec![0.0 as Float; resolution * resolution];
    let mut actual_sets = 0usize;
    if args.sampler == "cwd.pts" {
        for points in read_pts_files(args.npoints, args.nsets)? {
            power_spectrum(&points, resolution, &mut power);
            actual_sets += 1;
        }
    } else if args.sampler == "stdin.binary" {
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        for _ in 0..args.nsets {
            let Some(points) = read_binary_set(&mut reader, args.npoints)? else {
                break;
            };
            power_spectrum(&points, resolution, &mut power);
            actual_sets += 1;
        }
    } else if args.sampler == "stdin.dat" {
        let stdin = io::stdin();
        let mut input = String::new();
        stdin
            .lock()
            .read_to_string(&mut input)
            .map_err(|error| CliError::new(format!("stdin.dat: {error}")))?;
        for points in read_text_sets(&input, args.npoints, args.nsets)? {
            power_spectrum(&points, resolution, &mut power);
            actual_sets += 1;
        }
    } else {
        for set in 0..args.nsets {
            let points = generated_samples(&args.sampler, args.npoints, set)?;
            power_spectrum(&points, resolution, &mut power);
            actual_sets += 1;
        }
    }
    if actual_sets == 0 {
        return Err(CliError::new("no point sets were provided"));
    }
    let base = args
        .outbase
        .unwrap_or_else(|| format!("{}-{}pts-{}sets", args.sampler, args.npoints, actual_sets));
    write_outputs(&base, resolution, &power, args.npoints, actual_sets)
}

fn main() {
    match parse_args(std::env::args().skip(1)).and_then(run) {
        Ok(()) => {}
        Err(error) => {
            if !error.message.is_empty() {
                eprintln!("pspec: {}", error.message);
            }
            if error.usage {
                eprintln!("{USAGE}");
            }
            std::process::exit(1);
        }
    }
}
