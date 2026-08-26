use clap::*;

#[cfg(feature = "webgpu")]
use pbrt_r4::base::film::Film;
#[cfg(feature = "webgpu")]
use pbrt_r4::base::filter::Filter;
use pbrt_r4::displays::SequentialDisplay;
use pbrt_r4::displays::TevDisplay;
#[cfg(feature = "webgpu")]
use pbrt_r4::gpu::ir::{RenderConfig, RenderRequest};
#[cfg(feature = "webgpu")]
use pbrt_r4::gpu::webgpu::{PrepareOptions, Renderer};
#[cfg(feature = "webgpu")]
use pbrt_r4::parser::{parse_file, SceneBuilder};
use pbrt_r4::prelude::*;
use pbrt_r4::util::image::Image;
use pbrt_r4::util::imageio::{read_image, write_image};
use rayon::ThreadPoolBuilder;

use std::cell::RefCell;
use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::process;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread::available_parallelism;

use log::*;
/*
  --cropwindow <x0,x1,y0,y1> Specify an image crop window.
  --help               Print this help text.
  --nthreads <num>     Use specified number of threads for rendering.
  --outfile <filename> Write the final image to the given filename.
  --quick              Automatically reduce a number of quality settings to
                       render more quickly.
  --disable-pixel-jitter
                       Always sample pixels at their centers.
  --quiet              Suppress all text output other than error messages.
  --texture-cache <on|off>
                       Enable or disable r4's on-disk texture MIPMap cache.
  --texture-build <parallel|serial>
                       Control whether independent textures are realized in
                       parallel during scene construction.
  --scene-build <parallel|serial>
                       Control whether shapes and instances are realized in
                       parallel during scene construction.
  --scene-build-jobs <num>
                       Limit scene construction parallelism when
                       --scene-build parallel. Default: 4.

Logging options:
  --logdir <dir>       Specify directory that log files should be written to.
                       Default: system temp directory (e.g. $TMPDIR or /tmp).
  --logtostderr        Print all logging messages to stderr.
  --minloglevel <num>  Log messages at or above this level (0 -> INFO,
                       1 -> WARNING, 2 -> ERROR, 3-> FATAL). Default: 0.
  --v <verbosity>      Set VLOG verbosity.

Reformatting options:
  --cat                Print a reformatted version of the input file(s) to
                       standard output. Does not render an image.
  --toply              Print a reformatted version of the input file(s) to
                       standard output and convert all triangle meshes to
                       PLY files. Does not render an image.
*/

#[derive(Copy, Clone, Debug, ValueEnum)]
enum TextureCacheMode {
    On,
    Off,
}

impl TextureCacheMode {
    fn enabled(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum TextureBuildMode {
    Parallel,
    Serial,
}

impl TextureBuildMode {
    fn parallel(self) -> bool {
        matches!(self, Self::Parallel)
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum SceneBuildMode {
    Parallel,
    Serial,
}

impl SceneBuildMode {
    fn parallel(self) -> bool {
        matches!(self, Self::Parallel)
    }
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|e| format!("invalid positive integer: {e}"))?;
    if value == 0 {
        Err("value must be at least 1".to_string())
    } else {
        Ok(value)
    }
}

#[derive(Debug, Parser)]
#[clap(author, about, version, disable_help_flag = true)]
struct CommandOptions {
    /// Input .pbrt file.
    #[arg(short, long, value_name = "filename")]
    pub infile: Option<PathBuf>,

    /// Write the final image to the given filename.
    #[arg(short, long, value_name = "filename")]
    pub outfile: Option<PathBuf>,

    /// Specify an image crop window.
    #[arg(long, value_delimiter = ',', value_name = "x0,x1,y0,y1")]
    pub cropwindow: Option<Vec<f32>>,

    /// Print this help text.
    #[arg(short, long, action = clap::ArgAction::HelpLong)]
    pub help: Option<bool>,

    /// Use specified number of threads for rendering.
    #[arg(short = 'j', long = "nthreads", value_name = "num")]
    pub nthreads: Option<i32>,

    /// Automatically reduce a number of quality settings to render more quickly.
    #[arg(long, default_value = "false")]
    pub quick: bool,

    /// Render through the WebGPU wavefront backend.
    #[arg(long, default_value = "false")]
    pub gpu: bool,

    /// Always sample pixels at their centers.
    #[arg(long = "disable-pixel-jitter", default_value = "false")]
    pub disable_pixel_jitter: bool,

    /// Suppress all text output other than error messages.
    #[clap(long, default_value = "false")]
    pub quiet: bool,

    /// Enable or disable r4's on-disk texture MIPMap cache.
    #[arg(long = "texture-cache", value_enum, default_value_t = TextureCacheMode::On)]
    pub texture_cache: TextureCacheMode,

    /// Control whether independent textures are realized in parallel during scene construction.
    #[arg(long = "texture-build", value_enum, default_value_t = TextureBuildMode::Parallel)]
    pub texture_build: TextureBuildMode,

    /// Control whether shapes and instances are realized in parallel during scene construction.
    #[arg(long = "scene-build", value_enum, default_value_t = SceneBuildMode::Parallel)]
    pub scene_build: SceneBuildMode,

    /// Limit scene construction parallelism when --scene-build parallel.
    #[arg(
        long = "scene-build-jobs",
        value_name = "num",
        default_value_t = DEFAULT_SCENE_BUILD_JOBS,
        value_parser = parse_positive_usize
    )]
    pub scene_build_jobs: usize,

    /// Scale target triangle edge length for displacement mapping.
    #[arg(
        long = "displacement-edge-scale",
        value_name = "s",
        default_value_t = 1.0
    )]
    pub displacement_edge_scale: Float,

    // Logging options
    /// Specify directory that log files should be written to.
    /// Default: system temp directory (e.g. $TMPDIR or /tmp).
    #[arg(long, value_name = "dir")]
    pub logdir: Option<PathBuf>,

    /// Print all logging messages to stderr.
    #[arg(long, default_value = "false")]
    pub logtostderr: bool,

    /// Log messages at or above this level (0 -> INFO,
    /// 1 -> WARNING, 2 -> ERROR, 3-> FATAL).
    #[arg(long, value_name = "num")] //value_enum
    pub minloglevel: Option<i32>,

    /// Set VLOG verbosity.
    //#[arg(long = "v", default_value = "0", value_name = "verbosity")]

    // Reformatting options
    /// Print a reformatted version of the input file(s) to standard output.
    /// Does not render an image.
    #[arg(short, long, default_value = "false")]
    pub cat: bool,

    /// Print a reformatted version of the input file(s) to standard output and convert all triangle meshes to PLY files.
    /// Does not render an image.
    #[arg(short, long, default_value = "false")]
    pub toply: bool,

    /// Upgrade legacy pbrt-v3 input to pbrt-v4 format and write it without rendering.
    #[arg(long, default_value = "false")]
    pub upgrade: bool,

    /// Display-server ex. localhost:14158
    #[arg(long = "display-server", value_name = "url")]
    pub display_server: Option<String>,

    /// Set Pixelsamples.
    #[arg(short = 's', long = "pixelsamples", value_name = "num")]
    pub pixelsamples: Option<i32>,

    /// Quick full resolution.
    #[arg(long, default_value = "false")]
    pub quick_full_resolution: bool,

    /// Sequential display.
    #[arg(short = 'k', long = "sequential-display", value_name = "dir")]
    pub sequential_display: Option<PathBuf>,

    #[arg(value_name = "filename.pbrt")]
    pub pbrtfile: Option<Vec<PathBuf>>,
}

fn init_logger(opts: &CommandOptions) {
    if let Some(minloglevel) = opts.minloglevel {
        const LOG_LEVELS: &[&str] = &["trace", "debug", "info", "warn", "error"];
        let log_level = LOG_LEVELS[(minloglevel + 2).clamp(0, 4) as usize];
        env::set_var("RUST_LOG", log_level);
    } else {
        //default log level : warn
        let log_level = env::var("RUST_LOG").unwrap_or_else(|_| "warn".to_owned());
        env::set_var("RUST_LOG", log_level);
    }

    env_logger::Builder::from_default_env()
        //.format_timestamp(None)
        .format_target(false)
        .format_module_path(false)
        .init();
}

fn configure_render_threads(opts: &CommandOptions) {
    if let Some(nthreads) = opts.nthreads {
        if nthreads <= 0 {
            warn!("Ignoring invalid --nthreads value: {}", nthreads);
            return;
        }
        match ThreadPoolBuilder::new()
            .num_threads(nthreads as usize)
            .build_global()
        {
            Ok(_) => {}
            Err(e) => {
                warn!("Failed to apply --nthreads={}: {}", nthreads, e);
            }
        }
    }
}

fn create_print_target(outfile: Option<&PathBuf>) -> Result<PrintTarget, PbrtError> {
    if let Some(path) = outfile {
        match std::fs::File::create(path) {
            Ok(writer) => {
                return Ok(PrintTarget::new_with_params(
                    Arc::new(RefCell::new(writer)),
                    false,
                ));
            }
            Err(e) => {
                return Err(PbrtError::from(e));
            }
        }
    } else {
        Ok(PrintTarget::new_stdout(false))
    }
}

fn path_to_string(path: &Path) -> Result<String, PbrtError> {
    path.to_str()
        .map(|value| value.to_string())
        .ok_or_else(|| PbrtError::error("path is not valid UTF-8"))
}

fn cat_scene(input_path: &Path, opts: &CommandOptions) -> i32 {
    let outfile = opts.outfile.as_ref();
    let mut context = match create_print_target(outfile) {
        Ok(ctx) => ctx,
        Err(e) => {
            error!("{}", e);
            return -1;
        }
    };
    let input_path = match path_to_string(input_path) {
        Ok(path) => path,
        Err(e) => {
            error!("{}", e);
            return -1;
        }
    };
    match parse_file_without_include(&input_path, &mut context) {
        Ok(_) => {
            return 0;
        }
        Err(e) => {
            error!("{}", e);
            return -1;
        }
    }
}

fn upgrade_scene(input_path: &Path, opts: &CommandOptions) -> i32 {
    let outfile = opts.outfile.as_ref();
    let mut context = match create_print_target(outfile) {
        Ok(ctx) => ctx,
        Err(e) => {
            error!("{}", e);
            return -1;
        }
    };
    let input_path = match path_to_string(input_path) {
        Ok(path) => path,
        Err(e) => {
            error!("{}", e);
            return -1;
        }
    };
    match parse_file_upgraded(&input_path, &mut context) {
        Ok(_) => 0,
        Err(e) => {
            error!("{}", e);
            -1
        }
    }
}

fn toply_scene(input_path: &Path, opts: &CommandOptions) -> i32 {
    let dir = match input_path.parent() {
        Some(dir) => match path_to_string(dir) {
            Ok(path) => path,
            Err(e) => {
                error!("{}", e);
                return -1;
            }
        },
        None => {
            error!("input path has no parent directory");
            return -1;
        }
    };
    let mut dir = dir;
    let outfile = opts.outfile.as_ref();
    if let Some(outfile) = outfile {
        dir = match outfile.parent() {
            Some(parent) => match path_to_string(parent) {
                Ok(path) => path,
                Err(e) => {
                    error!("{}", e);
                    return -1;
                }
            },
            None => {
                error!("output path has no parent directory");
                return -1;
            }
        };
    }
    let context = match create_print_target(outfile) {
        Ok(ctx) => ctx,
        Err(e) => {
            error!("{}", e);
            return -1;
        }
    };
    let mut context = ToPlyTarget::new(&dir, Arc::new(RefCell::new(context)));
    let input_path = match path_to_string(input_path) {
        Ok(path) => path,
        Err(e) => {
            error!("{}", e);
            return -1;
        }
    };
    match parse_file_without_include(&input_path, &mut context) {
        Ok(_) => {
            return 0;
        }
        Err(e) => {
            error!("{}", e);
            return -1;
        }
    }
}

fn create_integrator(
    input_path: &Path,
    opts: &CommandOptions,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    {
        let mut builder = SceneBuilder::new();
        let path = path_to_string(input_path)?;
        parse_file(&path, &mut builder)?;

        info!(
            "SceneBuilder parse done: shapes={} animated_shapes={} \
             materials={} named_materials={} lights={} area_lights={} \
             media={} float_textures={} spectrum_textures={} \
             instance_defs={} instance_uses={}",
            builder.shapes.len(),
            builder.animated_shapes.len(),
            builder.materials.len(),
            builder.named_materials.len(),
            builder.lights.len(),
            builder.area_lights.len(),
            builder.media.len(),
            builder.float_textures.len(),
            builder.spectrum_textures.len(),
            builder.instance_definitions.len(),
            builder.instance_uses.len(),
        );

        // Apply CLI overrides (pixelsamples / outfile / cropwindow) directly
        // to the SceneBuilder param dictionaries before realising the scene.
        if let Some(pixelsamples) = opts.pixelsamples {
            let pixelsamples = i32::max(1, pixelsamples);
            builder
                .sampler_params
                .replace_one_int("integer pixelsamples", pixelsamples);
        }
        if let Some(outfile) = opts.outfile.as_ref() {
            let outfile = path_to_string(outfile.as_path())?;
            builder
                .film_params
                .replace_one_string("string filename", &outfile);
        }
        if let Some(cropwindow) = opts.cropwindow.as_ref() {
            if cropwindow.len() != 4 {
                return Err(PbrtError::error(
                    "\"--cropwindow\" expects four comma-separated values.",
                ));
            }
            let values: Vec<Float> = cropwindow.iter().map(|v| *v as Float).collect();
            builder.film_params.replace_floats("cropwindow", &values);
        }

        return builder.build();
    }
}

fn create_display(hostname: &str) -> Result<Arc<RwLock<dyn Display>>, PbrtError> {
    let mut tev = TevDisplay::new();
    tev.connect(hostname)?;
    Ok(Arc::new(RwLock::new(tev)))
}

fn render_scene(input_path: &Path, opts: &CommandOptions) -> i32 {
    #[cfg(feature = "webgpu")]
    if opts.gpu {
        return render_gpu_scene(input_path, opts);
    }
    #[cfg(not(feature = "webgpu"))]
    if opts.gpu {
        error!("--gpu requires building pbrt-r4 with --features webgpu");
        return -1;
    }
    if !opts.quiet {
        let nthreads = match available_parallelism() {
            Ok(value) => value.get(),
            Err(e) => {
                error!("Failed to detect available parallelism: {}", e);
                return -1;
            }
        };
        let version = env!("CARGO_PKG_VERSION");
        println!("pbrt-r4 version {} [Detected {} cores]", version, nthreads);
        println!();
        println!("This is an unofficial Rust port of pbrt-v4, evolved from a pbrt-r3 (pbrt-v3 Rust port) foundation.");
        println!("The source code of this port is distributed under the Apache 2.0 License (see LICENSE).");
        println!();
        println!("The license for the original implementation pbrt-v4 is as follows:");
        println!("--------------------------------------------------------------------------------------");
        println!("Copyright (c)1998-2021 Matt Pharr, Wenzel Jakob, and Greg Humphreys.");
        println!(
            "The source code to pbrt (but *not* the book contents) is covered by the Apache 2.0 License."
        );
        println!("See the file LICENSE.txt for the conditions of the license.");
        println!("--------------------------------------------------------------------------------------");
        println!();
    }

    let r = create_integrator(input_path, opts);
    match r {
        Ok(integrator) => {
            if let Some(hostname) = opts.display_server.as_ref() {
                match create_display(hostname) {
                    Ok(display) => {
                        let integrator = integrator.as_ref().read().unwrap();
                        let camera = integrator.get_camera();
                        let film = camera.as_ref().get_film();
                        let f = film.as_ref().read().unwrap();
                        f.add_display(&display);
                    }
                    Err(e) => {
                        warn!("{}", e);
                    }
                }
            }
            if let Some(out_dir) = opts.sequential_display.as_ref() {
                if let Err(e) = std::fs::create_dir_all(out_dir) {
                    error!("Failed to create sequential display directory: {}", e);
                    return -1;
                }
                let display: Arc<RwLock<dyn Display>> = Arc::new(RwLock::new(
                    SequentialDisplay::new(&out_dir.to_string_lossy()),
                ));
                let integrator = integrator.as_ref().read().unwrap();
                let camera = integrator.get_camera();
                let film = camera.as_ref().get_film();
                let f = film.as_ref().read().unwrap();
                f.add_display(&display);
            }

            {
                let mut integrator = integrator.as_ref().write().unwrap();
                integrator.render();
            }

            let options = PbrtOptions::get();
            if options.mse_reference_image.is_some() || options.mse_reference_output.is_some() {
                let (reference_path, output_path) = match (
                    options.mse_reference_image.as_deref(),
                    options.mse_reference_output.as_deref(),
                ) {
                    (Some(reference), Some(output)) => (reference, output),
                    _ => {
                        error!("MSE comparison requires both reference image and output paths.");
                        return -1;
                    }
                };
                let rendered = {
                    let integrator = integrator.as_ref().read().unwrap();
                    let camera = integrator.get_camera();
                    let film = camera.as_ref().get_film();
                    let film = film.as_ref().read().unwrap();
                    match film.to_image() {
                        Ok(image) => image,
                        Err(error) => {
                            error!("MSE comparison failed: {}", error);
                            return -1;
                        }
                    }
                };
                let (reference_texels, reference_resolution) = match read_image(reference_path) {
                    Ok(image) => image,
                    Err(error) => {
                        error!("Failed to read MSE reference image: {}", error);
                        return -1;
                    }
                };
                let reference = Image::new(reference_resolution, reference_texels);
                if rendered.resolution() != reference.resolution() {
                    error!(
                        "MSE reference resolution {:?} does not match rendered resolution {:?}.",
                        reference.resolution(),
                        rendered.resolution()
                    );
                    return -1;
                }
                let mse = rendered.mse(&reference);
                let values = mse.to_rgb();
                println!(
                    "MSE vs {}: R={} G={} B={} average={}",
                    reference_path,
                    values[0],
                    values[1],
                    values[2],
                    mse.average()
                );
                let mut error_pixels = Vec::with_capacity(rendered.texels().len() * 3);
                for (a, b) in rendered.texels().iter().zip(reference.texels()) {
                    let ar = a.to_rgb();
                    let br = b.to_rgb();
                    for c in 0..3 {
                        let delta = ar[c] - br[c];
                        error_pixels.push(delta * delta);
                    }
                }
                let bounds =
                    Bounds2i::from(((0, 0), (reference_resolution.x, reference_resolution.y)));
                if let Err(error) =
                    write_image(output_path, &error_pixels, &bounds, &reference_resolution)
                {
                    error!("Failed to write MSE image: {}", error);
                    return -1;
                }
            }
        }
        Err(e) => {
            let msg = format!("{}", e);
            error!("{}", msg);
            return -1;
        }
    }
    println!("\n");

    return 0;
}

#[cfg(feature = "webgpu")]
fn render_gpu_scene(input_path: &Path, opts: &CommandOptions) -> i32 {
    let mut builder = SceneBuilder::new();
    let path = match path_to_string(input_path) {
        Ok(path) => path,
        Err(error) => {
            error!("{:?}", error);
            return -1;
        }
    };
    if let Err(error) = parse_file(&path, &mut builder) {
        error!("{}", error);
        return -1;
    }
    if let Some(pixelsamples) = opts.pixelsamples {
        builder
            .sampler_params
            .replace_one_int("integer pixelsamples", pixelsamples.max(1));
    }
    if let Some(outfile) = opts.outfile.as_ref() {
        match path_to_string(outfile) {
            Ok(path) => builder
                .film_params
                .replace_one_string("string filename", &path),
            Err(error) => {
                error!("{}", error);
                return -1;
            }
        }
    }
    let filter = match Filter::create(&builder.filter_name, &builder.filter_params) {
        Ok(filter) => filter,
        Err(error) => {
            error!("{}", error);
            return -1;
        }
    };
    let film = match Film::create(&builder.film_name, &builder.film_params, &filter) {
        Ok(film) => film,
        Err(error) => {
            error!("{}", error);
            return -1;
        }
    };
    let compiled = match builder.build_gpu_ir() {
        Ok(scene) => scene,
        Err(error) => {
            error!("{:?}", error);
            return -1;
        }
    };
    let mut renderer = match Renderer::new(&PrepareOptions::default()) {
        Ok(renderer) => renderer,
        Err(error) => {
            error!("WebGPU initialization failed: {}", error);
            return -1;
        }
    };
    let executable = match renderer.prepare(&compiled) {
        Ok(scene) => scene,
        Err(error) => {
            error!("WebGPU scene preparation failed: {}", error);
            return -1;
        }
    };
    let render = compiled.view().render;
    let mut render_config = RenderConfig::default();
    render_config.sampler.samples_per_pixel = render.sampler.samples_per_pixel;
    let request = match RenderRequest::new(&render_config, 0, render.sampler.samples_per_pixel) {
        Ok(request) => request,
        Err(error) => {
            error!("invalid GPU render request: {:?}", error);
            return -1;
        }
    };
    let mut film = film.write().unwrap();
    if let Err(error) = renderer.render_to_film(&executable, &request, &mut film) {
        error!("WebGPU rendering failed: {}", error);
        return -1;
    }
    0
}

pub fn main() {
    let mut opts = CommandOptions::parse();
    if opts.quick_full_resolution {
        opts.quick = true;
    }
    {
        let mut options = PbrtOptions::default();
        options.quick_render = opts.quick;
        options.quick_render_full_resolution = opts.quick_full_resolution;
        options.disable_pixel_jitter = opts.disable_pixel_jitter;
        options.texture_cache = opts.texture_cache.enabled();
        options.parallel_texture_build = opts.texture_build.parallel();
        options.parallel_scene_build = opts.scene_build.parallel();
        options.scene_build_jobs = opts.scene_build_jobs;
        options.displacement_edge_scale = opts.displacement_edge_scale;
        PbrtOptions::set(options);
    }
    init_logger(&opts);
    configure_render_threads(&opts);
    let input = if let Some(infiles) = opts.pbrtfile.as_ref() {
        Some(infiles[0].clone())
    } else {
        opts.infile.as_ref().cloned()
    };

    if let Some(ipath) = input.as_ref() {
        if !ipath.exists() {
            println!("{}", CommandOptions::command().render_usage());
            process::exit(-1);
        }
    } else {
        println!("{}", CommandOptions::command().render_usage());
        process::exit(-1);
    }

    let input_path = match input {
        Some(input_path) => match input_path.canonicalize() {
            Ok(path) => path,
            Err(e) => {
                error!("Failed to canonicalize input path: {}", e);
                process::exit(-1);
            }
        },
        None => {
            process::exit(-1);
        }
    };

    if opts.upgrade {
        let ret = upgrade_scene(&input_path, &opts);
        process::exit(ret);
    } else if opts.cat {
        let ret = cat_scene(&input_path, &opts);
        process::exit(ret);
    } else if opts.toply {
        let ret = toply_scene(&input_path, &opts);
        process::exit(ret);
    } else {
        let ret = render_scene(&input_path, &opts);
        process::exit(ret);
    }
}
