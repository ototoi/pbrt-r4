use std::cell::LazyCell;
use std::sync::LazyLock;
use std::sync::Mutex;

use crate::util::base::Float;

pub const DEFAULT_SCENE_BUILD_JOBS: usize = 4;

/// pbrt-v4 `enum class RenderingCoordinateSystem` (options.h:18).
/// Controls where the scene gets re-centered before rendering.
///
/// * `Camera` — render space == camera space. The camera sits at
///   origin and looks down `-z`; world geometry is transformed by
///   `cameraFromWorld_at_mid_time`.
/// * `CameraWorld` (v4 default) — render space == world translated so
///   that the camera position at mid shutter time lands at the origin.
///   World rotation is preserved. This keeps coordinates near the
///   camera numerically small without losing the global orientation,
///   which matters for very large scenes (pavilion-night etc.).
/// * `World` — render space == world space (no re-centering). This is
///   what pbrt-r4 has always done before; preserves global scene coordinates.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RenderingCoordinateSystem {
    Camera,
    CameraWorld,
    World,
}

impl Default for RenderingCoordinateSystem {
    fn default() -> Self {
        // pbrt-v4 (options.h:33) defaults to `CameraWorld`.
        RenderingCoordinateSystem::CameraWorld
    }
}

#[derive(Debug, Clone)]
pub struct PbrtOptions {
    pub quick_render: bool,
    pub quick_render_full_resolution: bool,
    pub disable_pixel_jitter: bool,
    pub disable_wavelength_jitter: bool,
    pub disable_texture_filtering: bool,
    pub force_diffuse: bool,
    pub rendering_space: RenderingCoordinateSystem,
    pub texture_cache: bool,
    pub parallel_texture_build: bool,
    pub parallel_scene_build: bool,
    pub scene_build_jobs: usize,
    pub displacement_edge_scale: Float,
    pub seed: u32,
    pub record_pixel_statistics: bool,
    pub mse_reference_image: Option<String>,
    pub mse_reference_output: Option<String>,
}

impl Default for PbrtOptions {
    fn default() -> Self {
        PbrtOptions {
            quick_render: false,
            quick_render_full_resolution: false,
            disable_pixel_jitter: false,
            disable_wavelength_jitter: false,
            disable_texture_filtering: false,
            force_diffuse: false,
            rendering_space: RenderingCoordinateSystem::default(),
            texture_cache: true,
            parallel_texture_build: true,
            parallel_scene_build: true,
            scene_build_jobs: DEFAULT_SCENE_BUILD_JOBS,
            displacement_edge_scale: 1.0,
            seed: 0,
            record_pixel_statistics: false,
            mse_reference_image: None,
            mse_reference_output: None,
            //crop_window: [0.0, 1.0, 0.0, 1.0],
        }
    }
}

static PBRT_OPTIONS: LazyLock<Mutex<PbrtOptions>> =
    LazyLock::new(|| Mutex::new(PbrtOptions::new()));

thread_local!(
    static LOCAL_OPTIONS: LazyCell<PbrtOptions> = LazyCell::new(|| PbrtOptions::get_lock());
);

impl PbrtOptions {
    pub fn new() -> Self {
        PbrtOptions::default()
    }

    pub fn set(opt: PbrtOptions) {
        let mut options = PBRT_OPTIONS.lock().unwrap();
        *options = opt;
    }

    pub fn get() -> PbrtOptions {
        LOCAL_OPTIONS.with(|opt| (**opt).clone())
    }

    pub fn get_lock() -> PbrtOptions {
        let options = PBRT_OPTIONS.lock().unwrap();
        return options.clone();
    }

    /// Apply a single `Option "<name>" <value>` directive to the global
    /// options. Mirrors pbrt-v4 `BasicSceneBuilder::Option` (scene.cpp).
    pub fn apply_option(name: &str, value: &str) -> Result<(), String> {
        // pbrt's `normalizeArg` lower-cases the option name.
        let n = name.to_lowercase();
        let mut options = PbrtOptions::get_lock();

        let parse_bool = |v: &str| match v.to_lowercase().as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        };

        let parse_quoted = |v: &str| {
            (v.len() >= 2 && v.starts_with('"') && v.ends_with('"'))
                .then(|| v[1..v.len() - 1].to_string())
        };

        match n.as_str() {
            "disablepixeljitter" => match parse_bool(value) {
                Some(b) => options.disable_pixel_jitter = b,
                None => {
                    return Err(format!(
                        "Option \"{}\": expected \"true\" or \"false\".",
                        name
                    ));
                }
            },
            "disablewavelengthjitter" => match parse_bool(value) {
                Some(b) => options.disable_wavelength_jitter = b,
                None => {
                    return Err(format!(
                        "Option \"{}\": expected \"true\" or \"false\".",
                        name
                    ));
                }
            },
            "disabletexturefiltering" => match parse_bool(value) {
                Some(b) => options.disable_texture_filtering = b,
                None => {
                    return Err(format!(
                        "Option \"{}\": expected \"true\" or \"false\".",
                        name
                    ));
                }
            },
            "forcediffuse" => match parse_bool(value) {
                Some(b) => options.force_diffuse = b,
                None => {
                    return Err(format!(
                        "Option \"{}\": expected \"true\" or \"false\".",
                        name
                    ));
                }
            },
            "displacementedgescale" => match value.parse::<Float>() {
                Ok(f) => options.displacement_edge_scale = f,
                Err(_) => {
                    return Err(format!(
                        "Option \"{}\": expected a floating-point value.",
                        name
                    ));
                }
            },
            "rendercoordsys" => match value {
                "camera" => options.rendering_space = RenderingCoordinateSystem::Camera,
                "cameraworld" => options.rendering_space = RenderingCoordinateSystem::CameraWorld,
                "world" => options.rendering_space = RenderingCoordinateSystem::World,
                _ => {
                    return Err(format!(
                        "Option \"rendercoordsys\": unknown system \"{}\".",
                        value
                    ));
                }
            },
            "seed" => match value.parse::<u32>() {
                Ok(seed) => options.seed = seed,
                Err(_) => {
                    return Err(format!(
                        "Option \"{}\": expected a non-negative integer.",
                        name
                    ));
                }
            },
            "msereferenceimage" => match parse_quoted(value) {
                Some(path) if !path.is_empty() => options.mse_reference_image = Some(path),
                _ => return Err(format!("Option \"{}\": expected a quoted filename.", name)),
            },
            "msereferenceout" => match parse_quoted(value) {
                Some(path) if !path.is_empty() => options.mse_reference_output = Some(path),
                _ => return Err(format!("Option \"{}\": expected a quoted filename.", name)),
            },
            "pixelstats" => match parse_bool(value) {
                Some(b) => options.record_pixel_statistics = b,
                None => {
                    return Err(format!(
                        "Option \"{}\": expected \"true\" or \"false\".",
                        name
                    ));
                }
            },
            _ => {
                return Err(format!(
                    "Option \"{}\" is unsupported in the CPU pbrt-r4 renderer.",
                    name
                ));
            }
        }

        PbrtOptions::set(options);
        Ok(())
    }
}
