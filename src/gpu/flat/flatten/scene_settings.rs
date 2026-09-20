use crate::gpu::flat::{RenderSettings, SamplerKind, SamplerRandomization, MAX_GPU_RENDER_DEPTH};
use crate::gpu::node::{Integrator as NodeIntegrator, Sampler as NodeSampler};
use crate::options::PbrtOptions;
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;

pub fn register_root_component<T>(
    destination: &mut Option<T>,
    value: T,
    depth: usize,
    kind: &str,
) -> Result<(), PbrtError> {
    if depth != 1 {
        return Err(PbrtError::error(&format!(
            "{kind} component must be attached to the GPU root node."
        )));
    }
    if destination.replace(value).is_some() {
        return Err(PbrtError::error(&format!(
            "Multiple {kind} components were found while flattening GPU Node IR."
        )));
    }
    Ok(())
}

pub fn render_settings(
    sampler: &Option<NodeSampler>,
    integrator: &Option<NodeIntegrator>,
) -> Result<RenderSettings, PbrtError> {
    let sampler = sampler.as_ref();
    let integrator = integrator.as_ref();
    let sampler_kind = match sampler.map(|sampler| sampler.name.as_str()) {
        None | Some("independent") => SamplerKind::Independent,
        Some("halton") => SamplerKind::Halton,
        Some("sobol") => SamplerKind::Sobol,
        Some("paddedsobol") => SamplerKind::PaddedSobol,
        Some("zsobol") => SamplerKind::ZSobol,
        Some("pmj02bn") => SamplerKind::Pmj02Bn,
        Some("stratified") => SamplerKind::Stratified,
        Some(name) => {
            return Err(PbrtError::error(&format!(
                "GPU sampler '{name}' is not implemented."
            )))
        }
    };
    let randomization = match sampler_kind {
        SamplerKind::Halton => parse_randomization(sampler, "permutedigits", false)?,
        SamplerKind::Sobol | SamplerKind::PaddedSobol | SamplerKind::ZSobol => {
            parse_randomization(sampler, "fastowen", true)?
        }
        _ => SamplerRandomization::None,
    };
    if let Some(integrator) = integrator {
        if integrator.name != "path" && integrator.name != "volpath" {
            return Err(PbrtError::error(&format!(
                "Unsupported GPU integrator: {}.",
                integrator.name
            )));
        }
    }
    let default_samples_per_pixel = match sampler_kind {
        SamplerKind::Independent => 4,
        SamplerKind::Halton
        | SamplerKind::Sobol
        | SamplerKind::PaddedSobol
        | SamplerKind::ZSobol
        | SamplerKind::Pmj02Bn => 16,
        SamplerKind::Stratified => 16,
    };
    let configured_x_samples = sampler
        .filter(|_| sampler_kind == SamplerKind::Stratified)
        .map(|sampler| sampler.params.get_one_int("xsamples", 4))
        .unwrap_or(1);
    let configured_y_samples = sampler
        .filter(|_| sampler_kind == SamplerKind::Stratified)
        .map(|sampler| sampler.params.get_one_int("ysamples", 4))
        .unwrap_or(1);
    let configured_samples_per_pixel = if sampler_kind == SamplerKind::Stratified {
        configured_x_samples
            .checked_mul(configured_y_samples)
            .ok_or_else(|| PbrtError::error("GPU stratified samples per pixel overflowed i32."))?
    } else {
        sampler
            .map(|sampler| {
                sampler
                    .params
                    .get_one_int("pixelsamples", default_samples_per_pixel)
            })
            .unwrap_or(default_samples_per_pixel)
    };
    let samples_per_pixel = if PbrtOptions::get().quick_render {
        1
    } else {
        configured_samples_per_pixel
    };
    let (x_samples, y_samples) =
        if PbrtOptions::get().quick_render && sampler_kind == SamplerKind::Stratified {
            (1, 1)
        } else {
            (configured_x_samples, configured_y_samples)
        };
    let configured_max_depth = integrator
        .map(|integrator| integrator.params.get_one_int("maxdepth", 5))
        .unwrap_or(5);
    let max_gpu_render_depth = MAX_GPU_RENDER_DEPTH as i32;
    let max_depth = configured_max_depth.min(max_gpu_render_depth);
    if configured_max_depth > max_gpu_render_depth {
        log::warn!(
            "GPU maxdepth {} exceeds the backend limit {}; clamping to {}.",
            configured_max_depth,
            max_gpu_render_depth,
            max_gpu_render_depth
        );
    }
    let seed = sampler
        .map(|sampler| {
            sampler
                .params
                .get_one_int("seed", PbrtOptions::get().seed as i32)
        })
        .unwrap_or(PbrtOptions::get().seed as i32);
    let light_sampler = integrator
        .map(|integrator| integrator.params.get_one_string("lightsampler", "bvh"))
        .unwrap_or_else(|| "bvh".to_string());
    let jitter = sampler
        .filter(|_| sampler_kind == SamplerKind::Stratified)
        .map(|sampler| sampler.params.get_one_bool("jitter", true))
        .unwrap_or(false);
    if samples_per_pixel <= 0 || x_samples <= 0 || y_samples <= 0 || max_depth < 0 || seed < 0 {
        return Err(PbrtError::error(
            "GPU render settings must have positive samples and non-negative depth/seed.",
        ));
    }
    let samples_per_pixel = u32::try_from(samples_per_pixel)
        .map_err(|_| PbrtError::error("GPU samples per pixel do not fit in u32."))?;
    let samples_per_pixel = if sampler_kind == SamplerKind::ZSobol {
        1u32 << samples_per_pixel.ilog2()
    } else {
        samples_per_pixel
    };
    Ok(RenderSettings {
        sampler_kind,
        randomization,
        samples_per_pixel,
        x_samples: u32::try_from(x_samples)
            .map_err(|_| PbrtError::error("GPU stratified x samples do not fit in u32."))?,
        y_samples: u32::try_from(y_samples)
            .map_err(|_| PbrtError::error("GPU stratified y samples do not fit in u32."))?,
        jitter,
        max_depth: u32::try_from(max_depth)
            .map_err(|_| PbrtError::error("GPU max depth does not fit in u32."))?,
        seed: u32::try_from(seed).map_err(|_| PbrtError::error("GPU seed does not fit in u32."))?,
        light_sampler,
        disable_wavelength_jitter: PbrtOptions::get().disable_wavelength_jitter,
    })
}

fn parse_randomization(
    sampler: Option<&NodeSampler>,
    default: &str,
    supports_fast_owen: bool,
) -> Result<SamplerRandomization, PbrtError> {
    let value = sampler
        .map(|sampler| sampler.params.get_one_string("randomization", default))
        .unwrap_or_else(|| default.to_string());
    match value.as_str() {
        "none" => Ok(SamplerRandomization::None),
        "permutedigits" => Ok(SamplerRandomization::PermuteDigits),
        "fastowen" if supports_fast_owen => Ok(SamplerRandomization::FastOwen),
        "owen" => Ok(SamplerRandomization::Owen),
        _ => Err(PbrtError::error(&format!(
            "GPU sampler randomization '{value}' is not supported."
        ))),
    }
}

pub fn viewport_resolution(params: &ParameterDictionary) -> Result<[u32; 2], PbrtError> {
    let mut xresolution = params.get_one_int("xresolution", 1280);
    let mut yresolution = params.get_one_int("yresolution", 720);
    let options = crate::options::PbrtOptions::get();
    if options.quick_render && !options.quick_render_full_resolution {
        xresolution = (xresolution / 4).max(1);
        yresolution = (yresolution / 4).max(1);
    }
    let resolution = [
        u32::try_from(xresolution)
            .map_err(|_| PbrtError::error("Film xresolution must be positive and fit in u32."))?,
        u32::try_from(yresolution)
            .map_err(|_| PbrtError::error("Film yresolution must be positive and fit in u32."))?,
    ];
    if resolution.contains(&0) {
        return Err(PbrtError::error("Film resolution must be positive."));
    }
    Ok(resolution)
}

pub fn screen_window(
    params: &ParameterDictionary,
    resolution: [u32; 2],
) -> Result<[f32; 4], PbrtError> {
    if let Some(values) = params.get_floats_ref("screenwindow") {
        if values.len() != 4 {
            return Err(PbrtError::error(
                "Camera screenwindow must contain four values.",
            ));
        }
        return Ok([
            values[0] as f32,
            values[1] as f32,
            values[2] as f32,
            values[3] as f32,
        ]);
    }

    let frame = params.get_one_float(
        "frameaspectratio",
        resolution[0] as f32 / resolution[1] as f32,
    ) as f32;
    if frame > 1.0 {
        Ok([-frame, frame, -1.0, 1.0])
    } else {
        Ok([-1.0, 1.0, -1.0 / frame, 1.0 / frame])
    }
}
