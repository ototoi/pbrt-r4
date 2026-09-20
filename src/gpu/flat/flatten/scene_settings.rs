use super::MAX_GPU_RENDER_DEPTH;
use crate::gpu::flat::RenderSettings;
use crate::gpu::node::{Integrator as NodeIntegrator, Sampler as NodeSampler};
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
    if let Some(sampler) = sampler {
        if sampler.name != "independent" {
            log::warn!(
                "GPU sampler '{}' is not implemented; falling back to independent sampler.",
                sampler.name
            );
        }
    }
    if let Some(integrator) = integrator {
        if integrator.name != "path" && integrator.name != "volpath" {
            return Err(PbrtError::error(&format!(
                "Unsupported GPU integrator: {}.",
                integrator.name
            )));
        }
    }
    let configured_samples_per_pixel = sampler
        .map(|sampler| sampler.params.get_one_int("pixelsamples", 4))
        .unwrap_or(4);
    let samples_per_pixel = if crate::options::PbrtOptions::get().quick_render {
        1
    } else {
        configured_samples_per_pixel
    };
    let configured_max_depth = integrator
        .map(|integrator| integrator.params.get_one_int("maxdepth", 5))
        .unwrap_or(5);
    let max_depth = configured_max_depth.min(MAX_GPU_RENDER_DEPTH);
    if configured_max_depth > MAX_GPU_RENDER_DEPTH {
        log::warn!(
            "GPU maxdepth {} exceeds the backend limit {}; clamping to {}.",
            configured_max_depth,
            MAX_GPU_RENDER_DEPTH,
            MAX_GPU_RENDER_DEPTH
        );
    }
    let seed = sampler
        .map(|sampler| sampler.params.get_one_int("seed", 0))
        .unwrap_or(0);
    let light_sampler = integrator
        .map(|integrator| integrator.params.get_one_string("lightsampler", "bvh"))
        .unwrap_or_else(|| "bvh".to_string());
    if samples_per_pixel <= 0 || max_depth < 0 || seed < 0 {
        return Err(PbrtError::error(
            "GPU render settings must have positive samples and non-negative depth/seed.",
        ));
    }
    Ok(RenderSettings {
        samples_per_pixel: u32::try_from(samples_per_pixel)
            .map_err(|_| PbrtError::error("GPU samples per pixel do not fit in u32."))?,
        max_depth: u32::try_from(max_depth)
            .map_err(|_| PbrtError::error("GPU max depth does not fit in u32."))?,
        seed: u32::try_from(seed).map_err(|_| PbrtError::error("GPU seed does not fit in u32."))?,
        light_sampler,
        disable_wavelength_jitter: crate::options::PbrtOptions::get().disable_wavelength_jitter,
    })
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
