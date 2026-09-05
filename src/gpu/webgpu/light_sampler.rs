use crate::gpu::ir::flat;
use crate::util::error::PbrtError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightSamplerKind {
    Uniform,
    Bvh,
}

pub fn resolve_light_sampler(
    requested: &str,
    registered_light_count: usize,
) -> Result<LightSamplerKind, PbrtError> {
    if registered_light_count == 1 {
        return Ok(LightSamplerKind::Uniform);
    }

    match requested {
        "uniform" => Ok(LightSamplerKind::Uniform),
        "bvh" => Ok(LightSamplerKind::Bvh),
        "power" | "exhaustive" => Err(PbrtError::error(&format!(
            "WebGPU light sampler \"{requested}\" is not implemented."
        ))),
        unknown => {
            log::error!("Unknown WebGPU light sampler \"{unknown}\"; using bvh.");
            Ok(LightSamplerKind::Bvh)
        }
    }
}

pub fn resolve_scene_light_sampler(
    settings: &flat::RenderSettings,
    registered_lights: &[flat::LightRecord],
) -> Result<LightSamplerKind, PbrtError> {
    resolve_light_sampler(&settings.light_sampler, registered_lights.len())
}
