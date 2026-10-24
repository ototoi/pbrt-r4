use crate::gpu::flat;
use crate::util::error::PbrtError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightSamplerKind {
    Uniform,
    Bvh,
    Power,
    Exhaustive,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerAliasBin {
    pub q: f32,
    pub p: f32,
    pub alias: u32,
}

pub const EXHAUSTIVE_BOUNDS_WORDS: usize = 14;

pub fn pack_exhaustive_bounds(bounds: &[flat::LightBounds]) -> Result<Vec<u32>, PbrtError> {
    let count = u32::try_from(bounds.len())
        .map_err(|_| PbrtError::error("Exhaustive light bounds count exceeds u32."))?;
    let word_count = bounds
        .len()
        .checked_mul(EXHAUSTIVE_BOUNDS_WORDS)
        .and_then(|words| words.checked_add(1))
        .ok_or_else(|| PbrtError::error("Exhaustive light sampler data size overflowed."))?;
    let mut words = Vec::with_capacity(word_count);
    words.push(count);
    for (handle, bound) in bounds.iter().enumerate() {
        words.push(
            u32::try_from(handle)
                .map_err(|_| PbrtError::error("Exhaustive light handle exceeds u32."))?,
        );
        words.extend(bound.bounds.min.into_iter().map(f32::to_bits));
        words.extend(bound.bounds.max.into_iter().map(f32::to_bits));
        words.extend(bound.direction.into_iter().map(f32::to_bits));
        words.push(bound.phi.to_bits());
        words.push(bound.cos_theta_o.to_bits());
        words.push(bound.cos_theta_e.to_bits());
        words.push(u32::from(bound.two_sided));
    }
    Ok(words)
}

pub fn pack_power_weights(weights: &[f32]) -> Result<Vec<u32>, PbrtError> {
    let count = u32::try_from(weights.len())
        .map_err(|_| PbrtError::error("Power light count exceeds u32."))?;
    let bins = build_power_alias_table(weights)?;
    let word_count = bins
        .len()
        .checked_mul(3)
        .and_then(|words| words.checked_add(1))
        .ok_or_else(|| PbrtError::error("Power light sampler data size overflowed."))?;
    let mut words = Vec::with_capacity(word_count);
    words.push(count);
    for bin in bins {
        words.extend([bin.q.to_bits(), bin.p.to_bits(), bin.alias]);
    }
    Ok(words)
}

/// Builds the pbrt-v4 AliasTable representation for a fixed set of weights.
pub fn build_power_alias_table(weights: &[f32]) -> Result<Vec<PowerAliasBin>, PbrtError> {
    if weights.is_empty() {
        return Ok(Vec::new());
    }
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(PbrtError::error(
            "Power light weights must be finite and non-negative.",
        ));
    }
    let mut weights = weights.to_vec();
    if weights.iter().sum::<f32>() == 0.0 {
        weights.fill(1.0);
    }
    let sum = weights.iter().map(|weight| f64::from(*weight)).sum::<f64>() as f32;
    let mut bins = weights
        .iter()
        .map(|weight| PowerAliasBin {
            q: 0.0,
            p: *weight / sum,
            alias: u32::MAX,
        })
        .collect::<Vec<_>>();
    let mut under = Vec::new();
    let mut over = Vec::new();
    for (index, bin) in bins.iter().enumerate() {
        let p_hat = bin.p * bins.len() as f32;
        if p_hat < 1.0 {
            under.push((p_hat, index));
        } else {
            over.push((p_hat, index));
        }
    }
    while !under.is_empty() && !over.is_empty() {
        let (under_weight, under_index) = under.pop().unwrap();
        let (over_weight, over_index) = over.pop().unwrap();
        bins[under_index].q = under_weight;
        bins[under_index].alias = over_index as u32;
        let excess = under_weight + over_weight - 1.0;
        if excess < 1.0 {
            under.push((excess, over_index));
        } else {
            over.push((excess, over_index));
        }
    }
    for (_, index) in under.into_iter().chain(over) {
        bins[index].q = 1.0;
    }
    Ok(bins)
}

pub fn resolve_light_sampler(
    requested: &str,
    registered_light_count: usize,
) -> Result<LightSamplerKind, PbrtError> {
    if registered_light_count <= 1 {
        return Ok(LightSamplerKind::Uniform);
    }

    match requested {
        "uniform" => Ok(LightSamplerKind::Uniform),
        "bvh" => Ok(LightSamplerKind::Bvh),
        "power" => Ok(LightSamplerKind::Power),
        "exhaustive" => Ok(LightSamplerKind::Exhaustive),
        unknown => Err(PbrtError::error(&format!(
            "Unknown WebGPU light sampler \"{unknown}\"."
        ))),
    }
}

pub fn resolve_scene_light_sampler(
    settings: &flat::RenderSettings,
    registered_lights: &[flat::Light],
) -> Result<LightSamplerKind, PbrtError> {
    resolve_light_sampler(&settings.light_sampler, registered_lights.len())
}

pub fn resolve_scene_light_sampler_count(
    settings: &flat::RenderSettings,
    sampleable_light_count: usize,
) -> Result<LightSamplerKind, PbrtError> {
    resolve_light_sampler(&settings.light_sampler, sampleable_light_count)
}
