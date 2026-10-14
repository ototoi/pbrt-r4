use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use crate::gpu::flat::{Filter, FilterKind};
use crate::util::error::PbrtError;

use super::scene::medium::validate_storage_array_size;

pub struct FilterResources {
    pub buffer: wgpu::Buffer,
}

impl FilterResources {
    pub fn new(device: &wgpu::Device, filter: &Filter) -> Result<Self, PbrtError> {
        let mut count = 14usize;
        if let Some(d) = &filter.distribution {
            count = [
                d.values.len(),
                d.conditional_cdf.len(),
                d.marginal_cdf.len(),
            ]
            .into_iter()
            .try_fold(12usize, |sum, n| sum.checked_add(n))
            .and_then(|n| n.checked_add(n % 2))
            .ok_or_else(|| PbrtError::error("GPU filter buffer size overflows usize."))?;
        }
        validate_storage_array_size::<u32>(&device.limits(), count, "filter")?;
        let words = filter_buffer_data(filter)?;
        Ok(Self {
            buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 filter table"),
                contents: cast_slice(&words),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        })
    }
}

/// Header matches WGSL FilterData; offsets address its trailing f32 array.
pub fn filter_buffer_data(filter: &Filter) -> Result<Vec<u32>, PbrtError> {
    if filter.radius.iter().any(|r| !r.is_finite() || *r <= 0.0) {
        return Err(PbrtError::error("GPU filter radius is invalid."));
    }
    let mut words = vec![
        filter.kind as u32,
        0,
        0,
        0,
        filter.radius[0].to_bits(),
        filter.radius[1].to_bits(),
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    if matches!(filter.kind, FilterKind::Box | FilterKind::Triangle) {
        if filter.distribution.is_some() {
            return Err(PbrtError::error(
                "An analytic GPU filter must not have a distribution.",
            ));
        }
        words.extend([0, 0]);
        return Ok(words);
    }
    let d = filter
        .distribution
        .as_ref()
        .ok_or_else(|| PbrtError::error("GPU filter distribution is missing."))?;
    let [nx, ny] = d.dimensions.map(u64::from);
    let invalid = || PbrtError::error("GPU filter distribution layout or values are invalid.");
    let cells = nx.checked_mul(ny).ok_or_else(invalid)?;
    let conditional = (nx + 1).checked_mul(ny).ok_or_else(invalid)?;
    let total_words = cells
        .checked_add(conditional)
        .and_then(|n| n.checked_add(ny + 1 + 12))
        .ok_or_else(invalid)?;
    if nx == 0
        || ny == 0
        || total_words > u32::MAX as u64 / 4
        || d.values.len() as u64 != cells
        || d.conditional_cdf.len() as u64 != conditional
        || d.marginal_cdf.len() as u64 != ny + 1
        || !d.integral.is_finite()
        || d.integral <= 0.0
        || d.values.iter().any(|v| !v.is_finite())
    {
        return Err(invalid());
    }
    let valid_cdf = |cdf: &[f32]| {
        cdf.first() == Some(&0.0)
            && cdf.last() == Some(&1.0)
            && cdf.windows(2).all(|w| w[0].is_finite() && w[0] <= w[1])
    };
    if !valid_cdf(&d.marginal_cdf)
        || d.conditional_cdf
            .chunks_exact(nx as usize + 1)
            .any(|cdf| !valid_cdf(cdf))
    {
        return Err(invalid());
    }
    let area = 4.0 * filter.radius[0] * filter.radius[1];
    let absolute_integral = area * d.integral;
    if !area.is_finite() || area <= 0.0 || !absolute_integral.is_normal() {
        return Err(invalid());
    }
    words[1] = d.dimensions[0];
    words[2] = d.dimensions[1];
    words[6] = absolute_integral.to_bits();
    for (slot, values) in [
        (8, &d.values),
        (9, &d.conditional_cdf),
        (10, &d.marginal_cdf),
    ] {
        words[slot] = (words.len() - 12) as u32;
        words.extend(values.iter().map(|v| v.to_bits()));
    }
    // FilterData has 8-byte struct alignment due to its vec2 radius.
    if words.len() % 2 != 0 {
        words.push(0);
    }
    Ok(words)
}
