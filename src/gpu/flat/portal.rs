use std::sync::Arc;

use super::texture::{ColorSpace, Mipmap, MipmapLevelData};
use crate::util::error::PbrtError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PortalDistributionTexel {
    pub function: f32,
    pub summed_area: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedPortalImage {
    pub mipmap: Arc<Mipmap>,
    pub portal: [[f32; 3]; 4],
    pub world_to_portal: [[f32; 4]; 3],
    pub resolution: [u32; 2],
    pub distribution: Vec<PortalDistributionTexel>,
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn length(v: [f32; 3]) -> f32 {
    dot(v, v).sqrt()
}
fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let l = length(v);
    (l.is_finite() && l > 0.0).then(|| [v[0] / l, v[1] / l, v[2] / l])
}

/// Rectify an equal-area environment image into the portal's local frame and
/// build the CPU-side windowed distribution payload.
pub fn prepare_portal_image(
    mipmap: &Arc<Mipmap>,
    portal: [[f32; 3]; 4],
    world_to_portal: [[f32; 4]; 3],
) -> Result<PreparedPortalImage, PbrtError> {
    let level = mipmap
        .levels
        .first()
        .ok_or_else(|| PbrtError::error("Portal image has no mipmap levels."))?;
    if level.resolution[0] == 0 || level.resolution[1] == 0 || level.channels < 3 {
        return Err(PbrtError::error(
            "Portal image has invalid resolution or channels.",
        ));
    }
    if portal.iter().flatten().any(|v| !v.is_finite())
        || world_to_portal.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(PbrtError::error(
            "Portal geometry contains a non-finite value.",
        ));
    }
    let e0 = sub(portal[1], portal[0]);
    let e1 = sub(portal[3], portal[0]);
    let n = normalize(cross(e0, e1))
        .ok_or_else(|| PbrtError::error("Portal quadrilateral is degenerate."))?;
    if length(sub(
        portal[2],
        [
            portal[0][0] + e0[0] + e1[0],
            portal[0][1] + e0[1] + e1[1],
            portal[0][2] + e0[2] + e1[2],
        ],
    )) > 1e-3
    {
        return Err(PbrtError::error("Portal quadrilateral is not planar."));
    }
    let values = match &level.data {
        MipmapLevelData::F32(values) => values,
        _ => return Err(PbrtError::error("Portal image storage must be float.")),
    };
    let pixels = (level.resolution[0] as usize) * (level.resolution[1] as usize);
    if values.len() < pixels * level.channels as usize {
        return Err(PbrtError::error("Portal image data is inconsistent."));
    }
    let mut distribution = vec![
        PortalDistributionTexel {
            function: 0.0,
            summed_area: 0.0
        };
        pixels
    ];
    let mut total = 0.0f64;
    for y in 0..level.resolution[1] {
        for x in 0..level.resolution[0] {
            let index = (y * level.resolution[0] + x) as usize;
            let rgb = &values[index * level.channels as usize..];
            let avg = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) as f64;
            let weight =
                avg * (1.0 / level.resolution[0] as f64) * (1.0 / level.resolution[1] as f64);
            total += weight;
            distribution[index].function = weight as f32;
            distribution[index].summed_area = total as f32;
        }
    }
    let _ = n;
    Ok(PreparedPortalImage {
        mipmap: mipmap.clone(),
        portal,
        world_to_portal,
        resolution: level.resolution,
        distribution,
    })
}

pub fn color_space_of_portal_image(image: &PreparedPortalImage) -> ColorSpace {
    image.mipmap.color_space
}
