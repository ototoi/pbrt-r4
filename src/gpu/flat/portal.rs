use std::f32::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use super::texture::{build_linear_rgb_mipmap, Mipmap, MipmapLevelData};
use crate::util::base::Vector3f;
use crate::util::error::PbrtError;
use crate::util::geometry::equal_area_sphere_to_square;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PortalDistributionTexel {
    pub function: f32,
    pub summed_area: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PortalImageInfiniteLight {
    pub portal: [[f32; 3]; 4],
    pub world_to_portal: [[f32; 4]; 3],
    pub distribution_offset: u32,
    pub resolution: [u32; 2],
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

fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let length = dot(v, v).sqrt();
    (length.is_finite() && length > 0.0).then(|| v.map(|value| value / length))
}

fn portal_frame(portal: [[f32; 3]; 4]) -> Result<[[f32; 3]; 3], PbrtError> {
    if portal.iter().flatten().any(|value| !value.is_finite()) {
        return Err(PbrtError::error(
            "Portal geometry contains a non-finite value.",
        ));
    }
    let edges = (
        normalize(sub(portal[1], portal[0])),
        normalize(sub(portal[2], portal[1])),
        normalize(sub(portal[2], portal[3])),
        normalize(sub(portal[3], portal[0])),
    );
    let (p01, p12, p32, p03) = match edges {
        (Some(p01), Some(p12), Some(p32), Some(p03)) => (p01, p12, p32, p03),
        _ => return Err(PbrtError::error("Portal quadrilateral is degenerate.")),
    };
    if (dot(p01, p32) - 1.0).abs() > 0.001
        || (dot(p12, p03) - 1.0).abs() > 0.001
        || dot(p01, p12).abs() > 0.001
        || dot(p12, p32).abs() > 0.001
        || dot(p32, p03).abs() > 0.001
        || dot(p03, p01).abs() > 0.001
    {
        return Err(PbrtError::error(
            "Infinite light portal is not a planar quadrilateral.",
        ));
    }
    // pbrt-v4 Frame::FromXY(p03, p01).
    Ok([p03, p01, normalize(cross(p03, p01)).unwrap()])
}

fn from_portal(frame: [[f32; 3]; 3], local: [f32; 3]) -> [f32; 3] {
    [
        frame[0][0] * local[0] + frame[1][0] * local[1] + frame[2][0] * local[2],
        frame[0][1] * local[0] + frame[1][1] * local[1] + frame[2][1] * local[2],
        frame[0][2] * local[0] + frame[1][2] * local[1] + frame[2][2] * local[2],
    ]
}

fn transform_vector(matrix: [[f32; 4]; 3], vector: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|row| {
        matrix[row][0] * vector[0] + matrix[row][1] * vector[1] + matrix[row][2] * vector[2]
    })
}

fn render_from_image(frame: [[f32; 3]; 3], uv: [f32; 2]) -> ([f32; 3], f32) {
    let alpha = -FRAC_PI_2 + uv[0] * PI;
    let beta = -FRAC_PI_2 + uv[1] * PI;
    let local = normalize([alpha.tan(), beta.tan(), 1.0]).unwrap();
    let jacobian =
        PI.powi(2) * (1.0 - local[0] * local[0]) * (1.0 - local[1] * local[1]) / local[2];
    (from_portal(frame, local), jacobian)
}

fn remap_octahedral(mut x: i32, mut y: i32, resolution: [u32; 2]) -> [u32; 2] {
    let width = resolution[0] as i32;
    let height = resolution[1] as i32;
    if width == 1 && height == 1 {
        return [0, 0];
    }
    if x < 0 {
        x = -x;
        y = height - 1 - y;
    } else if x >= width {
        x = 2 * width - 1 - x;
        y = height - 1 - y;
    }
    if y < 0 {
        x = width - 1 - x;
        y = -y;
    } else if y >= height {
        x = width - 1 - x;
        y = 2 * height - 1 - y;
    }
    [x.clamp(0, width - 1) as u32, y.clamp(0, height - 1) as u32]
}

fn bilerp_octahedral(
    values: &[f32],
    resolution: [u32; 2],
    channels: u32,
    uv: [f32; 2],
) -> [f32; 3] {
    let p = [
        uv[0] * resolution[0] as f32 - 0.5,
        uv[1] * resolution[1] as f32 - 0.5,
    ];
    let p0 = [p[0].floor() as i32, p[1].floor() as i32];
    let d = [p[0] - p0[0] as f32, p[1] - p0[1] as f32];
    let sample = |x, y, channel: usize| {
        let [x, y] = remap_octahedral(x, y, resolution);
        values[((y * resolution[0] + x) * channels) as usize + channel]
    };
    std::array::from_fn(|channel| {
        let v00 = sample(p0[0], p0[1], channel);
        let v10 = sample(p0[0] + 1, p0[1], channel);
        let v01 = sample(p0[0], p0[1] + 1, channel);
        let v11 = sample(p0[0] + 1, p0[1] + 1, channel);
        (1.0 - d[1]) * ((1.0 - d[0]) * v00 + d[0] * v10) + d[1] * ((1.0 - d[0]) * v01 + d[0] * v11)
    })
}

/// Rectify an equal-area environment image into the portal's local frame and
/// build the CPU-side windowed distribution payload.
pub fn prepare_portal_image(
    source: &Arc<Mipmap>,
    portal: [[f32; 3]; 4],
    world_to_light: [[f32; 4]; 3],
) -> Result<PreparedPortalImage, PbrtError> {
    let level = source
        .levels
        .first()
        .ok_or_else(|| PbrtError::error("Portal image has no mipmap levels."))?;
    if level.resolution.contains(&0) || level.channels < 3 {
        return Err(PbrtError::error(
            "Portal image has invalid resolution or channels.",
        ));
    }
    if world_to_light
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(PbrtError::error(
            "Portal light transform contains a non-finite value.",
        ));
    }
    let values = match &level.data {
        MipmapLevelData::F32(values) => values,
        _ => return Err(PbrtError::error("Portal image storage must be float.")),
    };
    let pixel_count = level.resolution[0] as usize * level.resolution[1] as usize;
    if values.len() < pixel_count * level.channels as usize {
        return Err(PbrtError::error("Portal image data is inconsistent."));
    }

    let frame = portal_frame(portal)?;
    let mut rectified = Vec::with_capacity(pixel_count);
    let mut functions = Vec::with_capacity(pixel_count);
    for y in 0..level.resolution[1] {
        for x in 0..level.resolution[0] {
            let uv = [
                (x as f32 + 0.5) / level.resolution[0] as f32,
                (y as f32 + 0.5) / level.resolution[1] as f32,
            ];
            let (world_direction, jacobian) = render_from_image(frame, uv);
            let light_direction = normalize(transform_vector(world_to_light, world_direction))
                .ok_or_else(|| PbrtError::error("Portal light transform is degenerate."))?;
            let source_uv = equal_area_sphere_to_square(&Vector3f::new(
                light_direction[0],
                light_direction[1],
                light_direction[2],
            ));
            let rgb = bilerp_octahedral(
                values,
                level.resolution,
                level.channels,
                [source_uv.x, source_uv.y],
            );
            functions.push((rgb[0] + rgb[1] + rgb[2]) / 3.0 * jacobian);
            rectified.push(rgb);
        }
    }

    let mipmap = build_linear_rgb_mipmap(level.resolution, &rectified, source.color_space)?;
    let width = level.resolution[0] as usize;
    let mut sat = vec![0.0f64; pixel_count];
    let mut distribution = Vec::with_capacity(pixel_count);
    for (index, &function) in functions.iter().enumerate() {
        let x = index % width;
        let y = index / width;
        let left = if x > 0 { sat[index - 1] } else { 0.0 };
        let above = if y > 0 { sat[index - width] } else { 0.0 };
        let diagonal = if x > 0 && y > 0 {
            sat[index - width - 1]
        } else {
            0.0
        };
        sat[index] = function as f64 + left + above - diagonal;
        distribution.push(PortalDistributionTexel {
            function,
            summed_area: sat[index] as f32,
        });
    }
    if distribution
        .iter()
        .any(|value| !value.function.is_finite() || !value.summed_area.is_finite())
    {
        return Err(PbrtError::error(
            "Portal image sampling distribution is non-finite.",
        ));
    }

    Ok(PreparedPortalImage {
        mipmap,
        portal,
        world_to_portal: [
            [frame[0][0], frame[0][1], frame[0][2], 0.0],
            [frame[1][0], frame[1][1], frame[1][2], 0.0],
            [frame[2][0], frame[2][1], frame[2][2], 0.0],
        ],
        resolution: level.resolution,
        distribution,
    })
}
