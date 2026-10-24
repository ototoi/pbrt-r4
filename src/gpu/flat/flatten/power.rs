use super::super::texture::{Mipmap, MipmapLevelData};
use super::super::SourceShape;
use super::{transform_point, GeometryBuilder, Instance};
use crate::util::base::Float;
use crate::util::error::PbrtError;
use crate::util::spectrum::rgb_to_spectrum::RGBColorSpace;
use crate::util::spectrum::{safe_div, SampledSpectrum, SampledWavelengths, Spectrum};
use std::f32::consts::PI;

#[derive(Clone, Copy, Debug)]
pub(super) struct PowerInput {
    weight: f32,
    scales_with_scene_disk: bool,
}

impl PowerInput {
    pub(super) fn weight(self, scene_radius_squared: f32) -> f32 {
        if self.scales_with_scene_disk {
            self.weight * scene_radius_squared
        } else {
            self.weight
        }
    }

    pub(super) fn scales_with_scene_disk(self) -> bool {
        self.scales_with_scene_disk
    }
}

fn power_lambda() -> SampledWavelengths {
    SampledWavelengths::sample_visible(0.5)
}

fn power_weight(phi: SampledSpectrum, lambda: &SampledWavelengths) -> f32 {
    safe_div(phi, lambda.pdf_spectrum()).average().max(0.0) as f32
}

fn fixed(phi: impl FnOnce(&SampledWavelengths) -> SampledSpectrum) -> PowerInput {
    let lambda = power_lambda();
    PowerInput {
        weight: power_weight(phi(&lambda), &lambda),
        scales_with_scene_disk: false,
    }
}

fn per_scene_disk(phi: impl FnOnce(&SampledWavelengths) -> SampledSpectrum) -> PowerInput {
    PowerInput {
        scales_with_scene_disk: true,
        ..fixed(phi)
    }
}

pub(super) fn point_power(intensity: &Spectrum, scale: f32) -> PowerInput {
    fixed(|lambda| intensity.sample(lambda) * (4.0 * PI * scale))
}

pub(super) fn spot_power(
    intensity: &Spectrum,
    scale: f32,
    cos_falloff_start: f32,
    cos_falloff_end: f32,
) -> PowerInput {
    fixed(|lambda| {
        intensity.sample(lambda)
            * (scale
                * 2.0
                * PI
                * ((1.0 - cos_falloff_start) + (cos_falloff_start - cos_falloff_end) / 2.0))
    })
}

pub(super) fn area_power(
    emission: &Spectrum,
    scale: f32,
    area: f32,
    two_sided: bool,
) -> PowerInput {
    fixed(|lambda| {
        emission.sample(lambda) * (scale * area * PI * if two_sided { 2.0 } else { 1.0 })
    })
}

pub(super) fn distant_power(emission: &Spectrum, scale: f32) -> PowerInput {
    per_scene_disk(|lambda| emission.sample(lambda) * (scale * PI))
}

pub(super) fn uniform_infinite_power(emission: &Spectrum, scale: f32) -> PowerInput {
    per_scene_disk(|lambda| emission.sample(lambda) * (4.0 * PI * PI * scale))
}

/// Base-level texel illuminant sum of an image or portal infinite light, before `scale`.
pub(super) struct ImagePowerSum {
    sum: SampledSpectrum,
    texel_count: f32,
    portal_area: Option<f32>,
}

impl ImagePowerSum {
    /// v4 `ImageInfiniteLight::Phi`.
    pub(super) fn image(mipmap: &Mipmap, color_space: &RGBColorSpace) -> Result<Self, PbrtError> {
        let (sum, texel_count) = texel_sum(mipmap, color_space, |_| Some(1.0))?;
        Ok(Self {
            sum,
            texel_count,
            portal_area: None,
        })
    }

    /// v4 `PortalImageInfiniteLight::Phi` over the rectified portal image.
    pub(super) fn portal(
        mipmap: &Mipmap,
        color_space: &RGBColorSpace,
        portal: &[[f32; 3]; 4],
    ) -> Result<Self, PbrtError> {
        let (sum, texel_count) = texel_sum(mipmap, color_space, portal_duv_dw)?;
        let length = |a: [f32; 3], b: [f32; 3]| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        Ok(Self {
            sum,
            texel_count,
            portal_area: Some(length(portal[1], portal[0]) * length(portal[3], portal[0])),
        })
    }

    pub(super) fn power(&self, scale: f32) -> PowerInput {
        match self.portal_area {
            Some(area) => fixed(|_| self.sum * (scale * area / self.texel_count) as Float),
            None => {
                per_scene_disk(|_| self.sum * (4.0 * PI * PI * scale / self.texel_count) as Float)
            }
        }
    }
}

fn texel_sum(
    mipmap: &Mipmap,
    color_space: &RGBColorSpace,
    texel_duv_dw: impl Fn([f32; 2]) -> Option<f32>,
) -> Result<(SampledSpectrum, f32), PbrtError> {
    let level = mipmap
        .levels
        .first()
        .ok_or_else(|| PbrtError::error("Infinite light image has no mipmap levels."))?;
    let MipmapLevelData::F32(values) = &level.data else {
        return Err(PbrtError::error(
            "Infinite light image must have linear float storage.",
        ));
    };
    let [width, height] = level.resolution;
    let lambda = power_lambda();
    let mut sum = SampledSpectrum::zero();
    for y in 0..height {
        for x in 0..width {
            let st = [
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            ];
            let Some(duv_dw) = texel_duv_dw(st) else {
                continue;
            };
            let offset = ((y * width + x) * level.channels) as usize;
            let rgb = [
                values[offset] as Float,
                values[offset + 1] as Float,
                values[offset + 2] as Float,
            ];
            sum += color_space.illuminant_to_sampled_spectrum(rgb, &lambda) / duv_dw as Float;
        }
    }
    Ok((sum, (width * height) as f32))
}

/// `|duv/dw|` of v4 `PortalImageInfiniteLight::RenderFromImage`; `None` where it vanishes.
fn portal_duv_dw(st: [f32; 2]) -> Option<f32> {
    let x = (-0.5 * PI + st[0] * PI).tan();
    let y = (-0.5 * PI + st[1] * PI).tan();
    let length = (x * x + y * y + 1.0).sqrt();
    let (x, y, z) = (x / length, y / length, 1.0 / length);
    let duv_dw = PI * PI * (1.0 - x * x) * (1.0 - y * y) / z;
    (z > 0.0 && duv_dw > 0.0).then_some(duv_dw)
}

/// Squared bounding-sphere radius of the scene, as used by v4 infinite and distant lights.
/// Quadrics and object-instance shapes contribute their transformed object-space box
/// corners, like v4 quadric and `TransformedPrimitive` bounds; meshes contribute exact
/// transformed vertices.
pub(super) fn scene_radius_squared(
    instances: &[Instance],
    in_object_instance: &[bool],
    geometry: &GeometryBuilder,
) -> f32 {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut include = |point: [f32; 3]| {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    };
    let mut vertex_bounds = vec![None; geometry.geometries.len()];
    for (instance, &in_object) in instances.iter().zip(in_object_instance) {
        let index = instance.geometry as usize;
        let Some(range) = geometry.geometries.get(index) else {
            continue;
        };
        let vertices = &geometry.vertices
            [range.first_vertex as usize..(range.first_vertex + range.vertex_count) as usize];
        let quadric = matches!(
            range.source_shape,
            SourceShape::Disk
                | SourceShape::Sphere
                | SourceShape::Cylinder
                | SourceShape::Cone
                | SourceShape::Paraboloid
                | SourceShape::Hyperboloid
        );
        if !in_object && !quadric {
            for vertex in vertices {
                include(transform_point(&instance.transform, vertex.position));
            }
            continue;
        }
        let bounds = match geometry.object_bounds.get(index).copied().flatten() {
            Some(bounds) => bounds,
            None => *vertex_bounds[index].get_or_insert_with(|| {
                let mut bounds = [[f32::INFINITY; 3], [f32::NEG_INFINITY; 3]];
                for vertex in vertices {
                    for axis in 0..3 {
                        bounds[0][axis] = bounds[0][axis].min(vertex.position[axis]);
                        bounds[1][axis] = bounds[1][axis].max(vertex.position[axis]);
                    }
                }
                bounds
            }),
        };
        if bounds[0][0] > bounds[1][0] {
            continue;
        }
        for corner in 0..8 {
            include(transform_point(
                &instance.transform,
                std::array::from_fn(|axis| bounds[(corner >> axis) & 1][axis]),
            ));
        }
    }
    if min[0] > max[0] {
        return 0.0;
    }
    let diagonal_squared: f32 = (0..3).map(|axis| (max[axis] - min[axis]).powi(2)).sum();
    diagonal_squared / 4.0
}
