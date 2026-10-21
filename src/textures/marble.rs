use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

pub struct MarbleTexture {
    mapping: TextureMapping3D,
    octaves: u32,
    omega: Float,
    scale: Float,
    variation: Float,
}

/// De Casteljau evaluation of a cubic Bezier over RGB control points.
fn evaluate_cubic_bezier(cp: &[[Float; 3]], t: Float) -> [Float; 3] {
    let lerp = |a: &[Float; 3], b: &[Float; 3]| -> [Float; 3] {
        [
            (1.0 - t) * a[0] + t * b[0],
            (1.0 - t) * a[1] + t * b[1],
            (1.0 - t) * a[2] + t * b[2],
        ]
    };
    let cp1 = [
        lerp(&cp[0], &cp[1]),
        lerp(&cp[1], &cp[2]),
        lerp(&cp[2], &cp[3]),
    ];
    let cp2 = [lerp(&cp1[0], &cp1[1]), lerp(&cp1[1], &cp1[2])];
    lerp(&cp2[0], &cp2[1])
}

impl MarbleTexture {
    pub fn new(
        mapping: TextureMapping3D,
        octaves: u32,
        omega: Float,
        scale: Float,
        variation: Float,
    ) -> Self {
        Self {
            mapping,
            octaves,
            omega,
            scale,
            variation,
        }
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Spectrum {
        self.evaluate_colors(&C, ctx)
    }

    pub fn evaluate_colors(&self, colors: &[[Float; 3]], ctx: &TextureEvalContext) -> Spectrum {
        let variation = self.variation;
        let scale = self.scale;
        let omega = self.omega;
        let octaves = self.octaves;
        let (p, dpdx, dpdy) = self.mapping.map(ctx);
        let p = scale * p;
        let marble = p.y + variation * fbm(&p, &(scale * dpdx), &(scale * dpdy), omega, octaves);
        let t = 0.5 + 0.5 * Float::sin(marble);
        // Evaluate marble spline at _t_ (pbrt-v4 textures.cpp:487-501). The
        // cubic Bezier is evaluated in RGB and the result becomes an albedo
        // spectrum.
        let nseg = colors.len() - 3;
        let first = usize::min(Float::floor(t * nseg as Float) as usize, nseg - 1);
        let t = t * nseg as Float - first as Float;
        let rgb = evaluate_cubic_bezier(&colors[first..first + 4], t);
        // Extra scale of 1.5 to increase variation among colors
        Spectrum::from_rgb_albedo(&rgb.map(|c| 1.5 * c))
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        _spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        let map = TextureMapping3D::create(parameters.parameter_dictionary(), render_from_texture);
        let octaves = parameters.get_one_int("octaves", 8) as u32;
        let roughness = parameters.get_one_float("roughness", 0.5);
        let scale = parameters.get_one_float("scale", 1.0);
        let variation = parameters.get_one_float("variation", 0.2);
        Ok(MarbleTexture::new(
            map, octaves, roughness, scale, variation,
        ))
    }
}

const C: [[Float; 3]; 9] = [
    [0.58, 0.58, 0.6],
    [0.58, 0.58, 0.6],
    [0.58, 0.58, 0.6],
    [0.5, 0.5, 0.5],
    [0.6, 0.59, 0.58],
    [0.58, 0.58, 0.6],
    [0.58, 0.58, 0.6],
    [0.2, 0.2, 0.33],
    [0.58, 0.58, 0.6],
];
