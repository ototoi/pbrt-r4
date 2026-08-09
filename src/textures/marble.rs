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

fn lerps(c0: &Spectrum, c1: &Spectrum, t: Float) -> Spectrum {
    return c0.clone() * (1.0 - t) + c1.clone() * t;
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
        // Evaluate marble spline at _t_
        let nc = colors.len();
        let nseg = nc - 3;
        let first = usize::min(1, Float::floor(t * nseg as Float) as usize);
        let t = t * nseg as Float - first as Float;
        let c0 = Spectrum::from_rgb_albedo(&colors[first]);
        let c1 = Spectrum::from_rgb_albedo(&colors[first + 1]);
        let c2 = Spectrum::from_rgb_albedo(&colors[first + 2]);
        let c3 = Spectrum::from_rgb_albedo(&colors[first + 3]);
        // Bezier spline evaluated with de Castilejau's algorithm
        let s0 = lerps(&c0, &c1, t);
        let s1 = lerps(&c1, &c2, t);
        let s2 = lerps(&c2, &c3, t);
        let s0 = lerps(&s0, &s1, t);
        let s1 = lerps(&s1, &s2, t);
        // Extra scale of 1.5 to increase variation among colors
        return lerps(&s0, &s1, t) * 1.5;
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
