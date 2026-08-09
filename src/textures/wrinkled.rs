use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

pub struct WrinkledTexture {
    mapping: TextureMapping3D,
    octaves: u32,
    omega: Float,
}

impl WrinkledTexture {
    pub fn new(mapping: TextureMapping3D, octaves: u32, omega: Float) -> Self {
        Self {
            mapping,
            octaves,
            omega,
        }
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let (p, dpdx, dpdy) = self.mapping.map(ctx);
        return turbulence(&p, &dpdx, &dpdy, self.omega, self.octaves);
    }

    pub fn evaluate_spectrum(&self, ctx: &TextureEvalContext) -> Spectrum {
        return Spectrum::from(self.evaluate(ctx));
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
    ) -> Result<Self, PbrtError> {
        let map = TextureMapping3D::create(parameters.parameter_dictionary(), render_from_texture);
        let octaves = parameters.get_one_int("octaves", 8) as u32;
        let roughness = parameters.get_one_float("roughness", 0.5);
        Ok(WrinkledTexture::new(map, octaves, roughness))
    }
}
