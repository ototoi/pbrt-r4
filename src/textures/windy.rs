use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

pub struct WindyTexture {
    mapping: TextureMapping3D,
}

impl WindyTexture {
    pub fn new(mapping: TextureMapping3D) -> Self {
        Self { mapping }
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let (p, dpdx, dpdy) = self.mapping.map(ctx);
        let wind_strength = fbm(&(0.1 * p), &(0.1 * dpdx), &(0.1 * dpdy), 0.5, 3);
        let wave_height = fbm(&p, &dpdx, &dpdy, 0.5, 6);
        return Float::abs(wind_strength) * wave_height;
    }

    pub fn evaluate_spectrum(&self, ctx: &TextureEvalContext) -> Spectrum {
        return Spectrum::from(self.evaluate(ctx));
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
    ) -> Result<Self, PbrtError> {
        let map = TextureMapping3D::create(parameters.parameter_dictionary(), render_from_texture);
        Ok(WindyTexture::new(map))
    }
}
