use crate::paramdict::*;
use crate::textures::TextureEvalContext;

use crate::shapes::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

pub struct NormalTexture {}

impl NormalTexture {
    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Spectrum {
        return Spectrum::from_rgb_albedo(&[
            (ctx.n.x + 1.0) * 0.5,
            (ctx.n.y + 1.0) * 0.5,
            (ctx.n.z + 1.0) * 0.5,
        ]);
    }

    pub fn create(
        _render_from_texture: &Transform,
        _parameters: &TextureParameterDictionary,
        _spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        Ok(NormalTexture {})
    }
}
