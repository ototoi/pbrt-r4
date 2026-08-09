use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

pub struct UVTexture {
    mapping: TextureMapping2D,
}

impl UVTexture {
    pub fn new(mapping: TextureMapping2D) -> Self {
        UVTexture { mapping }
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Spectrum {
        let (st, _dstdx, _dstdy) = self.mapping.map(ctx);
        let rgb = [
            st[0] - Float::floor(st[0]),
            st[1] - Float::floor(st[1]),
            0.0,
        ];
        return Spectrum::from_rgb_albedo(&rgb);
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        _spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        let map = TextureMapping2D::create(render_from_texture, parameters.parameter_dictionary())?;
        Ok(UVTexture::new(map))
    }
}
