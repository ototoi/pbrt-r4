use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

use std::ops::*;

pub struct BilerpTexture<T> {
    mapping: TextureMapping2D,
    v00: T,
    v01: T,
    v10: T,
    v11: T,
}

impl<T: Clone + Add<T, Output = T> + Mul<Float, Output = T>> BilerpTexture<T> {
    pub fn new(mapping: TextureMapping2D, v00: &T, v01: &T, v10: &T, v11: &T) -> Self {
        return BilerpTexture::<T> {
            mapping,
            v00: v00.clone(),
            v01: v01.clone(),
            v10: v10.clone(),
            v11: v11.clone(),
        };
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> T {
        let (st, _dstdx, _dstdy) = self.mapping.map(ctx);
        let a = (1.0 - st[0]) * (1.0 - st[1]);
        let b = (1.0 - st[0]) * (st[1]);
        let c = (st[0]) * (1.0 - st[1]);
        let d = (st[0]) * (st[1]);
        return self.v00.clone() * a
            + self.v01.clone() * b
            + self.v10.clone() * c
            + self.v11.clone() * d;
    }
}

impl BilerpTexture<Float> {
    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
    ) -> Result<FloatTexture, PbrtError> {
        let map = TextureMapping2D::create(render_from_texture, parameters.parameter_dictionary())?;
        let v00 = parameters.get_one_float("v00", 0.0);
        let v01 = parameters.get_one_float("v01", 1.0);
        let v10 = parameters.get_one_float("v10", 0.0);
        let v11 = parameters.get_one_float("v11", 1.0);
        Ok(FloatTexture::Bilerp(BilerpTexture::<Float>::new(
            map, &v00, &v01, &v10, &v11,
        )))
    }
}

impl BilerpTexture<Spectrum> {
    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        let map = TextureMapping2D::create(render_from_texture, parameters.parameter_dictionary())?;
        let v00 = parameters.get_one_spectrum_typed("v00", &Spectrum::zero(), spectrum_type);
        let v01 = parameters.get_one_spectrum_typed("v01", &Spectrum::one(), spectrum_type);
        let v10 = parameters.get_one_spectrum_typed("v10", &Spectrum::zero(), spectrum_type);
        let v11 = parameters.get_one_spectrum_typed("v11", &Spectrum::one(), spectrum_type);
        Ok(BilerpTexture::<Spectrum>::new(map, &v00, &v01, &v10, &v11))
    }
}
