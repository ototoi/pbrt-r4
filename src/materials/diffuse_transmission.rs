use super::material_eval_context::MaterialEvalContext;
use super::normal_bump::{apply_normal_or_bump, get_normal_map, NormalMap};
use crate::bxdfs::DiffuseTransmissionBxDF;
use crate::interaction::SurfaceInteraction;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::spectrum::*;

use std::sync::Arc;

pub struct DiffuseTransmissionMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<NormalMap>>,
    reflectance: Arc<SpectrumTexture>,
    transmittance: Arc<SpectrumTexture>,
    scale: Float,
}

impl DiffuseTransmissionMaterial {
    pub fn new(
        reflectance: &Arc<SpectrumTexture>,
        transmittance: &Arc<SpectrumTexture>,
        scale: Float,
        displacement: &Option<Arc<FloatTexture>>,
        normal_map: &Option<Arc<NormalMap>>,
    ) -> Self {
        DiffuseTransmissionMaterial {
            displacement: displacement.clone(),
            normal_map: normal_map.clone(),
            reflectance: reflectance.clone(),
            transmittance: transmittance.clone(),
            scale,
        }
    }

    pub fn apply_displacement(&self, si: &mut SurfaceInteraction) {
        apply_normal_or_bump(&self.normal_map, &self.displacement, si);
    }

    /// Translation of pbrt-v4 `DiffuseTransmissionMaterial::GetBxDF`
    /// (`materials.h`): evaluate reflectance / transmittance at
    /// `lambda`, sample to a `SampledSpectrum`, clamp to [0, 1] after
    /// scale, then construct the BxDF.
    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> DiffuseTransmissionBxDF {
        let texture_ctx = ctx.texture_context();
        let r = (tex_eval.evaluate_spectrum(self.reflectance.as_ref(), texture_ctx, lambda)
            * self.scale)
            .clamp(0.0, 1.0);
        let t = (tex_eval.evaluate_spectrum(self.transmittance.as_ref(), texture_ctx, lambda)
            * self.scale)
            .clamp(0.0, 1.0);
        DiffuseTransmissionBxDF::new(r, t)
    }

    pub fn create(
        mp: &TextureParameterDictionary,
    ) -> Result<DiffuseTransmissionMaterial, PbrtError> {
        let reflectance = mp
            .get_spectrum_texture_or_null_typed("reflectance", SpectrumType::Albedo)?
            .or(mp.get_spectrum_texture_or_null_typed("Kd", SpectrumType::Albedo)?)
            .unwrap_or_else(|| {
                Arc::new(SpectrumTexture::Constant(ConstantTexture::new(
                    &Spectrum::from(0.25),
                )))
            });
        let transmittance = mp
            .get_spectrum_texture_or_null_typed("transmittance", SpectrumType::Albedo)?
            .or(mp.get_spectrum_texture_or_null_typed("Kt", SpectrumType::Albedo)?)
            .unwrap_or_else(|| {
                Arc::new(SpectrumTexture::Constant(ConstantTexture::new(
                    &Spectrum::from(0.25),
                )))
            });
        let scale = mp.get_one_float("scale", 1.0);
        let displacement = mp.get_float_texture_or_null("displacement")?;
        let normal_map = get_normal_map(mp)?;
        Ok(DiffuseTransmissionMaterial::new(
            &reflectance,
            &transmittance,
            scale,
            &displacement,
            &normal_map,
        ))
    }
}
