use super::material_eval_context::MaterialEvalContext;
use super::normal_bump::{apply_normal_or_bump, get_normal_map, NormalMap};
use crate::bxdfs::DiffuseBxDF;
use crate::interaction::SurfaceInteraction;
use crate::paramdict::*;
use crate::textures::*;
use crate::util::error::*;
use crate::util::spectrum::*;
use std::sync::Arc;

pub struct DiffuseMaterial {
    reflectance: Arc<SpectrumTexture>,
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<NormalMap>>,
}

impl DiffuseMaterial {
    pub fn new(
        kd: &Arc<SpectrumTexture>,
        displacement: &Option<Arc<FloatTexture>>,
        normal_map: &Option<Arc<NormalMap>>,
    ) -> Self {
        DiffuseMaterial {
            reflectance: kd.clone(),
            displacement: displacement.clone(),
            normal_map: normal_map.clone(),
        }
    }
}

impl DiffuseMaterial {
    pub fn apply_displacement(&self, si: &mut SurfaceInteraction) {
        apply_normal_or_bump(&self.normal_map, &self.displacement, si);
    }

    /// Translation of pbrt-v4 `DiffuseMaterial::GetBxDF`
    /// (`materials.h:1059`): evaluate the reflectance texture at
    /// `lambda` to get a `SampledSpectrum`, clamp to [0, 1] (v4
    /// `Clamp(..., 0, 1)`), and build the `DiffuseBxDF`.
    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> DiffuseBxDF {
        let r = tex_eval
            .evaluate_spectrum(self.reflectance.as_ref(), ctx.texture_context(), lambda)
            .clamp(0.0, 1.0);
        DiffuseBxDF::new(r)
    }

    pub fn create(mp: &TextureParameterDictionary) -> Result<DiffuseMaterial, PbrtError> {
        let reflectance = mp
            .get_spectrum_texture_or_null_typed("reflectance", SpectrumType::Albedo)?
            .unwrap_or_else(|| {
                Arc::new(SpectrumTexture::Constant(ConstantTexture::new(
                    &Spectrum::from(0.5),
                )))
            });
        let displacement = mp.get_float_texture_or_null("displacement")?;
        let normal_map = get_normal_map(mp)?;
        Ok(DiffuseMaterial::new(
            &reflectance,
            &displacement,
            &normal_map,
        ))
    }
}
