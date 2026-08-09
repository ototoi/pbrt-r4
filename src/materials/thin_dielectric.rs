use super::material_eval_context::MaterialEvalContext;
use super::normal_bump::{apply_normal_or_bump, get_normal_map, NormalMap};
use crate::bxdfs::ThinDielectricBxDF;
use crate::interaction::SurfaceInteraction;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::spectrum::*;

use std::sync::Arc;

/// pbrt-v4 `ThinDielectricMaterial` (materials.h:215) — `eta` stored
/// as a `Spectrum`, sampled at `lambda[0]` per shade.
pub struct ThinDielectricMaterial {
    eta: Spectrum,
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<NormalMap>>,
}

impl ThinDielectricMaterial {
    pub fn new(
        eta: Spectrum,
        displacement: &Option<Arc<FloatTexture>>,
        normal_map: &Option<Arc<NormalMap>>,
    ) -> Self {
        ThinDielectricMaterial {
            eta,
            displacement: displacement.clone(),
            normal_map: normal_map.clone(),
        }
    }

    pub fn apply_displacement(&self, si: &mut SurfaceInteraction) {
        apply_normal_or_bump(&self.normal_map, &self.displacement, si);
    }

    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        _tex_eval: &E,
        _ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> ThinDielectricBxDF {
        let eta = self.eta_for_wavelengths(lambda);
        let eta = if eta > 0.0 { eta } else { 1.0 };
        ThinDielectricBxDF::new(eta)
    }

    fn eta_for_wavelengths(&self, lambda: &SampledWavelengths) -> Float {
        eta_from_spectrum(self.eta.clamp_zero(), lambda, 1.5)
    }

    pub fn maybe_terminate_secondary_wavelengths(
        &self,
        _si: &SurfaceInteraction,
        lambda: &SampledWavelengths,
    ) -> Option<SampledWavelengths> {
        if !self.eta.is_constant_spectrum() {
            let mut new_lambda = *lambda;
            new_lambda.terminate_secondary();
            Some(new_lambda)
        } else {
            None
        }
    }
    pub fn create(mp: &TextureParameterDictionary) -> Result<ThinDielectricMaterial, PbrtError> {
        let mut eta = mp
            .get_spectrum_or_null_typed("eta", SpectrumType::Unbounded)
            .or_else(|| mp.get_spectrum_or_null_typed("index", SpectrumType::Unbounded))
            .unwrap_or_else(|| Spectrum::from(mp.get_one_float("eta", 1.5)));
        let probe_lambda = SampledWavelengths::sample_visible(0.5);
        if eta_from_spectrum(eta.clamp_zero(), &probe_lambda, -1.0) <= 0.0 {
            log::warn!(
                "Material \"thindielectric\": invalid eta {:?}. Falling back to 1.5",
                eta
            );
            eta = Spectrum::from(1.5);
        }
        let displacement = mp.get_float_texture_or_null("displacement")?;
        let normal_map = get_normal_map(mp)?;
        Ok(ThinDielectricMaterial::new(eta, &displacement, &normal_map))
    }
}
