use super::material_eval_context::MaterialEvalContext;
use super::normal_bump::{apply_normal_or_bump, get_normal_map, NormalMap};
use crate::bxdfs::DielectricBxDF;
use crate::interaction::SurfaceInteraction;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::*;
use crate::util::distribution::TrowbridgeReitzDistribution;
use crate::util::error::*;
use crate::util::spectrum::*;

use std::sync::Arc;

/// pbrt-v4 `DielectricMaterial` (materials.h:141) — stores `eta`
/// directly as a `Spectrum` and samples it at `lambda[0]` per shade.
pub struct DielectricMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<NormalMap>>,
    u_roughness: Arc<FloatTexture>,
    v_roughness: Arc<FloatTexture>,
    remap_roughness: bool,
    eta: Spectrum,
}

impl DielectricMaterial {
    pub fn new(
        displacement: &Option<Arc<FloatTexture>>,
        normal_map: &Option<Arc<NormalMap>>,
        u_roughness: &Arc<FloatTexture>,
        v_roughness: &Arc<FloatTexture>,
        remap_roughness: bool,
        eta: Spectrum,
    ) -> Self {
        DielectricMaterial {
            displacement: displacement.clone(),
            normal_map: normal_map.clone(),
            u_roughness: u_roughness.clone(),
            v_roughness: v_roughness.clone(),
            remap_roughness,
            eta,
        }
    }

    pub fn apply_displacement(&self, si: &mut SurfaceInteraction) {
        apply_normal_or_bump(&self.normal_map, &self.displacement, si);
    }

    /// pbrt-v4 `DielectricMaterial::GetBxDF` (materials.h:181-203) —
    /// `sampledEta = eta(lambda[0])`, `TerminateSecondary` if eta isn't
    /// constant, edge-case default of 1 if the hero wavelength is
    /// outside the stored spectrum.
    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> DielectricBxDF {
        let texture_ctx = ctx.texture_context();
        let mut u_rough = tex_eval.evaluate_float(self.u_roughness.as_ref(), texture_ctx);
        let mut v_rough = tex_eval.evaluate_float(self.v_roughness.as_ref(), texture_ctx);
        if self.remap_roughness {
            u_rough = TrowbridgeReitzDistribution::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitzDistribution::roughness_to_alpha(v_rough);
        }
        let distrib = TrowbridgeReitzDistribution::new(u_rough, v_rough, true);
        let sampled_eta = self.eta.sample(lambda)[0];
        let eta = if sampled_eta > 0.0 { sampled_eta } else { 1.0 };
        DielectricBxDF::new(eta, distrib)
    }

    /// pbrt-v4 inlines `if (!eta.template Is<ConstantSpectrum>())
    /// lambda.TerminateSecondary();` in `GetBxDF`. We keep a separate
    /// hook here so `SurfaceInteraction::get_bsdf` can mutate `lambda`
    /// before constructing the BSDF.
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

    pub fn create(mp: &TextureParameterDictionary) -> Result<DielectricMaterial, PbrtError> {
        let roughness = mp.get_float_texture("roughness", 0.0)?;
        let u_roughness = mp
            .get_float_texture_or_null("uroughness")?
            .unwrap_or_else(|| roughness.clone());
        let v_roughness = mp
            .get_float_texture_or_null("vroughness")?
            .unwrap_or_else(|| roughness.clone());
        let eta_array = mp.get_float_array("eta");
        let mut eta = if eta_array.is_empty() {
            if let Some(spectrum) = mp.get_spectrum_or_null_typed("eta", SpectrumType::Unbounded) {
                spectrum
            } else if mp.has_texture_name("eta") {
                return Err(PbrtError::error(
                    "Material \"dielectric\": couldn't find spectrum texture named \"eta\".",
                ));
            } else {
                Spectrum::from(1.5)
            }
        } else {
            Spectrum::from(eta_array[0])
        };
        let probe_lambda = SampledWavelengths::sample_visible(0.5);
        if eta_from_spectrum(eta.clamp_zero(), &probe_lambda, -1.0) <= 0.0 {
            log::warn!(
                "Material \"dielectric\": invalid eta {:?}. Falling back to 1.5",
                eta
            );
            eta = Spectrum::from(1.5);
        }
        let displacement = mp.get_float_texture_or_null("displacement")?;
        let normal_map = get_normal_map(mp)?;
        let remap_roughness = mp.get_one_bool("remaproughness", true);
        Ok(DielectricMaterial::new(
            &displacement,
            &normal_map,
            &u_roughness,
            &v_roughness,
            remap_roughness,
            eta,
        ))
    }
}
