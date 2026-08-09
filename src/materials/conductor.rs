use super::material_eval_context::MaterialEvalContext;
use super::normal_bump::{apply_normal_or_bump, get_normal_map, NormalMap};
use crate::bxdfs::ConductorBxDF;
use crate::interaction::SurfaceInteraction;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::*;
use crate::util::distribution::TrowbridgeReitzDistribution;
use crate::util::error::*;
use crate::util::spectrum::*;

use std::sync::Arc;

pub struct ConductorMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<NormalMap>>,
    eta: Option<Arc<SpectrumTexture>>,
    k: Option<Arc<SpectrumTexture>>,
    reflectance: Option<Arc<SpectrumTexture>>,
    u_roughness: Arc<FloatTexture>,
    v_roughness: Arc<FloatTexture>,
    remap_roughness: bool,
}

impl ConductorMaterial {
    pub fn new(
        eta: &Option<Arc<SpectrumTexture>>,
        k: &Option<Arc<SpectrumTexture>>,
        reflectance: &Option<Arc<SpectrumTexture>>,
        u_roughness: &Arc<FloatTexture>,
        v_roughness: &Arc<FloatTexture>,
        displacement: &Option<Arc<FloatTexture>>,
        normal_map: &Option<Arc<NormalMap>>,
        remap_roughness: bool,
    ) -> Self {
        ConductorMaterial {
            displacement: displacement.clone(),
            normal_map: normal_map.clone(),
            eta: eta.clone(),
            k: k.clone(),
            reflectance: reflectance.clone(),
            u_roughness: u_roughness.clone(),
            v_roughness: v_roughness.clone(),
            remap_roughness,
        }
    }

    pub fn apply_displacement(&self, si: &mut SurfaceInteraction) {
        apply_normal_or_bump(&self.normal_map, &self.displacement, si);
    }

    /// Translation of pbrt-v4 `ConductorMaterial::GetBxDF`.
    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> ConductorBxDF {
        let texture_ctx = ctx.texture_context();
        let mut u_rough = tex_eval.evaluate_float(self.u_roughness.as_ref(), texture_ctx);
        let mut v_rough = tex_eval.evaluate_float(self.v_roughness.as_ref(), texture_ctx);
        if self.remap_roughness {
            u_rough = TrowbridgeReitzDistribution::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitzDistribution::roughness_to_alpha(v_rough);
        }
        let distrib = TrowbridgeReitzDistribution::new(u_rough, v_rough, true);
        let (eta, k) = if let Some(eta) = &self.eta {
            let eta = tex_eval
                .evaluate_spectrum(eta.as_ref(), texture_ctx, lambda)
                .clamp_zero();
            let k = self.k.as_ref().map_or(SampledSpectrum::zero(), |k| {
                tex_eval
                    .evaluate_spectrum(k.as_ref(), texture_ctx, lambda)
                    .clamp_zero()
            });
            (eta, k)
        } else {
            debug_assert!(
                self.reflectance.is_some(),
                "reflectance must be set when eta is absent"
            );
            let reflectance = self
                .reflectance
                .as_ref()
                .expect("reflectance must be set when eta is absent");
            let r = tex_eval
                .evaluate_spectrum(reflectance.as_ref(), texture_ctx, lambda)
                .clamp(0.0, 0.9999);
            let eta = SampledSpectrum::one();
            let k = 2.0 * r.sqrt() / (SampledSpectrum::one() - r).clamp_zero().sqrt();
            (eta, k)
        };
        ConductorBxDF::new(distrib, eta, k)
    }

    pub fn create(mp: &TextureParameterDictionary) -> Result<ConductorMaterial, PbrtError> {
        let mut eta = mp.get_spectrum_texture_or_null_typed("eta", SpectrumType::Unbounded)?;
        let mut k = mp.get_spectrum_texture_or_null_typed("k", SpectrumType::Unbounded)?;
        let reflectance =
            mp.get_spectrum_texture_or_null_typed("reflectance", SpectrumType::Albedo)?;

        let has_reflectance = mp
            .params
            .get_keys()
            .iter()
            .any(|key| mp.params.get_key_name(key) == "reflectance");
        let has_eta = mp
            .params
            .get_keys()
            .iter()
            .any(|key| mp.params.get_key_name(key) == "eta");
        let has_k = mp
            .params
            .get_keys()
            .iter()
            .any(|key| mp.params.get_key_name(key) == "k");

        if has_reflectance && (has_eta || has_k) {
            return Err(PbrtError::error(
                "For the conductor material, both \"reflectance\" and \"eta\" and \"k\" can't be provided.",
            ));
        }

        if reflectance.is_none() {
            if eta.is_none() {
                eta = Some(Arc::new(
                    spectrum_texture_from_named_spectrum("metal-Cu-eta").ok_or_else(|| {
                        PbrtError::error("Named spectrum metal-Cu-eta should exist.")
                    })?,
                ));
            }
            if k.is_none() {
                k = Some(Arc::new(
                    spectrum_texture_from_named_spectrum("metal-Cu-k").ok_or_else(|| {
                        PbrtError::error("Named spectrum metal-Cu-k should exist.")
                    })?,
                ));
            }
        }

        let roughness = mp.get_float_texture("roughness", 0.0)?;
        let u_roughness = mp
            .get_float_texture_or_null("uroughness")?
            .unwrap_or_else(|| Arc::clone(&roughness));
        let v_roughness = mp
            .get_float_texture_or_null("vroughness")?
            .unwrap_or_else(|| Arc::clone(&roughness));
        let remap_roughness = mp.get_one_bool("remaproughness", true);

        let displacement = mp.get_float_texture_or_null("displacement")?;
        let normal_map = get_normal_map(mp)?;
        Ok(ConductorMaterial::new(
            &eta,
            &k,
            &reflectance,
            &u_roughness,
            &v_roughness,
            &displacement,
            &normal_map,
            remap_roughness,
        ))
    }
}
