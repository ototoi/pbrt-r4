use super::material_eval_context::MaterialEvalContext;
use super::normal_bump::{apply_normal_or_bump, get_normal_map, NormalMap};
use crate::bxdfs::CoatedConductorBxDF;
use crate::interaction::SurfaceInteraction;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::*;
use crate::util::base::*;
use crate::util::distribution::TrowbridgeReitzDistribution;
use crate::util::error::*;
use crate::util::spectrum::*;

use std::sync::Arc;

pub struct CoatedConductorMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<NormalMap>>,
    interface_u_roughness: Arc<FloatTexture>,
    interface_v_roughness: Arc<FloatTexture>,
    thickness: Arc<FloatTexture>,
    interface_eta: Spectrum,
    g: Arc<FloatTexture>,
    albedo: Arc<SpectrumTexture>,
    conductor_u_roughness: Arc<FloatTexture>,
    conductor_v_roughness: Arc<FloatTexture>,
    conductor_eta: Option<Arc<SpectrumTexture>>,
    k: Option<Arc<SpectrumTexture>>,
    reflectance: Option<Arc<SpectrumTexture>>,
    remap_roughness: bool,
    max_depth: usize,
    n_samples: usize,
}

impl CoatedConductorMaterial {
    pub fn new(
        interface_u_roughness: &Arc<FloatTexture>,
        interface_v_roughness: &Arc<FloatTexture>,
        thickness: &Arc<FloatTexture>,
        interface_eta: Spectrum,
        g: &Arc<FloatTexture>,
        albedo: &Arc<SpectrumTexture>,
        conductor_u_roughness: &Arc<FloatTexture>,
        conductor_v_roughness: &Arc<FloatTexture>,
        conductor_eta: &Option<Arc<SpectrumTexture>>,
        k: &Option<Arc<SpectrumTexture>>,
        reflectance: &Option<Arc<SpectrumTexture>>,
        displacement: &Option<Arc<FloatTexture>>,
        normal_map: &Option<Arc<NormalMap>>,
        remap_roughness: bool,
        max_depth: usize,
        n_samples: usize,
    ) -> Self {
        CoatedConductorMaterial {
            displacement: displacement.clone(),
            normal_map: normal_map.clone(),
            interface_u_roughness: interface_u_roughness.clone(),
            interface_v_roughness: interface_v_roughness.clone(),
            thickness: thickness.clone(),
            interface_eta,
            g: g.clone(),
            albedo: albedo.clone(),
            conductor_u_roughness: conductor_u_roughness.clone(),
            conductor_v_roughness: conductor_v_roughness.clone(),
            conductor_eta: conductor_eta.clone(),
            k: k.clone(),
            reflectance: reflectance.clone(),
            remap_roughness,
            max_depth,
            n_samples,
        }
    }

    pub fn apply_displacement(&self, si: &mut SurfaceInteraction) {
        apply_normal_or_bump(&self.normal_map, &self.displacement, si);
    }

    /// Translation of pbrt-v4 `CoatedConductorMaterial::GetBxDF`.
    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> CoatedConductorBxDF {
        let texture_ctx = ctx.texture_context();
        let (top_u, top_v) = roughness_pair(
            &self.interface_u_roughness,
            &self.interface_v_roughness,
            tex_eval,
            ctx,
            self.remap_roughness,
        );
        let thickness = tex_eval.evaluate_float(self.thickness.as_ref(), texture_ctx);
        let mut interface_eta = self.interface_eta.sample_at(lambda[0]);
        if interface_eta == 0.0 {
            interface_eta = 1.0;
        }

        let (mut conductor_eta, mut conductor_k) = if let Some(conductor_eta) = &self.conductor_eta
        {
            let eta = tex_eval.evaluate_spectrum(conductor_eta.as_ref(), texture_ctx, lambda);
            let k = self.k.as_ref().map_or(SampledSpectrum::zero(), |k| {
                tex_eval.evaluate_spectrum(k.as_ref(), texture_ctx, lambda)
            });
            (eta, k)
        } else {
            debug_assert!(
                self.reflectance.is_some(),
                "reflectance must be set when conductor_eta is absent"
            );
            let reflectance = self
                .reflectance
                .as_ref()
                .expect("reflectance must be set when conductor_eta is absent");
            let r = tex_eval
                .evaluate_spectrum(reflectance.as_ref(), texture_ctx, lambda)
                .clamp(0.0, 0.9999);
            let eta = SampledSpectrum::one();
            let k = 2.0 * r.sqrt() / (SampledSpectrum::one() - r).clamp_zero().sqrt();
            (eta, k)
        };
        conductor_eta /= interface_eta;
        conductor_k /= interface_eta;

        let (cond_u, cond_v) = roughness_pair(
            &self.conductor_u_roughness,
            &self.conductor_v_roughness,
            tex_eval,
            ctx,
            self.remap_roughness,
        );
        let conductor_distribution = TrowbridgeReitzDistribution::new(cond_u, cond_v, true);

        let albedo = tex_eval
            .evaluate_spectrum(self.albedo.as_ref(), texture_ctx, lambda)
            .clamp(0.0, 1.0);
        let g = tex_eval
            .evaluate_float(self.g.as_ref(), texture_ctx)
            .clamp(-1.0, 1.0);

        CoatedConductorBxDF::new(
            interface_eta,
            top_u,
            top_v,
            conductor_distribution,
            conductor_eta,
            conductor_k,
            albedo,
            g,
            thickness,
            self.max_depth,
            self.n_samples,
        )
    }

    pub fn maybe_terminate_secondary_wavelengths(
        &self,
        _si: &SurfaceInteraction,
        lambda: &SampledWavelengths,
    ) -> Option<SampledWavelengths> {
        if !self.interface_eta.is_constant_spectrum() {
            let mut new_lambda = *lambda;
            new_lambda.terminate_secondary();
            Some(new_lambda)
        } else {
            None
        }
    }

    pub fn create(mp: &TextureParameterDictionary) -> Result<CoatedConductorMaterial, PbrtError> {
        let interface_roughness = mp.get_float_texture("interface.roughness", 0.0)?;
        let interface_u_roughness = mp
            .get_float_texture_or_null("interface.uroughness")?
            .unwrap_or_else(|| interface_roughness.clone());
        let interface_v_roughness = mp
            .get_float_texture_or_null("interface.vroughness")?
            .unwrap_or_else(|| interface_roughness.clone());
        let thickness = mp.get_float_texture("thickness", 0.01)?;
        let interface_eta = mp
            .get_spectrum_or_null_typed("interface.eta", SpectrumType::Unbounded)
            .unwrap_or_else(|| Spectrum::from(mp.get_one_float("interface.eta", 1.5)));

        let conductor_roughness = mp.get_float_texture("conductor.roughness", 0.0)?;
        let conductor_u_roughness = mp
            .get_float_texture_or_null("conductor.uroughness")?
            .unwrap_or_else(|| conductor_roughness.clone());
        let conductor_v_roughness = mp
            .get_float_texture_or_null("conductor.vroughness")?
            .unwrap_or_else(|| conductor_roughness.clone());
        let mut conductor_eta =
            mp.get_spectrum_texture_or_null_typed("conductor.eta", SpectrumType::Unbounded)?;
        let mut k =
            mp.get_spectrum_texture_or_null_typed("conductor.k", SpectrumType::Unbounded)?;
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
            .any(|key| mp.params.get_key_name(key) == "conductor.eta");
        let has_k = mp
            .params
            .get_keys()
            .iter()
            .any(|key| mp.params.get_key_name(key) == "conductor.k");

        if has_reflectance && (has_eta || has_k) {
            return Err(PbrtError::error(
                "For the coated conductor material, both \"reflectance\" and \"eta\" and \"k\" can't be provided.",
            ));
        }
        if reflectance.is_none() {
            if conductor_eta.is_none() {
                conductor_eta = Some(Arc::new(
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

        let albedo = mp
            .get_spectrum_texture_or_null_typed("albedo", SpectrumType::Albedo)?
            .unwrap_or_else(|| {
                Arc::new(SpectrumTexture::Constant(ConstantTexture::new(
                    &Spectrum::zero(),
                )))
            });
        let g = mp.get_float_texture("g", 0.0)?;

        let max_depth = mp.get_one_int("maxdepth", 10) as usize;
        let n_samples = mp.get_one_int("nsamples", 1) as usize;
        let remap_roughness = mp.get_one_bool("remaproughness", true);
        let displacement = mp.get_float_texture_or_null("displacement")?;
        let normal_map = get_normal_map(mp)?;

        Ok(CoatedConductorMaterial::new(
            &interface_u_roughness,
            &interface_v_roughness,
            &thickness,
            interface_eta,
            &g,
            &albedo,
            &conductor_u_roughness,
            &conductor_v_roughness,
            &conductor_eta,
            &k,
            &reflectance,
            &displacement,
            &normal_map,
            remap_roughness,
            max_depth,
            n_samples,
        ))
    }
}

fn roughness_pair(
    u_tex: &Arc<FloatTexture>,
    v_tex: &Arc<FloatTexture>,
    tex_eval: &impl TextureEvaluator,
    ctx: &MaterialEvalContext,
    remap: bool,
) -> (Float, Float) {
    let texture_ctx = ctx.texture_context();
    let mut u = tex_eval.evaluate_float(u_tex.as_ref(), texture_ctx);
    let mut v = tex_eval.evaluate_float(v_tex.as_ref(), texture_ctx);
    if remap {
        u = TrowbridgeReitzDistribution::roughness_to_alpha(u);
        v = TrowbridgeReitzDistribution::roughness_to_alpha(v);
    }
    (u, v)
}
