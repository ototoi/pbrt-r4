use super::material_eval_context::MaterialEvalContext;
use super::normal_bump::{apply_normal_or_bump, get_normal_map, NormalMap};
use crate::bxdfs::CoatedDiffuseBxDF;
use crate::interaction::SurfaceInteraction;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::*;
use crate::util::distribution::TrowbridgeReitzDistribution;
use crate::util::error::*;
use crate::util::spectrum::*;

use std::sync::Arc;

pub struct CoatedDiffuseMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<NormalMap>>,
    reflectance: Arc<SpectrumTexture>,
    u_roughness: Arc<FloatTexture>,
    v_roughness: Arc<FloatTexture>,
    thickness: Arc<FloatTexture>,
    albedo: Arc<SpectrumTexture>,
    g: Arc<FloatTexture>,
    eta: Spectrum,
    remap_roughness: bool,
    max_depth: usize,
    n_samples: usize,
}

impl CoatedDiffuseMaterial {
    pub fn new(
        reflectance: &Arc<SpectrumTexture>,
        u_roughness: &Arc<FloatTexture>,
        v_roughness: &Arc<FloatTexture>,
        thickness: &Arc<FloatTexture>,
        albedo: &Arc<SpectrumTexture>,
        g: &Arc<FloatTexture>,
        eta: Spectrum,
        displacement: &Option<Arc<FloatTexture>>,
        normal_map: &Option<Arc<NormalMap>>,
        remap_roughness: bool,
        max_depth: usize,
        n_samples: usize,
    ) -> Self {
        CoatedDiffuseMaterial {
            displacement: displacement.clone(),
            normal_map: normal_map.clone(),
            reflectance: reflectance.clone(),
            u_roughness: u_roughness.clone(),
            v_roughness: v_roughness.clone(),
            thickness: thickness.clone(),
            albedo: albedo.clone(),
            g: g.clone(),
            eta,
            remap_roughness,
            max_depth,
            n_samples,
        }
    }

    pub fn apply_displacement(&self, si: &mut SurfaceInteraction) {
        apply_normal_or_bump(&self.normal_map, &self.displacement, si);
    }

    /// Translation of pbrt-v4 `CoatedDiffuseMaterial::GetBxDF`
    /// (`materials.h`). Evaluate textures at lambda to SampledSpectrum
    /// and construct the layered BxDF directly with those.
    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> CoatedDiffuseBxDF {
        let texture_ctx = ctx.texture_context();
        let reflectance = tex_eval
            .evaluate_spectrum(self.reflectance.as_ref(), texture_ctx, lambda)
            .clamp(0.0, 1.0);
        let albedo = tex_eval
            .evaluate_spectrum(self.albedo.as_ref(), texture_ctx, lambda)
            .clamp(0.0, 1.0);
        let g = tex_eval
            .evaluate_float(self.g.as_ref(), texture_ctx)
            .clamp(-1.0, 1.0);
        let mut u_rough = tex_eval.evaluate_float(self.u_roughness.as_ref(), texture_ctx);
        let mut v_rough = tex_eval.evaluate_float(self.v_roughness.as_ref(), texture_ctx);
        if self.remap_roughness {
            u_rough = TrowbridgeReitzDistribution::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitzDistribution::roughness_to_alpha(v_rough);
        }
        let thickness = tex_eval.evaluate_float(self.thickness.as_ref(), texture_ctx);
        let mut eta = self.eta.sample_at(lambda[0]);
        if eta == 0.0 {
            eta = 1.0;
        }
        CoatedDiffuseBxDF::new(
            reflectance,
            albedo,
            g,
            eta,
            u_rough,
            v_rough,
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
        if !self.eta.is_constant_spectrum() {
            let mut new_lambda = *lambda;
            new_lambda.terminate_secondary();
            Some(new_lambda)
        } else {
            None
        }
    }
    pub fn create(mp: &TextureParameterDictionary) -> Result<CoatedDiffuseMaterial, PbrtError> {
        let reflectance = mp
            .get_spectrum_texture_or_null_typed("reflectance", SpectrumType::Albedo)?
            .unwrap_or_else(|| {
                Arc::new(SpectrumTexture::Constant(ConstantTexture::new(
                    &Spectrum::from(0.5),
                )))
            });

        let eta = mp
            .get_spectrum_or_null_typed("eta", SpectrumType::Unbounded)
            .unwrap_or_else(|| Spectrum::from(mp.get_one_float("eta", 1.5)));

        let roughness = mp.get_float_texture("roughness", 0.0)?;
        let u_roughness = mp
            .get_float_texture_or_null("uroughness")?
            .unwrap_or_else(|| roughness.clone());
        let v_roughness = mp
            .get_float_texture_or_null("vroughness")?
            .unwrap_or_else(|| roughness.clone());
        let thickness = mp.get_float_texture("thickness", 0.01)?;
        let max_depth = mp.get_one_int("maxdepth", 10) as usize;
        let n_samples = mp.get_one_int("nsamples", 1) as usize;
        let g = mp.get_float_texture("g", 0.0)?;
        let albedo = mp
            .get_spectrum_texture_or_null_typed("albedo", SpectrumType::Albedo)?
            .unwrap_or_else(|| {
                Arc::new(SpectrumTexture::Constant(ConstantTexture::new(
                    &Spectrum::from(0.0),
                )))
            });
        let remap_roughness = mp.get_one_bool("remaproughness", true);
        let displacement = mp.get_float_texture_or_null("displacement")?;
        let normal_map = get_normal_map(mp)?;
        Ok(CoatedDiffuseMaterial::new(
            &reflectance,
            &u_roughness,
            &v_roughness,
            &thickness,
            &albedo,
            &g,
            eta,
            &displacement,
            &normal_map,
            remap_roughness,
            max_depth,
            n_samples,
        ))
    }
}
