use super::material_eval_context::MaterialEvalContext;
use super::normal_bump::{apply_normal_or_bump, get_normal_map, NormalMap};
use crate::bssrdf::{
    compute_beam_diffusion_bssrdf, subsurface_from_diffuse_sampled, BSSRDFTable, TabulatedBSSRDF,
};
use crate::bxdfs::DielectricBxDF;
use crate::interaction::SurfaceInteraction;
use crate::media::get_medium_scattering_properties;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::*;
use crate::util::base::*;
use crate::util::distribution::TrowbridgeReitzDistribution;
use crate::util::error::*;
use crate::util::spectrum::*;

use std::sync::Arc;

pub struct SubsurfaceMaterial {
    scale: Float,
    sigma_a: Option<Arc<SpectrumTexture>>,
    sigma_s: Option<Arc<SpectrumTexture>>,
    reflectance: Option<Arc<SpectrumTexture>>,
    mfp: Option<Arc<SpectrumTexture>>,
    eta: Float,
    u_roughness: Arc<FloatTexture>,
    v_roughness: Arc<FloatTexture>,
    remap_roughness: bool,
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<NormalMap>>,
    table: Arc<BSSRDFTable>,
}

impl SubsurfaceMaterial {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scale: Float,
        sigma_a: &Option<Arc<SpectrumTexture>>,
        sigma_s: &Option<Arc<SpectrumTexture>>,
        reflectance: &Option<Arc<SpectrumTexture>>,
        mfp: &Option<Arc<SpectrumTexture>>,
        g: Float,
        eta: Float,
        u_roughness: &Arc<FloatTexture>,
        v_roughness: &Arc<FloatTexture>,
        remap_roughness: bool,
        displacement: &Option<Arc<FloatTexture>>,
        normal_map: &Option<Arc<NormalMap>>,
    ) -> Self {
        let mut table = BSSRDFTable::new(100, 64);
        compute_beam_diffusion_bssrdf(g, eta, &mut table);

        Self {
            scale,
            sigma_a: sigma_a.clone(),
            sigma_s: sigma_s.clone(),
            reflectance: reflectance.clone(),
            mfp: mfp.clone(),
            eta,
            u_roughness: u_roughness.clone(),
            v_roughness: v_roughness.clone(),
            remap_roughness,
            displacement: displacement.clone(),
            normal_map: normal_map.clone(),
            table: Arc::new(table),
        }
    }

    pub fn apply_displacement(&self, si: &mut SurfaceInteraction) {
        apply_normal_or_bump(&self.normal_map, &self.displacement, si);
    }

    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        _lambda: &SampledWavelengths,
    ) -> DielectricBxDF {
        let texture_ctx = ctx.texture_context();
        let mut u_rough = tex_eval
            .evaluate_float(self.u_roughness.as_ref(), texture_ctx)
            .clamp(0.0, 1.0);
        let mut v_rough = tex_eval
            .evaluate_float(self.v_roughness.as_ref(), texture_ctx)
            .clamp(0.0, 1.0);
        if self.remap_roughness {
            u_rough = TrowbridgeReitzDistribution::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitzDistribution::roughness_to_alpha(v_rough);
        }
        let distrib = TrowbridgeReitzDistribution::new(u_rough, v_rough, true);
        DielectricBxDF::new(self.eta, distrib)
    }

    /// pbrt-v4 `SubsurfaceMaterial::GetBSSRDF` (materials.h:747-765) —
    /// evaluate `sigma_a` / `sigma_s` (or reflectance + mfp) at the
    /// current pixel's wavelengths, returning a `SampledSpectrum` pair
    /// instead of allocating a `DenselySampledSpectrum` per shade.
    pub fn get_bssrdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> Option<TabulatedBSSRDF> {
        let (sigma_a, sigma_s) = self.evaluate_scattering(tex_eval, ctx, lambda);
        if (sigma_a + sigma_s).is_black() {
            return None;
        }

        Some(TabulatedBSSRDF::new(
            ctx.texture_ctx.p,
            ctx.ns,
            ctx.wo,
            self.eta,
            sigma_a,
            sigma_s,
            self.table.clone(),
        ))
    }

    fn evaluate_scattering<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> (SampledSpectrum, SampledSpectrum) {
        let texture_ctx = ctx.texture_context();
        if let (Some(sigma_a), Some(sigma_s)) = (&self.sigma_a, &self.sigma_s) {
            let sigma_a = (tex_eval.evaluate_spectrum(sigma_a.as_ref(), texture_ctx, lambda)
                * self.scale)
                .clamp_zero();
            let sigma_s = (tex_eval.evaluate_spectrum(sigma_s.as_ref(), texture_ctx, lambda)
                * self.scale)
                .clamp_zero();
            return (sigma_a, sigma_s);
        }

        let reflectance = self
            .reflectance
            .as_ref()
            .map(|tex| {
                tex_eval
                    .evaluate_spectrum(tex.as_ref(), texture_ctx, lambda)
                    .clamp(0.0, 1.0)
            })
            .unwrap_or_else(SampledSpectrum::zero);
        let mfp = self
            .mfp
            .as_ref()
            .map(|tex| {
                (tex_eval.evaluate_spectrum(tex.as_ref(), texture_ctx, lambda) * self.scale)
                    .clamp_zero()
            })
            .unwrap_or_else(SampledSpectrum::one);
        subsurface_from_diffuse_sampled(&self.table, &reflectance, &mfp)
    }

    pub fn create(mp: &TextureParameterDictionary) -> Result<SubsurfaceMaterial, PbrtError> {
        let g_default = mp.get_one_float("g", 0.0);
        let name = mp.get_one_string("name", "");
        let (sigma_a, sigma_s, reflectance, mfp, g) = if !name.is_empty() {
            if let Some((preset_sigma_a, preset_sigma_s)) = get_medium_scattering_properties(&name)
            {
                if g_default != 0.0 {
                    log::warn!(
                        "Material \"subsurface\": non-zero g ignored with named preset \"{}\".",
                        name
                    );
                }
                (
                    Some(constant_spectrum_texture(preset_sigma_a)),
                    Some(constant_spectrum_texture(preset_sigma_s)),
                    None,
                    None,
                    0.0,
                )
            } else {
                return Err(PbrtError::error(&format!(
                    "Material \"subsurface\": preset \"{}\" was not found.",
                    name
                )));
            }
        } else {
            let sigma_a =
                mp.get_spectrum_texture_or_null_typed("sigma_a", SpectrumType::Unbounded)?;
            let sigma_s =
                mp.get_spectrum_texture_or_null_typed("sigma_s", SpectrumType::Unbounded)?;
            if sigma_a.is_some() ^ sigma_s.is_some() {
                return Err(PbrtError::error(
                "Material \"subsurface\": both \"sigma_a\" and \"sigma_s\" are required together.",
            ));
            }

            if sigma_a.is_none() && sigma_s.is_none() {
                let reflectance =
                    mp.get_spectrum_texture_or_null_typed("reflectance", SpectrumType::Albedo)?;
                if reflectance.is_some() {
                    (
                        None,
                        None,
                        reflectance,
                        Some(mp.get_spectrum_texture_typed(
                            "mfp",
                            &Spectrum::one(),
                            SpectrumType::Unbounded,
                        )?),
                        g_default,
                    )
                } else {
                    (
                        Some(constant_spectrum_texture(Spectrum::from([
                            0.0011, 0.0024, 0.014,
                        ]))),
                        Some(constant_spectrum_texture(Spectrum::from([
                            2.55, 3.21, 3.77,
                        ]))),
                        None,
                        None,
                        g_default,
                    )
                }
            } else {
                (sigma_a, sigma_s, None, None, g_default)
            }
        };

        let mut eta = mp
            .get_spectrum_or_null_typed("eta", SpectrumType::Unbounded)
            .map(|eta_spec| {
                let lambda = SampledWavelengths::sample_visible(0.5);
                eta_from_spectrum(eta_spec.clamp_zero(), &lambda, 1.33)
            })
            .unwrap_or_else(|| mp.get_one_float("eta", 1.33));
        if eta <= 0.0 {
            log::warn!(
                "Material \"subsurface\": invalid eta {}. Falling back to 1.33",
                eta
            );
            eta = 1.33;
        }

        let scale = mp.get_one_float("scale", 1.0);
        let roughness = mp.get_float_texture("roughness", 0.0)?;
        let u_roughness = mp
            .get_float_texture_or_null("uroughness")?
            .unwrap_or_else(|| roughness.clone());
        let v_roughness = mp
            .get_float_texture_or_null("vroughness")?
            .unwrap_or_else(|| roughness.clone());
        let remap_roughness = mp.get_one_bool("remaproughness", true);
        let displacement = mp.get_float_texture_or_null("displacement")?;
        let normal_map = get_normal_map(mp)?;

        Ok(SubsurfaceMaterial::new(
            scale,
            &sigma_a,
            &sigma_s,
            &reflectance,
            &mfp,
            g,
            eta,
            &u_roughness,
            &v_roughness,
            remap_roughness,
            &displacement,
            &normal_map,
        ))
    }
}

fn constant_spectrum_texture(value: Spectrum) -> Arc<SpectrumTexture> {
    Arc::new(SpectrumTexture::Constant(ConstantTexture::new(&value)))
}
