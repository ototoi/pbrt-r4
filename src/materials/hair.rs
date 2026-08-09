use super::material_eval_context::MaterialEvalContext;
use crate::bxdfs::HairBxDF;
use crate::interaction::SurfaceInteraction;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::spectrum::*;

use std::sync::Arc;

pub struct HairMaterial {
    sigma_a: Option<Arc<SpectrumTexture>>,
    color: Option<Arc<SpectrumTexture>>,
    eumelanin: Option<Arc<FloatTexture>>,
    pheomelanin: Option<Arc<FloatTexture>>,
    eta: Arc<FloatTexture>,
    beta_m: Arc<FloatTexture>,
    beta_n: Arc<FloatTexture>,
    alpha: Arc<FloatTexture>,
}

impl HairMaterial {
    pub fn new(
        sigma_a: &Option<Arc<SpectrumTexture>>,
        color: &Option<Arc<SpectrumTexture>>,
        eumelanin: &Option<Arc<FloatTexture>>,
        pheomelanin: &Option<Arc<FloatTexture>>,
        eta: &Arc<FloatTexture>,
        beta_m: &Arc<FloatTexture>,
        beta_n: &Arc<FloatTexture>,
        alpha: &Arc<FloatTexture>,
    ) -> Self {
        HairMaterial {
            sigma_a: sigma_a.clone(),
            color: color.clone(),
            eumelanin: eumelanin.clone(),
            pheomelanin: pheomelanin.clone(),
            eta: eta.clone(),
            beta_m: beta_m.clone(),
            beta_n: beta_n.clone(),
            alpha: alpha.clone(),
        }
    }

    pub fn apply_displacement(&self, _si: &mut SurfaceInteraction) {}

    /// Translation of pbrt-v4 `HairMaterial::GetBxDF`. sigma_a is built
    /// directly as a `SampledSpectrum` at `lambda`.
    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> HairBxDF {
        let texture_ctx = ctx.texture_context();
        let eta = self.eval_eta(tex_eval, ctx);
        let beta_m = tex_eval
            .evaluate_float(self.beta_m.as_ref(), texture_ctx)
            .clamp(1e-2, 1.0);
        let beta_n = tex_eval
            .evaluate_float(self.beta_n.as_ref(), texture_ctx)
            .clamp(1e-2, 1.0);
        let alpha = tex_eval.evaluate_float(self.alpha.as_ref(), texture_ctx);
        let sigma_a = self.eval_sigma_a(tex_eval, ctx, beta_n, lambda);
        let h = -1.0 + 2.0 * ctx.texture_ctx.uv[1];
        HairBxDF::new(h, eta, sigma_a, beta_m, beta_n, alpha)
    }

    fn eval_sigma_a<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        beta_n: Float,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        let texture_ctx = ctx.texture_context();
        if let Some(sigma_a) = &self.sigma_a {
            return tex_eval
                .evaluate_spectrum(sigma_a.as_ref(), texture_ctx, lambda)
                .clamp_zero();
        }
        if let Some(color) = &self.color {
            return HairBxDF::sigma_a_from_reflectance(
                tex_eval
                    .evaluate_spectrum(color.as_ref(), texture_ctx, lambda)
                    .clamp(0.0, 1.0),
                beta_n,
            );
        }

        let ce = self
            .eumelanin
            .as_ref()
            .map(|t| tex_eval.evaluate_float(t.as_ref(), texture_ctx).max(0.0))
            .unwrap_or(0.0);
        let cp = self
            .pheomelanin
            .as_ref()
            .map(|t| tex_eval.evaluate_float(t.as_ref(), texture_ctx).max(0.0))
            .unwrap_or(0.0);
        HairBxDF::sigma_a_from_concentration(ce, cp, lambda)
    }

    fn eval_eta<E: TextureEvaluator>(&self, tex_eval: &E, ctx: &MaterialEvalContext) -> Float {
        tex_eval.evaluate_float(self.eta.as_ref(), ctx.texture_context())
    }
    pub fn create(mp: &TextureParameterDictionary) -> Result<HairMaterial, PbrtError> {
        let sigma_a_param =
            mp.get_spectrum_texture_or_null_typed("sigma_a", SpectrumType::Unbounded)?;
        let color_param =
            mp.get_spectrum_texture_or_null_typed("reflectance", SpectrumType::Albedo)?;
        let color_param = color_param.or_else(|| {
            mp.get_spectrum_texture_or_null_typed("color", SpectrumType::Albedo)
                .ok()
                .flatten()
        });
        let eumelanin_param = mp.get_float_texture_or_null("eumelanin")?;
        let pheomelanin_param = mp.get_float_texture_or_null("pheomelanin")?;

        let (sigma_a, color, eumelanin, pheomelanin) = if let Some(sigma_a) = sigma_a_param {
            if color_param.is_some() {
                log::warn!(
                "Material \"hair\": ignoring \"reflectance\"/\"color\" because \"sigma_a\" was provided."
            );
            }
            if eumelanin_param.is_some() {
                log::warn!(
                    "Material \"hair\": ignoring \"eumelanin\" because \"sigma_a\" was provided."
                );
            }
            if pheomelanin_param.is_some() {
                log::warn!(
                    "Material \"hair\": ignoring \"pheomelanin\" because \"sigma_a\" was provided."
                );
            }
            (Some(sigma_a), None, None, None)
        } else if let Some(color) = color_param {
            if eumelanin_param.is_some() {
                log::warn!(
                "Material \"hair\": ignoring \"eumelanin\" because \"reflectance\"/\"color\" was provided."
            );
            }
            if pheomelanin_param.is_some() {
                log::warn!(
                "Material \"hair\": ignoring \"pheomelanin\" because \"reflectance\"/\"color\" was provided."
            );
            }
            (None, Some(color), None, None)
        } else if eumelanin_param.is_some() || pheomelanin_param.is_some() {
            (None, None, eumelanin_param, pheomelanin_param)
        } else {
            let sigma_a = Arc::new(SpectrumTexture::Constant(ConstantTexture::new(
                &sigma_a_from_concentration_spectrum(1.3, 0.0),
            )));
            (Some(sigma_a), None, None, None)
        };

        let eta = mp.get_float_texture("eta", 1.55)?;
        let beta_m = mp.get_float_texture("beta_m", 0.3)?;
        let beta_n = mp.get_float_texture("beta_n", 0.3)?;
        let alpha = mp.get_float_texture("alpha", 2.0)?;
        Ok(HairMaterial::new(
            &sigma_a,
            &color,
            &eumelanin,
            &pheomelanin,
            &eta,
            &beta_m,
            &beta_n,
            &alpha,
        ))
    }
}

fn sigma_a_from_concentration_spectrum(eumelanin: Float, pheomelanin: Float) -> Spectrum {
    let ce = eumelanin.max(0.0);
    let cp = pheomelanin.max(0.0);
    let eumelanin_rgb = [0.419, 0.697, 1.37];
    let pheomelanin_rgb = [0.187, 0.4, 1.05];
    Spectrum::from_rgb_unbounded(&[
        ce * eumelanin_rgb[0] + cp * pheomelanin_rgb[0],
        ce * eumelanin_rgb[1] + cp * pheomelanin_rgb[1],
        ce * eumelanin_rgb[2] + cp * pheomelanin_rgb[2],
    ])
}
