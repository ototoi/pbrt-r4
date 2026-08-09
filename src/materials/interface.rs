use super::material_eval_context::MaterialEvalContext;
use crate::bxdfs::DielectricBxDF;
use crate::interaction::SurfaceInteraction;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::TextureEvaluator;
use crate::util::distribution::TrowbridgeReitzDistribution;
use crate::util::error::*;
use crate::util::spectrum::*;

pub struct InterfaceMaterial {}

impl InterfaceMaterial {
    pub fn new() -> Self {
        InterfaceMaterial {}
    }

    pub fn apply_displacement(&self, _si: &mut SurfaceInteraction) {}

    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        _tex_eval: &E,
        _ctx: &MaterialEvalContext,
        _lambda: &SampledWavelengths,
    ) -> DielectricBxDF {
        // v4's "interface" material returns a unit-eta DielectricBxDF
        // with a smooth (alpha=0) Trowbridge-Reitz distribution — so its
        // BxDF::Sample_f short-circuits to a pure pass-through.
        DielectricBxDF::new(1.0, TrowbridgeReitzDistribution::new(0.0, 0.0, true))
    }

    pub fn create(_mp: &TextureParameterDictionary) -> Result<InterfaceMaterial, PbrtError> {
        Ok(InterfaceMaterial::new())
    }
}
