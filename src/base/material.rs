use crate::base::bxdf::BxDF;
use crate::bsdf::BSDF;
use crate::interaction::SurfaceInteraction;
use crate::paramdict::*;
use crate::util::base::Float;
use crate::util::error::*;
use crate::util::spectrum::SampledWavelengths;

use super::bssrdf::BSSRDF;
use crate::materials::CoatedConductorMaterial;
use crate::materials::CoatedDiffuseMaterial;
use crate::materials::ConductorMaterial;
use crate::materials::DielectricMaterial;
use crate::materials::DiffuseMaterial;
use crate::materials::DiffuseTransmissionMaterial;
use crate::materials::HairMaterial;
use crate::materials::InterfaceMaterial;
use crate::materials::MaterialEvalContext;
use crate::materials::MeasuredMaterial;
use crate::materials::MixMaterial;
use crate::materials::SubsurfaceMaterial;
use crate::materials::ThinDielectricMaterial;
use crate::textures::{TextureEvaluator, UniversalTextureEvaluator};

/// Material enum that unifies all material types
///
/// This enum-based approach replaces the dynamic trait object pattern.
/// This corresponds to pbrt-v4's `TaggedPointer` material dispatch.
pub enum Material {
    CoatedDiffuse(CoatedDiffuseMaterial),
    CoatedConductor(CoatedConductorMaterial),
    Diffuse(DiffuseMaterial),
    DiffuseTransmission(DiffuseTransmissionMaterial),
    Conductor(ConductorMaterial),
    Dielectric(DielectricMaterial),
    ThinDielectric(ThinDielectricMaterial),
    Hair(HairMaterial),
    Interface(InterfaceMaterial),
    Measured(MeasuredMaterial),
    Mix(MixMaterial),
    Subsurface(SubsurfaceMaterial),
}

impl Material {
    pub fn diagnostic_id(&self) -> Float {
        match self {
            Self::Diffuse(_) => 1.0,
            Self::CoatedDiffuse(_) => 2.0,
            Self::Subsurface(_) => 3.0,
            Self::DiffuseTransmission(_) => 4.0,
            Self::Conductor(_) => 5.0,
            Self::Dielectric(_) => 6.0,
            Self::ThinDielectric(_) => 7.0,
            Self::Hair(_) => 8.0,
            Self::Interface(_) => 9.0,
            Self::Measured(_) => 10.0,
            Self::Mix(_) => 11.0,
            Self::CoatedConductor(_) => 12.0,
        }
    }
    /// Create a material from name and parameters
    ///
    /// Corresponds to pbrt-v4's Material::Create
    ///
    /// # Arguments
    /// * `name` - Material type name (diffuse, conductor, dielectric, etc.)
    /// * `params` - Material parameters
    ///
    /// # Returns
    /// * `Result<Material, PbrtError>` - Created material
    pub fn create(name: &str, params: &TextureParameterDictionary) -> Result<Material, PbrtError> {
        // Handle deprecated/special cases
        if name.is_empty() || name == "none" {
            log::warn!("Material \"{}\" is deprecated; using \"interface\".", name);
            return Ok(Material::Interface(InterfaceMaterial::create(params)?));
        }

        if name == "interface" {
            return Ok(Material::Interface(InterfaceMaterial::create(params)?));
        }

        if let Some(hint) = legacy_material_hint(name) {
            return Err(PbrtError::error(&format!(
                "Material \"{}\" is a legacy material name and is not accepted by the runtime. {}",
                name, hint
            )));
        }

        // Create material based on type
        let material = match name {
            "diffuse" => Material::Diffuse(DiffuseMaterial::create(params)?),
            "coateddiffuse" => Material::CoatedDiffuse(CoatedDiffuseMaterial::create(params)?),
            "coatedconductor" => {
                Material::CoatedConductor(CoatedConductorMaterial::create(params)?)
            }
            "diffusetransmission" => {
                Material::DiffuseTransmission(DiffuseTransmissionMaterial::create(params)?)
            }
            "dielectric" => Material::Dielectric(DielectricMaterial::create(params)?),
            "thindielectric" => Material::ThinDielectric(ThinDielectricMaterial::create(params)?),
            "hair" => Material::Hair(HairMaterial::create(params)?),
            "interface" => Material::Interface(InterfaceMaterial::create(params)?),
            "conductor" => Material::Conductor(ConductorMaterial::create(params)?),
            "measured" => Material::Measured(MeasuredMaterial::create(params)?),
            "subsurface" => Material::Subsurface(SubsurfaceMaterial::create(params)?),
            "mix" => {
                return Err(PbrtError::error(
                    "Material \"mix\" requires named-material resolution by the scene builder.",
                ));
            }
            _ => {
                return Err(PbrtError::error(&format!(
                    "{}: material type unknown.",
                    name
                )));
            }
        };

        Ok(material)
    }

    pub fn get_bsdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, bxdf)
    }

    pub fn apply_displacement(&self, si: &mut SurfaceInteraction) {
        match self {
            Material::CoatedDiffuse(m) => m.apply_displacement(si),
            Material::CoatedConductor(m) => m.apply_displacement(si),
            Material::Diffuse(m) => m.apply_displacement(si),
            Material::DiffuseTransmission(m) => m.apply_displacement(si),
            Material::Conductor(m) => m.apply_displacement(si),
            Material::Dielectric(m) => m.apply_displacement(si),
            Material::Hair(m) => m.apply_displacement(si),
            Material::Interface(m) => m.apply_displacement(si),
            Material::ThinDielectric(m) => m.apply_displacement(si),
            Material::Measured(m) => m.apply_displacement(si),
            Material::Mix(m) => {
                let tex_eval = UniversalTextureEvaluator;
                let ctx = MaterialEvalContext::from(&*si);
                m.choose_material(&tex_eval, &ctx)
                    .as_ref()
                    .apply_displacement(si)
            }
            Material::Subsurface(m) => m.apply_displacement(si),
        }
    }

    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> BxDF {
        match self {
            Material::CoatedDiffuse(m) => {
                BxDF::CoatedDiffuse(Box::new(m.get_bxdf(tex_eval, ctx, lambda)))
            }
            Material::CoatedConductor(m) => {
                BxDF::CoatedConductor(Box::new(m.get_bxdf(tex_eval, ctx, lambda)))
            }
            Material::Diffuse(m) => BxDF::Diffuse(Box::new(m.get_bxdf(tex_eval, ctx, lambda))),
            Material::DiffuseTransmission(m) => {
                BxDF::DiffuseTransmission(Box::new(m.get_bxdf(tex_eval, ctx, lambda)))
            }
            Material::Conductor(m) => BxDF::Conductor(Box::new(m.get_bxdf(tex_eval, ctx, lambda))),
            Material::Dielectric(m) => {
                BxDF::Dielectric(Box::new(m.get_bxdf(tex_eval, ctx, lambda)))
            }
            Material::Hair(m) => BxDF::Hair(Box::new(m.get_bxdf(tex_eval, ctx, lambda))),
            Material::Interface(m) => BxDF::Dielectric(Box::new(m.get_bxdf(tex_eval, ctx, lambda))),
            Material::ThinDielectric(m) => {
                BxDF::ThinDielectric(Box::new(m.get_bxdf(tex_eval, ctx, lambda)))
            }
            Material::Measured(m) => BxDF::Measured(Box::new(m.get_bxdf(tex_eval, ctx, lambda))),
            Material::Mix(m) => m
                .choose_material(tex_eval, ctx)
                .as_ref()
                .get_bxdf(tex_eval, ctx, lambda),
            Material::Subsurface(m) => {
                BxDF::Dielectric(Box::new(m.get_bxdf(tex_eval, ctx, lambda)))
            }
        }
    }

    pub fn get_bssrdf<E: TextureEvaluator>(
        &self,
        tex_eval: &E,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> Option<BSSRDF> {
        match self {
            Material::Mix(m) => m
                .choose_material(tex_eval, ctx)
                .as_ref()
                .get_bssrdf(tex_eval, ctx, lambda),
            Material::Subsurface(m) => m.get_bssrdf(tex_eval, ctx, lambda).map(BSSRDF::Tabulated),
            _ => None,
        }
    }

    pub fn maybe_terminate_secondary_wavelengths(
        &self,
        si: &SurfaceInteraction,
        lambda: &SampledWavelengths,
    ) -> Option<SampledWavelengths> {
        match self {
            Material::CoatedDiffuse(m) => m.maybe_terminate_secondary_wavelengths(si, lambda),
            Material::CoatedConductor(m) => m.maybe_terminate_secondary_wavelengths(si, lambda),
            Material::Dielectric(m) => m.maybe_terminate_secondary_wavelengths(si, lambda),
            Material::ThinDielectric(m) => m.maybe_terminate_secondary_wavelengths(si, lambda),
            Material::Mix(m) => {
                let tex_eval = UniversalTextureEvaluator;
                let ctx = MaterialEvalContext::from(si);
                m.choose_material(&tex_eval, &ctx)
                    .as_ref()
                    .maybe_terminate_secondary_wavelengths(si, lambda)
            }
            _ => None,
        }
    }
}

fn legacy_material_hint(name: &str) -> Option<&'static str> {
    match name {
        "uber" => {
            Some("Use the v4 input upgrade path to convert it to \"coateddiffuse\" or \"diffuse\".")
        }
        "matte" => Some("Use the v4 input upgrade path to convert it to \"diffuse\"."),
        "kdsubsurface" => Some("Use the v4 input upgrade path to convert it to \"subsurface\"."),
        "plastic" | "substrate" => {
            Some("Use the v4 input upgrade path to convert it to \"coateddiffuse\".")
        }
        "translucent" => {
            Some("Use the v4 input upgrade path to convert it to \"diffusetransmission\".")
        }
        "mirror" | "metal" => Some("Use the v4 input upgrade path to convert it to \"conductor\"."),
        "glass" => Some("Use the v4 input upgrade path to convert it to \"dielectric\"."),
        "disney" => Some("Use the v4 input upgrade path to convert it to \"diffuse\"."),
        "fourier" => Some(
            "The v4 runtime no longer supports \"fourier\"; use \"measured\" where applicable.",
        ),
        _ => None,
    }
}
