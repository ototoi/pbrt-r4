use super::material_eval_context::MaterialEvalContext;
use super::normal_bump::{apply_normal_or_bump, get_normal_map, NormalMap};
use crate::bxdfs::{MeasuredBxDF, MeasuredBxDFData};
use crate::interaction::SurfaceInteraction;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::*;
use crate::util::error::*;
use crate::util::spectrum::SampledWavelengths;

use std::sync::Arc;

pub struct MeasuredMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<NormalMap>>,
    data: Arc<MeasuredBxDFData>,
}

impl MeasuredMaterial {
    pub fn new(
        filename: String,
        displacement: &Option<Arc<FloatTexture>>,
        normal_map: &Option<Arc<NormalMap>>,
    ) -> Self {
        MeasuredMaterial {
            displacement: displacement.clone(),
            normal_map: normal_map.clone(),
            data: MeasuredBxDFData::from_file(&filename),
        }
    }

    pub fn apply_displacement(&self, si: &mut SurfaceInteraction) {
        apply_normal_or_bump(&self.normal_map, &self.displacement, si);
    }

    /// Translation of pbrt-v4 `MeasuredMaterial::GetBxDF`: pass through
    /// the cached tabulated data plus the current SampledWavelengths.
    pub fn get_bxdf<E: TextureEvaluator>(
        &self,
        _tex_eval: &E,
        _ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> MeasuredBxDF {
        MeasuredBxDF::new(self.data.clone(), lambda)
    }

    pub fn filename(&self) -> &str {
        &self.data.filename
    }
    pub fn create(mp: &TextureParameterDictionary) -> Result<MeasuredMaterial, PbrtError> {
        let filename = mp.get_one_string("filename", "");
        if filename.is_empty() {
            return Err(PbrtError::error(
                "Filename must be provided for MeasuredMaterial",
            ));
        }
        let displacement = mp.get_float_texture_or_null("displacement")?;
        let normal_map = get_normal_map(mp)?;
        Ok(MeasuredMaterial::new(filename, &displacement, &normal_map))
    }
}
