use super::lightdistrib::*;
use super::power::*;
use super::spatial::*;
use super::uniform::*;
use crate::cpu::integrators::IntegratorBase;
use crate::util::error::*;

use log::*;
use std::sync::Arc;

pub fn create_light_sample_distribution(
    name: &str,
    base: &IntegratorBase,
) -> Result<Arc<dyn LightDistribution>, PbrtError> {
    if base.lights.is_empty() {
        let msg = format!(
            "Light sample distribution type \"{}\" cannot create since no light.",
            name
        );
        return Err(PbrtError::error(&msg));
    }
    match name {
        "uniform" => {
            if base.lights.len() != 1 {
                return create_light_sample_distribution("spatial", base);
            } else {
                return Ok(Arc::new(UniformLightDistribution::new(base)));
            }
        }
        "power" => {
            return Ok(Arc::new(PowerLightDistribution::new(base)));
        }
        "spatial" => {
            let max_voxels = 64;
            return Ok(Arc::new(SpatialLightDistribution::new(base, max_voxels)));
        }
        s => {
            warn!(
                "Light sample distribution type \"{}\" unknown. Using \"spatial\".",
                s
            );
            return create_light_sample_distribution("spatial", base);
        }
    }
}
