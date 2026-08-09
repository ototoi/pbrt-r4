use super::ao::*;
use super::bdpt::*;
use super::function::*;
use super::light_path::*;
use super::mlt::*;
use super::path::*;
use super::random_walk::*;
use super::simple_path::*;
use super::simple_volpath::*;
use super::sppm::*;
use super::volpath::*;
use crate::base::camera::Camera;
use crate::base::sampler::Sampler;
use crate::cpu::integrators::*;
use crate::paramdict::*;

use crate::scene::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

use std::sync::Arc;
use std::sync::RwLock;

pub fn create_integrator(
    name: &str,
    params: &ParameterDictionary,
    sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    match name {
        "lightpath" => create_light_path_integrator(params, sampler, camera, scene),
        "path" => create_path_integrator(params, sampler, camera, scene),
        "randomwalk" => create_random_walk_integrator(params, sampler, camera, scene),
        "simplepath" => create_simple_path_integrator(params, sampler, camera, scene),
        "simplevolpath" => create_simple_volpath_integrator(params, sampler, camera, scene),
        "volpath" => create_volpath_integrator(params, sampler, camera, scene),
        "bdpt" => create_bdpt_integrator(params, sampler, camera, scene),
        "mlt" => create_mlt_integrator(params, sampler, camera, scene),
        "ambientocclusion" => create_ao_integrator(params, sampler, camera, scene),
        "sppm" => create_sppm_integrator(params, sampler, camera, scene),
        "function" => create_function_integrator(params, sampler, camera, scene),
        "diagnostic" | "depth" => create_diagnostic_integrator(params, sampler, camera, scene),
        _ => {
            let msg = format!("Integrator \"{}\" unknown.", name);
            return Err(PbrtError::error(&msg));
        }
    }
}
