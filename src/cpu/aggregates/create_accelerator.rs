use super::bvh::*;
use super::exhaustive::*;
use super::kdtree::*;
use crate::cpu::primitive::*;
use crate::paramdict::*;
use crate::util::error::*;

use std::sync::Arc;

pub fn create_accelerator(
    name: &str,
    prims: &[Arc<Primitive>],
    params: &ParameterDictionary,
) -> Result<Primitive, PbrtError> {
    match name {
        "bvh" => create_bvh_accelerator(prims, params),
        "kdtree" => create_kdtree_accelerator(prims, params),
        "exhaustive" => create_exhaustive_accelerator(prims, params),
        _ => Err(PbrtError::error(&format!(
            "Accelerator \"{}\" unknown.",
            name
        ))),
    }
}
