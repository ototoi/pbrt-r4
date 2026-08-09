use super::accel::BVHAccel;
use super::build::*;
use crate::cpu::aggregates::Accel;
use crate::cpu::primitive::*;
use crate::paramdict::*;
use crate::util::error::*;

use std::sync::Arc;

pub fn create_bvh_accelerator(
    prims: &[Arc<Primitive>],
    params: &ParameterDictionary,
) -> Result<Primitive, PbrtError> {
    if prims.is_empty() {
        return Err(PbrtError::error(
            "BVH accelerator requires at least one primitive.",
        ));
    }

    let split_name = params.get_one_string("splitmethod", "sah");
    let split_method = match &split_name as &str {
        "sah" => SplitMethod::SAH,
        "hlbvh" => SplitMethod::HLBVH,
        "middle" => SplitMethod::Middle,
        "equal" => SplitMethod::EqualCounts,
        _ => SplitMethod::SAH,
    };

    let max_prims_in_node: usize = params.get_one_int("maxnodeprims", 4) as usize;
    return Ok(Primitive::Accel(Accel::BVH(Arc::new(BVHAccel::new(
        prims,
        max_prims_in_node,
        split_method,
    )))));
}
