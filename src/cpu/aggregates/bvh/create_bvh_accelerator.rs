use super::accel::BVHAccel;
use super::build::*;
use crate::cpu::aggregates::Accel;
use crate::cpu::primitive::*;
use crate::paramdict::*;
use crate::util::error::*;

use std::sync::Arc;

const DEFAULT_SPLIT_METHOD: &str = "hlbvh";
const DEFAULT_MAX_PRIMS_IN_NODE: i32 = 8;

pub fn create_bvh_accelerator(
    prims: &[Arc<Primitive>],
    params: &ParameterDictionary,
) -> Result<Primitive, PbrtError> {
    if prims.is_empty() {
        return Err(PbrtError::error(
            "BVH accelerator requires at least one primitive.",
        ));
    }

    // Use HLBVH until the current SAH implementation's over-splitting is
    // corrected. An explicit scene parameter still takes precedence. The
    // environment variable is an operational override for unparameterized
    // scenes, useful for comparing BVH builders without editing scene files.
    let default_split_method = std::env::var("PBRT_BVH_SPLITMETHOD")
        .ok()
        .filter(|value| matches!(value.as_str(), "sah" | "hlbvh" | "middle" | "equal"))
        .unwrap_or_else(|| DEFAULT_SPLIT_METHOD.to_owned());
    let split_name = params.get_one_string("splitmethod", &default_split_method);
    let split_method = match &split_name as &str {
        "sah" => SplitMethod::SAH,
        "hlbvh" => SplitMethod::HLBVH,
        "middle" => SplitMethod::Middle,
        "equal" => SplitMethod::EqualCounts,
        _ => SplitMethod::SAH,
    };

    let default_max_prims_in_node = std::env::var("PBRT_BVH_MAXNODEPRIMS")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(DEFAULT_MAX_PRIMS_IN_NODE);
    let max_prims_in_node: usize =
        params.get_one_int("maxnodeprims", default_max_prims_in_node) as usize;
    return Ok(Primitive::Accel(Accel::BVH(Arc::new(BVHAccel::new(
        prims,
        max_prims_in_node,
        split_method,
    )))));
}
