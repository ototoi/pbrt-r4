use super::node::*;
use super::types::*;

pub fn split_equal_counts(
    dim: usize,
    primitive_info: &mut [BVHPrimitiveInfo],
    ordered_indices: &mut Vec<usize>,
    max_prims_in_node: usize,
    split_method: SplitMethod,
) -> Box<BVHBuildNode> {
    let info = primitive_info;
    info.sort_by(|a, b| a.centroid[dim].total_cmp(&b.centroid[dim]));
    let mid = info.len() / 2;

    let c0 = Some(recursive_build(
        &mut info[0..mid],
        ordered_indices,
        max_prims_in_node,
        split_method,
    ));
    let c1 = Some(recursive_build(
        &mut info[mid..],
        ordered_indices,
        max_prims_in_node,
        split_method,
    ));
    return Box::new(BVHBuildNode::init_interior(dim, c0, c1));
}
