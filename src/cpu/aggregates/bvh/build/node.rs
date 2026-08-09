use super::equal_counts::*;
use super::hlbvh::*;
use super::middle::*;
use super::sah::*;
use super::types::*;
use crate::cpu::primitive::*;
use crate::util::base::*;
use crate::util::geometry::*;

use std::sync::Arc;

const BOUND_EPS: Float = (f32::EPSILON * 2.0) as Float;
fn inflate_bound(b: &Bounds3f, delta: Float) -> Bounds3f {
    let mut min = b.min;
    let mut max = b.max;
    min.x -= delta;
    min.y -= delta;
    min.z -= delta;
    max.x += delta;
    max.y += delta;
    max.z += delta;
    return Bounds3f::new(&min, &max);
}

pub fn recursive_build(
    primitive_info: &mut [BVHPrimitiveInfo],
    ordered_indices: &mut Vec<usize>,
    max_prims_in_node: usize,
    split_method: SplitMethod,
) -> Box<BVHBuildNode> {
    let start = 0;
    let end = primitive_info.len();
    let n_primitives = end;
    let mut bounds = primitive_info[start].bounds.clone();
    for i in (start + 1)..end {
        bounds = Bounds3f::union(&bounds, &primitive_info[i].bounds);
    }
    if bounds.surface_area() == 0.0 || n_primitives == 1 {
        let offset = ordered_indices.len();
        for i in start..end {
            ordered_indices.push(primitive_info[i].primitive_number);
        }
        return Box::new(BVHBuildNode::init_leaf(offset, n_primitives, &bounds));
    } else {
        let center = primitive_info[start].centroid;
        let mut c_bounds = Bounds3f::new(&center, &center);
        for i in (start + 1)..end {
            let p = primitive_info[i].centroid;
            c_bounds = c_bounds.union_p(&p);
        }
        let dim = c_bounds.maximum_extent();
        if c_bounds.min[dim] == c_bounds.max[dim] {
            let offset = ordered_indices.len();
            for i in start..end {
                ordered_indices.push(primitive_info[i].primitive_number);
            }
            return Box::new(BVHBuildNode::init_leaf(offset, n_primitives, &bounds));
        }
        match split_method {
            SplitMethod::Middle => {
                let p_mid = (c_bounds.min[dim] + c_bounds.max[dim]) / 2.0;
                return split_middle(
                    dim,
                    p_mid,
                    primitive_info,
                    ordered_indices,
                    max_prims_in_node,
                    split_method,
                );
            }
            SplitMethod::EqualCounts => {
                return split_equal_counts(
                    dim,
                    primitive_info,
                    ordered_indices,
                    max_prims_in_node,
                    split_method,
                );
            }
            SplitMethod::SAH => {
                return split_sah(
                    dim,
                    primitive_info,
                    ordered_indices,
                    max_prims_in_node,
                    split_method,
                );
            }
            _ => {
                return split_equal_counts(
                    dim,
                    primitive_info,
                    ordered_indices,
                    max_prims_in_node,
                    split_method,
                );
            }
        }
    }
}

fn create_bvh_node_indices(
    ordered_indices: &mut Vec<usize>,
    prims: &[Arc<Primitive>],
    max_prims_in_node: usize,
    split_method: SplitMethod,
) -> Box<BVHBuildNode> {
    if prims.is_empty() {
        return Box::new(BVHBuildNode::init_leaf(0, 0, &Bounds3f::default()));
    }
    let mut primitive_info: Vec<BVHPrimitiveInfo> = prims
        .iter()
        .enumerate()
        .map(|(i, p)| -> BVHPrimitiveInfo {
            let b = p.bounds();
            let b = inflate_bound(&b, BOUND_EPS);
            let c = (b.min + b.max) * 0.5;
            return BVHPrimitiveInfo::new(i, &b, &c);
        })
        .collect();
    match split_method {
        SplitMethod::HLBVH => {
            return hlbvh_build(&mut primitive_info, ordered_indices, max_prims_in_node)
        }
        _ => {
            return recursive_build(
                &mut primitive_info,
                ordered_indices,
                max_prims_in_node,
                split_method,
            )
        }
    };
}

pub fn create_bvh_node(
    ordered_prims: &mut Vec<Arc<Primitive>>,
    prims: &[Arc<Primitive>],
    max_prims_in_node: usize,
    split_method: SplitMethod,
) -> Box<BVHBuildNode> {
    let mut ordered_indices = Vec::new();
    let node =
        create_bvh_node_indices(&mut ordered_indices, prims, max_prims_in_node, split_method);
    for index in ordered_indices {
        ordered_prims.push(prims[index].clone());
    }
    return node;
}
