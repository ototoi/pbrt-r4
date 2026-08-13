use super::equal_counts::*;
use super::node::*;
use super::types::*;
use crate::util::base::*;
use crate::util::geometry::*;

#[derive(Debug, Clone, Copy, Default)]
struct BVHSplitBucket {
    count: i32,
    bounds: Bounds3f,
}

fn get_bounds(primitive_info: &[BVHPrimitiveInfo]) -> Bounds3f {
    let mut bounds = primitive_info[0].bounds;
    for pi in primitive_info.iter() {
        bounds = Bounds3f::union(&bounds, &pi.bounds);
    }
    bounds
}

fn get_centroid_bounds(primitive_info: &[BVHPrimitiveInfo]) -> Bounds3f {
    let mut bounds = Bounds3f::new(&primitive_info[0].centroid, &primitive_info[0].centroid);
    for pi in primitive_info.iter() {
        bounds = Bounds3f::union_p(&bounds, &pi.centroid);
    }
    bounds
}

pub fn split_sah(
    dim: usize,
    primitive_info: &mut [BVHPrimitiveInfo],
    ordered_indices: &mut Vec<usize>,
    max_prims_in_node: usize,
    split_method: SplitMethod,
) -> Box<BVHBuildNode> {
    // Partition primitives using approximate SAH
    let n_primitives = primitive_info.len();
    if n_primitives == 1 {
        // Create leaf _BVHBuildNode_
        let first_prim_offset = ordered_indices.len();
        for pi in primitive_info.iter() {
            ordered_indices.push(pi.primitive_number);
        }
        return Box::new(BVHBuildNode::init_leaf(
            first_prim_offset,
            n_primitives,
            &primitive_info[0].bounds,
        ));
    } else if n_primitives == 2 {
        // Partition primitives into equally sized subsets
        return split_equal_counts(
            dim,
            primitive_info,
            ordered_indices,
            max_prims_in_node,
            split_method,
        );
    }

    // Compute bounds of all primitives in BVH node
    let bounds = get_bounds(primitive_info);

    // Compute bound of primitive centroids, choose split dimension _dim_
    let centroid_bounds = get_centroid_bounds(primitive_info);

    // Allocate _BVHSplitBucket_ for SAH partition buckets
    const N_BUCKETS: usize = 12;
    let mut buckets = [BVHSplitBucket::default(); N_BUCKETS];

    // Initialize _BVHSplitBucket_ for SAH partition buckets
    for i in 0..primitive_info.len() {
        let b = (Float::floor(
            N_BUCKETS as Float * centroid_bounds.offset(&primitive_info[i].centroid)[dim],
        ) as i32)
            .min(N_BUCKETS as i32 - 1)
            .max(0) as usize;
        buckets[b].count += 1;
        buckets[b].bounds = Bounds3f::union(&buckets[b].bounds, &primitive_info[i].bounds);
    }

    // Compute costs for splitting after each bucket
    let mut cost = [0.0; N_BUCKETS - 1];

    // Partially initialize _costs_ using a forward scan over splits
    let mut count_below = 0;
    let mut bound_below = buckets[0].bounds;
    for i in 0..(N_BUCKETS - 1) {
        bound_below = Bounds3f::union(&bound_below, &buckets[i].bounds);
        count_below += buckets[i].count;
        cost[i] += count_below as Float * bound_below.surface_area();
    }

    // Finish initializing _costs_ using a backwards scan over splits
    let mut count_above = 0;
    let mut bound_above = Bounds3f::default();
    for i in (1..N_BUCKETS).rev() {
        bound_above = Bounds3f::union(&bound_above, &buckets[i].bounds);
        count_above += buckets[i].count;
        cost[i - 1] += count_above as Float * bound_above.surface_area();
    }

    // Find bucket to split at that minimizes SAH metric
    let mut min_cost = Float::INFINITY;
    let mut min_cost_split_bucket = 0;
    for i in 0..(N_BUCKETS - 1) {
        // Compute cost for candidate split and update minimum if necessary
        if cost[i] < min_cost {
            min_cost = cost[i];
            min_cost_split_bucket = i;
        }
    }

    // Compute leaf cost and SAH split cost for chosen split
    let leaf_cost = n_primitives as Float;
    min_cost = 0.5 + min_cost / bounds.surface_area();

    // Either create leaf or split primitives at selected SAH bucket
    if n_primitives > max_prims_in_node || min_cost < leaf_cost {
        let (mut left, mut right): (Vec<BVHPrimitiveInfo>, Vec<BVHPrimitiveInfo>) =
            primitive_info.iter().partition(|pi| -> bool {
                let b =
                    (Float::floor(N_BUCKETS as Float * centroid_bounds.offset(&pi.centroid)[dim])
                        as i32)
                        .min(N_BUCKETS as i32 - 1)
                        .max(0) as usize;
                b <= min_cost_split_bucket
            });
        let c0 = Some(recursive_build(
            &mut left,
            ordered_indices,
            max_prims_in_node,
            split_method,
        ));
        let c1 = Some(recursive_build(
            &mut right,
            ordered_indices,
            max_prims_in_node,
            split_method,
        ));
        return Box::new(BVHBuildNode::init_interior(dim, c0, c1));
    } else {
        // Create leaf _BVHBuildNode_
        let first_prim_offset = ordered_indices.len();
        for i in 0..n_primitives {
            ordered_indices.push(primitive_info[i].primitive_number);
        }
        return Box::new(BVHBuildNode::init_leaf(
            first_prim_offset,
            n_primitives,
            &bounds,
        ));
    }
}
