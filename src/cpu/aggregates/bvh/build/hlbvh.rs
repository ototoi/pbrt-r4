use super::types::*;
use crate::util::base::*;
use crate::util::geometry::*;

#[derive(Debug, Clone, Copy, Default)]
struct MortonPrimitive {
    primitive_index: u32,
    morton_code: u32,
}

#[derive(Default)]
struct LBVHTreelet {
    pub start_index: u32,
    pub n_primitives: u32,
    pub build_nodes: Option<Box<BVHBuildNode>>,
}

const MORTON_BITS: u32 = 10;
const MORTON_SCALE: u32 = 1 << MORTON_BITS;

#[rustfmt::skip]
fn left_shift3(x: u32) -> u32 {
    assert!(x <= MORTON_SCALE);

    let mut x = x;
    if x >= MORTON_SCALE {
        x = MORTON_SCALE - 1;
    }
    // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x | (x.wrapping_shl(16))) & 0b_0000_0011_0000_0000_0000_0000_1111_1111; //0x30000ff;
    // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x.wrapping_shl(8))) & 0b_0000_0011_0000_0000_1111_0000_0000_1111; //0x300f00f
    // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x.wrapping_shl(4))) & 0b_0000_0011_0000_1100_0011_0000_1100_0011; //0x30c30c3
    // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x.wrapping_shl(2))) & 0b_0000_1001_0010_0100_1001_0010_0100_1001; //0x9249249
    // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

fn encode_morton3(v: &[Float]) -> u32 {
    let x = Float::ceil(v[0]) as u32;
    let y = Float::ceil(v[1]) as u32;
    let z = Float::ceil(v[2]) as u32;
    return (left_shift3(z) << 2) | (left_shift3(y) << 1) | left_shift3(x);
}

fn radix_sort(v0: Vec<MortonPrimitive>) -> Vec<MortonPrimitive> {
    const BITS_PER_PASS: u32 = 6;
    const N_BITS: u32 = MORTON_BITS * 3; //30
    const N_PASSES: u32 = N_BITS / BITS_PER_PASS; //5
    const N_BUCKETS: u32 = 1 << BITS_PER_PASS;
    const BIT_MASK: u32 = (1 << BITS_PER_PASS) - 1;

    let mut v0 = v0;
    let mut v1 = v0.clone();
    for pass in 0..N_PASSES {
        // Perform one pass of radix sort, sorting _bitsPerPass_ bits
        let low_bit = pass * BITS_PER_PASS;

        // Set in and out vector pointers for radix sort pass
        let (in_v, out_v) = if (pass & 1) == 0 {
            (&v0, &mut v1)
        } else {
            (&v1, &mut v0)
        };

        let mut bucket_count = [0; N_BUCKETS as usize];

        for mp in in_v.iter() {
            let bucket = (mp.morton_code >> low_bit) & BIT_MASK;
            bucket_count[bucket as usize] += 1;
        }

        // Compute starting index in output array for each bucket
        let mut out_index = [0; N_BUCKETS as usize];
        for i in 1..out_index.len() {
            out_index[i] = out_index[i - 1] + bucket_count[i - 1];
        }

        // Store sorted values in output array
        for mp in in_v.iter() {
            let bucket = (mp.morton_code >> low_bit) & BIT_MASK;
            out_v[out_index[bucket as usize]] = *mp;
            out_index[bucket as usize] += 1;
        }
    }
    //0:i o
    //1:o i
    //2:i o
    //3:o i
    //4:i o
    //5 -> o
    if (N_PASSES & 1) != 0 {
        return v1;
    } else {
        return v0;
    }
}

fn split_node(
    morton_prims: &mut [MortonPrimitive],
    primitive_info: &mut [BVHPrimitiveInfo],
    ordered_indices: &mut Vec<usize>,
    max_prims_in_node: usize,
    dim: usize,
) -> Box<BVHBuildNode> {
    let n_primitives = morton_prims.len();
    if n_primitives <= max_prims_in_node {
        let first_prim_offset = ordered_indices.len();
        let mut bounds = primitive_info[morton_prims[0].primitive_index as usize].bounds;
        ordered_indices.push(morton_prims[0].primitive_index as usize);
        for i in 1..n_primitives {
            let primitive_index = morton_prims[i].primitive_index as usize;
            ordered_indices.push(primitive_index);
            bounds = Bounds3f::union(&bounds, &primitive_info[primitive_index].bounds);
        }
        let node = Box::new(BVHBuildNode::init_leaf(
            first_prim_offset,
            n_primitives,
            &bounds,
        ));
        assert_eq!(n_primitives, node.primitive_count());
        return node;
    } else {
        morton_prims.sort_by(|a, b| {
            let ap = &primitive_info[a.primitive_index as usize];
            let bp = &primitive_info[b.primitive_index as usize];
            ap.centroid[dim].total_cmp(&bp.centroid[dim])
        });

        let next_dim = (dim + 3 - 1) % 3;
        let split_offset = n_primitives / 2;

        let node0 = split_node(
            &mut morton_prims[0..split_offset],
            primitive_info,
            ordered_indices,
            max_prims_in_node,
            next_dim,
        );
        let node1 = split_node(
            &mut morton_prims[split_offset..n_primitives],
            primitive_info,
            ordered_indices,
            max_prims_in_node,
            next_dim,
        );

        let node = Box::new(BVHBuildNode::init_interior(dim, Some(node0), Some(node1)));
        return node;
    }
}

fn emit_lbvh(
    morton_prims: &[MortonPrimitive],
    primitive_info: &mut [BVHPrimitiveInfo],
    ordered_indices: &mut Vec<usize>,
    max_prims_in_node: usize,
    bit_index: i32,
) -> Box<BVHBuildNode> {
    let n_primitives = morton_prims.len();
    if n_primitives <= max_prims_in_node {
        let first_prim_offset = ordered_indices.len();
        let mut bounds = primitive_info[morton_prims[0].primitive_index as usize].bounds;
        ordered_indices.push(morton_prims[0].primitive_index as usize);
        for i in 1..n_primitives {
            let primitive_index = morton_prims[i].primitive_index as usize;
            ordered_indices.push(primitive_index);
            bounds = Bounds3f::union(&bounds, &primitive_info[primitive_index].bounds);
        }
        let node = Box::new(BVHBuildNode::init_leaf(
            first_prim_offset,
            n_primitives,
            &bounds,
        ));
        assert_eq!(n_primitives, node.primitive_count());
        return node;
    } else {
        if bit_index == -1 {
            let mut new_morton = morton_prims.to_vec();
            return split_node(
                &mut new_morton,
                primitive_info,
                ordered_indices,
                max_prims_in_node,
                2,
            );
        } else {
            let mask: u32 = 1 << bit_index;
            // Advance to next subtree level if there's no LBVH split for this bit
            if (morton_prims[0].morton_code & mask)
                == (morton_prims[n_primitives - 1].morton_code & mask)
            {
                return emit_lbvh(
                    morton_prims,
                    primitive_info,
                    ordered_indices,
                    max_prims_in_node,
                    bit_index - 1,
                );
            }

            // Find LBVH split point for this dimension
            let mut search_start = 0;
            let mut search_end = n_primitives - 1;
            while search_start + 1 != search_end {
                let mid = (search_start + search_end) / 2;
                if (morton_prims[search_start].morton_code & mask)
                    == (morton_prims[mid].morton_code & mask)
                {
                    search_start = mid;
                } else {
                    search_end = mid;
                }
            }
            let split_offset = search_end;
            // Create and return interior LBVH node
            let node0 = emit_lbvh(
                &morton_prims[0..split_offset],
                primitive_info,
                ordered_indices,
                max_prims_in_node,
                bit_index - 1,
            );
            let node1 = emit_lbvh(
                &morton_prims[split_offset..n_primitives],
                primitive_info,
                ordered_indices,
                max_prims_in_node,
                bit_index - 1,
            );

            let axis = bit_index % 3;
            let node = Box::new(BVHBuildNode::init_interior(
                axis as usize,
                Some(node0),
                Some(node1),
            ));
            return node;
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct BucketInfo {
    pub count: u32,
    pub bounds: Bounds3f,
}

fn build_upper_sah(treelet_roots: &[Box<BVHBuildNode>]) -> Box<BVHBuildNode> {
    if treelet_roots.len() <= 1 {
        return treelet_roots[0].clone();
    }

    let mut bounds = treelet_roots[0].bounds;
    for i in 1..treelet_roots.len() {
        bounds = Bounds3f::union(&bounds, &treelet_roots[i].bounds);
    }

    // Compute bound of HLBVH node centroids, choose split dimension _dim_
    let center = (treelet_roots[0].bounds.min + treelet_roots[0].bounds.max) * 0.5;
    let mut centroid_bounds = Bounds3f::new(&center, &center);
    for i in 1..treelet_roots.len() {
        let c = (treelet_roots[i].bounds.min + treelet_roots[i].bounds.max) * 0.5;
        centroid_bounds = Bounds3f::union_p(&centroid_bounds, &c);
    }

    let dim = centroid_bounds.maximum_extent();
    const N_BUCKETS: usize = 12;
    let mut buckets = [BucketInfo::default(); N_BUCKETS];
    for i in 0..treelet_roots.len() {
        let c = (treelet_roots[i].bounds.min[dim] + treelet_roots[i].bounds.max[dim]) * 0.5;
        let b = (N_BUCKETS as Float
            * ((c - centroid_bounds.min[dim])
                / (centroid_bounds.max[dim] - centroid_bounds.min[dim]))) as usize;
        let b = usize::min(b, N_BUCKETS - 1);
        buckets[b].count += 1;
        if buckets[b].count == 1 {
            buckets[b].bounds = treelet_roots[i].bounds;
        } else {
            buckets[b].bounds = Bounds3f::union(&buckets[b].bounds, &treelet_roots[i].bounds);
        }
    }

    // Compute costs for splitting after each bucket
    let mut cost: [Float; N_BUCKETS - 1] = [0.0; N_BUCKETS - 1];
    for i in 0..(N_BUCKETS - 1) {
        let mut b0 = Bounds3f::default();
        let mut b1 = Bounds3f::default();
        let mut count0 = 0;
        let mut count1 = 0;
        for j in 0..=i {
            b0 = Bounds3f::union(&b0, &buckets[j].bounds);
            count0 += buckets[j].count;
        }
        for j in (i + 1)..N_BUCKETS {
            b1 = Bounds3f::union(&b1, &buckets[j].bounds);
            count1 += buckets[j].count;
        }
        cost[i] = 0.125
            + ((count0 as Float * b0.surface_area() + count1 as Float * b1.surface_area())
                / bounds.surface_area());
    }

    // Find bucket to split at that minimizes SAH metric
    let mut min_cost = cost[0];
    let mut min_cost_split_bucket = 0;
    for i in 1..(N_BUCKETS - 1) {
        if cost[i] < min_cost {
            min_cost = cost[i];
            min_cost_split_bucket = i;
        }
    }

    // Split nodes and create interior HLBVH SAH node
    let (c0, c1): (Vec<&Box<BVHBuildNode>>, Vec<&Box<BVHBuildNode>>) =
        treelet_roots.iter().partition(|node| -> bool {
            let c = (node.bounds.min[dim] + node.bounds.max[dim]) * 0.5;
            let b = (N_BUCKETS as Float
                * ((c - centroid_bounds.min[dim])
                    / (centroid_bounds.max[dim] - centroid_bounds.min[dim])))
                as usize;
            let b = usize::min(b, N_BUCKETS - 1);
            return b <= min_cost_split_bucket;
        });

    let node0;
    {
        let mut cc: Vec<Box<BVHBuildNode>> = Vec::new();
        for c in c0.iter() {
            cc.push((*c).clone());
        }
        node0 = Some(build_upper_sah(&cc));
    }

    let node1;
    {
        let mut cc: Vec<Box<BVHBuildNode>> = Vec::new();
        for c in c1.iter() {
            cc.push((*c).clone());
        }
        node1 = Some(build_upper_sah(&cc));
    }

    let node = Box::new(BVHBuildNode::init_interior(dim, node0, node1));
    return node;
}

pub fn hlbvh_build(
    primitive_info: &mut [BVHPrimitiveInfo],
    ordered_indices: &mut Vec<usize>,
    max_prims_in_node: usize,
) -> Box<BVHBuildNode> {
    let mut bounds = primitive_info[0].bounds.clone();
    for i in 1..primitive_info.len() {
        bounds = Bounds3f::union(&bounds, &primitive_info[i].bounds);
    }

    let mut morton_prims = vec![MortonPrimitive::default(); primitive_info.len()];
    for i in 0..morton_prims.len() {
        // Initialize _mortonPrims[i]_ for _i_th primitive
        morton_prims[i].primitive_index = primitive_info[i].primitive_number as u32; //
        let centroid_offset = bounds.offset(&primitive_info[i].centroid);
        let scaled_centroid = [
            centroid_offset.x.clamp(0.0, 1.0) * MORTON_SCALE as Float,
            centroid_offset.y.clamp(0.0, 1.0) * MORTON_SCALE as Float,
            centroid_offset.z.clamp(0.0, 1.0) * MORTON_SCALE as Float,
        ];
        morton_prims[i].morton_code = encode_morton3(&scaled_centroid);
    }

    // Radix sort primitive Morton indices
    morton_prims = radix_sort(morton_prims);
    {
        for i in 0..(morton_prims.len() - 1) {
            assert!(morton_prims[i].morton_code <= morton_prims[i + 1].morton_code);
        }
    }

    // Create LBVH treelets at bottom of BVH

    // Find intervals of primitives for each treelet
    let mut treelets_to_build = Vec::new();
    let mut start = 0;
    for end in 1..(morton_prims.len() + 1) {
        const MASK: u32 = 0b_0011_1111_1111_1100_0000_0000_0000_0000;
        if end == morton_prims.len()
            || ((morton_prims[start].morton_code & MASK) != (morton_prims[end].morton_code & MASK))
        {
            // Add entry to _treeletsToBuild_ for this treelet
            let n_primitives = end - start;
            let treelet = LBVHTreelet {
                start_index: start as u32,
                n_primitives: n_primitives as u32,
                build_nodes: None,
            };
            treelets_to_build.push(treelet);
            start = end;
        }
    }

    ordered_indices.reserve(primitive_info.len());
    const FIRST_BIT_INDEX: i32 = 29 - 12; //3 * 10 - 1 - 12
    for i in 0..treelets_to_build.len() {
        let tr = &mut treelets_to_build[i];
        let start = tr.start_index as usize;
        let end = start + tr.n_primitives as usize;
        tr.build_nodes = Some(emit_lbvh(
            &morton_prims[start..end],
            primitive_info,
            ordered_indices,
            max_prims_in_node,
            FIRST_BIT_INDEX,
        ));
    }

    // Create and return SAH BVH from LBVH treelets
    let mut finished_treelets = Vec::with_capacity(treelets_to_build.len());
    for treelet in treelets_to_build {
        finished_treelets.push(treelet.build_nodes.unwrap().clone());
    }
    let node = build_upper_sah(&finished_treelets);
    return node;
}
