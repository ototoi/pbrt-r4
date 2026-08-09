use super::super::super::build::*;
use crate::cpu::primitive::*;
use crate::interaction::*;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::profile::*;

use std::sync::Arc;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct SIMDBVHNode {
    pub bboxes: [[float32x4_t; 3]; 2], // 96 bytes
    pub children: [u32; 4],
    flags: u8,
}

#[inline]
unsafe fn get_mask(x: uint32x4_t) -> u32 {
    let mask_list: [u32; 4] = [0x1, 0x2, 0x4, 0x8];
    let mm_a = vandq_u32(x, vld1q_u32(&mask_list as *const u32)); // [0 1 2 3]
    let mm_b = vextq_u32(mm_a, mm_a, 2); // [2 3 0 1]
    let mm_c = vorrq_u32(mm_a, mm_b); // [0+2 1+3 0+2 1+3]
    let mm_d = vextq_u32(mm_c, mm_c, 3); // [1+3 0+2 1+3 0+2]
    let mm_e = vorrq_u32(mm_c, mm_d); // [0+1+2+3 ...]
    return vgetq_lane_u32(mm_e, 0);
}

unsafe fn test_aabb(
    bboxes: &[[float32x4_t; 3]; 2], //4boxes : min-max[2] of xyz[3] of boxes[4]
    org: &[float32x4_t; 3],         //ray origin
    idir: &[float32x4_t; 3],        //ray inveresed direction
    sign: &[usize; 3],              //ray xyz direction -> +:0,-:1
    tmin: float32x4_t,              //ray range tmin
    tmax: float32x4_t,              //ray range tmax
) -> u32 {
    let mut tmin = tmin;
    let mut tmax = tmax;
    // x coordinate
    tmin = vmaxq_f32(
        tmin,
        vmulq_f32(vsubq_f32(bboxes[sign[0]][0], org[0]), idir[0]),
    );
    tmax = vminq_f32(
        tmax,
        vmulq_f32(vsubq_f32(bboxes[1 - sign[0]][0], org[0]), idir[0]),
    );
    // y coordinate
    tmin = vmaxq_f32(
        tmin,
        vmulq_f32(vsubq_f32(bboxes[sign[1]][1], org[1]), idir[1]),
    );
    tmax = vminq_f32(
        tmax,
        vmulq_f32(vsubq_f32(bboxes[1 - sign[1]][1], org[1]), idir[1]),
    );
    // z coordinate
    tmin = vmaxq_f32(
        tmin,
        vmulq_f32(vsubq_f32(bboxes[sign[2]][2], org[2]), idir[2]),
    );
    tmax = vminq_f32(
        tmax,
        vmulq_f32(vsubq_f32(bboxes[1 - sign[2]][2], org[2]), idir[2]),
    );
    return get_mask(vcgeq_f32(tmax, tmin));
}

const EMPTY_MASK: u32 = !0;

#[inline]
fn is_empty(i: u32) -> bool {
    return i == EMPTY_MASK;
}

#[inline]
fn pack_child_index(i: usize) -> u32 {
    u32::try_from(i).expect("QBVH child index exceeds u32 range")
}

#[inline]
fn pack_axis(axis: u8) -> u8 {
    debug_assert!(axis < 4);
    axis
}

impl SIMDBVHNode {
    const LEAF_FLAG: u8 = 1;
    const AXIS_TOP_SHIFT: u8 = 1;
    const AXIS_LEFT_SHIFT: u8 = 3;
    const AXIS_RIGHT_SHIFT: u8 = 5;
    const AXIS_MASK: u8 = 0b11;

    #[inline]
    fn is_leaf(&self) -> bool {
        (self.flags & Self::LEAF_FLAG) != 0
    }

    #[inline]
    fn set_leaf(&mut self) {
        self.flags |= Self::LEAF_FLAG;
    }

    #[inline]
    fn set_axes(&mut self, axis_top: u8, axis_left: u8, axis_right: u8) {
        self.flags = (self.flags & Self::LEAF_FLAG)
            | (pack_axis(axis_top) << Self::AXIS_TOP_SHIFT)
            | (pack_axis(axis_left) << Self::AXIS_LEFT_SHIFT)
            | (pack_axis(axis_right) << Self::AXIS_RIGHT_SHIFT);
    }

    #[inline]
    fn axis_top(&self) -> usize {
        ((self.flags >> Self::AXIS_TOP_SHIFT) & Self::AXIS_MASK) as usize
    }

    #[inline]
    fn axis_left(&self) -> usize {
        ((self.flags >> Self::AXIS_LEFT_SHIFT) & Self::AXIS_MASK) as usize
    }

    #[inline]
    fn axis_right(&self) -> usize {
        ((self.flags >> Self::AXIS_RIGHT_SHIFT) & Self::AXIS_MASK) as usize
    }
}

impl Default for SIMDBVHNode {
    fn default() -> Self {
        unsafe {
            let bboxes = [[vdupq_n_f32(0.0); 3]; 2];
            let children = [0, 0, 0, 0];
            SIMDBVHNode {
                bboxes,
                children,
                flags: 0,
            }
        }
    }
}

fn flatten_qbvh_side(
    nodes: &mut Vec<SIMDBVHNode>,
    node: Box<BVHBuildNode>,
    indices: &mut [u32; 4],
    boxes: &mut [[Vector3f; 2]; 4],
    slot: usize,
) -> u8 {
    let split_axis = node.split_axis;
    if node.n_primitives > 0 {
        let bounds = node.bounds;
        indices[slot] = pack_child_index(flatten_qbvh_tree(nodes, node));
        indices[slot + 1] = EMPTY_MASK;
        boxes[slot][0] = bounds.min;
        boxes[slot][1] = bounds.max;
    } else {
        let [c0, c1] = node.children;
        let c0 = c0.expect("BVH interior node missing left child");
        let c1 = c1.expect("BVH interior node missing right child");
        boxes[slot][0] = c0.bounds.min;
        boxes[slot][1] = c0.bounds.max;
        boxes[slot + 1][0] = c1.bounds.min;
        boxes[slot + 1][1] = c1.bounds.max;
        indices[slot] = pack_child_index(flatten_qbvh_tree(nodes, c0));
        indices[slot + 1] = pack_child_index(flatten_qbvh_tree(nodes, c1));
    }
    split_axis
}

fn flatten_qbvh_tree(nodes: &mut Vec<SIMDBVHNode>, node: Box<BVHBuildNode>) -> usize {
    let offset = nodes.len();
    nodes.push(SIMDBVHNode::default());
    if node.n_primitives > 0 {
        nodes[offset].set_leaf();
        nodes[offset].children[0] = pack_child_index(node.first_prim_offset);
        nodes[offset].children[1] = pack_child_index(node.n_primitives);
    } else {
        let split_axis = node.split_axis;
        debug_assert!(
            node.children[0].is_some() && node.children[1].is_some(),
            "interior QBVH node must have two children"
        );
        let mut indices: [u32; 4] = [0; 4];
        let mut boxes: [[Vector3f; 2]; 4] = [[Vector3f::zero(); 2]; 4];

        let [c0, c1] = node.children;
        let c0_axis = flatten_qbvh_side(
            nodes,
            c0.expect("BVH interior node missing left child"),
            &mut indices,
            &mut boxes,
            0,
        );
        let c1_axis = flatten_qbvh_side(
            nodes,
            c1.expect("BVH interior node missing right child"),
            &mut indices,
            &mut boxes,
            2,
        );

        //convert & swizzle
        let mut bboxes: [[[f32; 4]; 3]; 2] = [[[0.0; 4]; 3]; 2];
        for j in 0..3 {
            //xyz
            for k in 0..4 {
                bboxes[0][j][k] = boxes[k][0][j] as f32;
                bboxes[1][j][k] = boxes[k][1][j] as f32;
            }
        }

        unsafe {
            for m in 0..2 {
                for j in 0..3 {
                    let a = bboxes[m][j][0];
                    let b = bboxes[m][j][1];
                    let c = bboxes[m][j][2];
                    let d = bboxes[m][j][3];
                    let l = [a, b, c, d];
                    nodes[offset].bboxes[m][j] = vld1q_f32(&l as *const f32);
                }
            }
        }
        //for i in 0..4 {
        //    nodes[offset].children[i] = indices[i];
        nodes[offset].children = indices;
        nodes[offset].set_axes(split_axis, c0_axis, c1_axis);
    }
    return offset;
}

#[rustfmt::skip]
const ORDER_TABLE: [u32; 128] = [
    0x44444, 0x44444, 0x44444, 0x44444, 0x44444, 0x44444, 0x44444, 0x44444,
    0x44440, 0x44440, 0x44440, 0x44440, 0x44440, 0x44440, 0x44440, 0x44440,
    0x44441, 0x44441, 0x44441, 0x44441, 0x44441, 0x44441, 0x44441, 0x44441,
    0x44401, 0x44401, 0x44410, 0x44410, 0x44401, 0x44401, 0x44410, 0x44410,
    0x44442, 0x44442, 0x44442, 0x44442, 0x44442, 0x44442, 0x44442, 0x44442,
    0x44402, 0x44402, 0x44402, 0x44402, 0x44420, 0x44420, 0x44420, 0x44420,
    0x44412, 0x44412, 0x44412, 0x44412, 0x44421, 0x44421, 0x44421, 0x44421,
    0x44012, 0x44012, 0x44102, 0x44102, 0x44201, 0x44201, 0x44210, 0x44210,
    0x44443, 0x44443, 0x44443, 0x44443, 0x44443, 0x44443, 0x44443, 0x44443,
    0x44403, 0x44403, 0x44403, 0x44403, 0x44430, 0x44430, 0x44430, 0x44430,
    0x44413, 0x44413, 0x44413, 0x44413, 0x44431, 0x44431, 0x44431, 0x44431,
    0x44013, 0x44013, 0x44103, 0x44103, 0x44301, 0x44301, 0x44310, 0x44310,
    0x44423, 0x44432, 0x44423, 0x44432, 0x44423, 0x44432, 0x44423, 0x44432,
    0x44023, 0x44032, 0x44023, 0x44032, 0x44230, 0x44320, 0x44230, 0x44320,
    0x44123, 0x44132, 0x44123, 0x44132, 0x44231, 0x44321, 0x44231, 0x44321,
    0x40123, 0x40132, 0x41023, 0x41032, 0x42301, 0x43201, 0x42310, 0x43210,
];

#[inline]
fn intersect_primitives(
    primitives: &[Arc<Primitive>],
    r: &Ray,
    t_max: Float,
) -> Option<ShapeIntersection> {
    let mut isect = None;
    let mut current_t_max = t_max;
    for prim in primitives.iter() {
        if let Some(isect_n) = prim.intersect(r, current_t_max) {
            current_t_max = isect_n.t_hit;
            isect = Some(isect_n);
        }
    }
    return isect;
}

#[inline]
fn intersect_primitives_p(primitives: &[Arc<Primitive>], r: &Ray, t_max: Float) -> bool {
    for prim in primitives.iter() {
        if prim.intersect_p(r, t_max) {
            return true;
        }
    }
    return false;
}

fn get_sign(x: Float) -> usize {
    if x.is_sign_negative() {
        1
    } else {
        0
    }
}

fn is_valid_hitmask(
    order: u32,
    sign: &[usize; 3],
    axis_top: u8,
    axis_left: u8,
    axis_right: u8,
) -> bool {
    let org_order = order;
    let mut order = order;
    let mut indices = Vec::new();
    while (order & 0x4) == 0 {
        let cidx = order & 0x3;
        indices.push(cidx);
        order >>= 4;
    }
    indices.reverse();

    let sign_top = sign[axis_top as usize];
    let sign_left = sign[axis_left as usize];
    let sign_right = sign[axis_right as usize];
    let mut predicts = vec![0, 1, 2, 3];
    if sign_left != 0 {
        predicts.swap(0, 1);
    }
    if sign_right != 0 {
        predicts.swap(2, 3);
    }
    if sign_top != 0 {
        predicts.swap(0, 2);
        predicts.swap(1, 3);
    }
    let org_predicts = predicts.clone();
    for i in 0..indices.len() {
        let idx = indices[i];
        if predicts.len() == 0 {
            println!(
                "order: {:X}, indices: {:?}, predicts: {:?}, sign: {:?}, axis: [{}, {}, {}]",
                org_order, indices, org_predicts, sign, axis_top, axis_left, axis_right
            );
            return false;
        }
        while predicts.len() > 0 {
            let first = *predicts.first().unwrap();
            if first == idx {
                predicts.remove(0);
                break;
            } else {
                predicts.remove(0);
            }
        }
    }
    return true;
}

#[inline]
unsafe fn intersect_simd(
    primitives: &[Arc<Primitive>],
    nodes: &[SIMDBVHNode],
    r: &Ray,
    tmin: Float,
    tmax: Float,
    original_t_max: Float,
) -> Option<ShapeIntersection> {
    let mut isect = None;
    let mut nodes_to_visit: Vec<usize> = Vec::with_capacity(16);

    let org: [float32x4_t; 3] = [
        vdupq_n_f32(r.o.x as f32),
        vdupq_n_f32(r.o.y as f32),
        vdupq_n_f32(r.o.z as f32),
    ];

    let idir: [float32x4_t; 3] = [
        vdupq_n_f32(r.d.x.recip() as f32),
        vdupq_n_f32(r.d.y.recip() as f32),
        vdupq_n_f32(r.d.z.recip() as f32),
    ];

    let sign = [get_sign(r.d.x), get_sign(r.d.y), get_sign(r.d.z)];

    // Per-primitive cutoff starts at the ORIGINAL t_max (not bbox-clamped),
    // matching pbrt-v4 BVHAccel::Intersect behaviour. The SIMD tmax (in the
    // float32x4_t register) keeps the bbox-clamped value for sub-tree pruning.
    let mut ftmax = original_t_max;

    let tmin = vdupq_n_f32(tmin as f32);
    let mut tmax = vdupq_n_f32(tmax as f32);

    nodes_to_visit.push(0);
    while let Some(current_node_index) = nodes_to_visit.pop() {
        debug_assert_ne!(current_node_index, EMPTY_MASK as usize);
        let node = &nodes[current_node_index];
        if !node.is_leaf() {
            let hit_mask = test_aabb(&node.bboxes, &org, &idir, &sign, tmin, tmax) as usize;
            if hit_mask != 0 {
                let axis_top = node.axis_top();
                let axis_left = node.axis_left();
                let axis_right = node.axis_right();
                let node_idx = (sign[axis_top] << 2) | (sign[axis_left] << 1) | sign[axis_right];
                let mut order = ORDER_TABLE[hit_mask * 8 + node_idx];

                if cfg!(debug_assertions) {
                    assert!(is_valid_hitmask(
                        order,
                        &sign,
                        axis_top as u8,
                        axis_left as u8,
                        axis_right as u8
                    ));
                }

                while (order & 0x4) == 0 {
                    let cidx = node.children[(order & 0x3) as usize];
                    if !is_empty(cidx) {
                        nodes_to_visit.push(cidx as usize);
                    }
                    order >>= 4;
                }
            }
        } else {
            let start = node.children[0] as usize;
            let end = start + node.children[1] as usize;
            if let Some(isect_n) = intersect_primitives(&primitives[start..end], r, ftmax) {
                let t_hit = isect_n.t_hit;
                tmax = vdupq_n_f32(t_hit as f32);
                ftmax = t_hit;
                isect = Some(isect_n);
            }
        }
    }
    return isect;
}

#[inline]
unsafe fn intersect_simd_p(
    primitives: &[Arc<Primitive>],
    nodes: &[SIMDBVHNode],
    r: &Ray,
    tmin: Float,
    tmax: Float,
    original_t_max: Float,
) -> bool {
    let mut nodes_to_visit: Vec<usize> = Vec::with_capacity(16);

    let org: [float32x4_t; 3] = [
        vdupq_n_f32(r.o.x as f32),
        vdupq_n_f32(r.o.y as f32),
        vdupq_n_f32(r.o.z as f32),
    ];

    let idir: [float32x4_t; 3] = [
        vdupq_n_f32(r.d.x.recip() as f32),
        vdupq_n_f32(r.d.y.recip() as f32),
        vdupq_n_f32(r.d.z.recip() as f32),
    ];

    let sign = [get_sign(r.d.x), get_sign(r.d.y), get_sign(r.d.z)];

    let ftmax = original_t_max;

    let tmin = vdupq_n_f32(tmin as f32);
    let tmax = vdupq_n_f32(tmax as f32);

    nodes_to_visit.push(0);
    while let Some(current_node_index) = nodes_to_visit.pop() {
        debug_assert_ne!(current_node_index, EMPTY_MASK as usize);
        let node = &nodes[current_node_index];
        if !node.is_leaf() {
            let hit_mask = test_aabb(&node.bboxes, &org, &idir, &sign, tmin, tmax) as usize;
            if hit_mask != 0 {
                let node_idx = (sign[node.axis_top()] << 2)
                    | (sign[node.axis_left()] << 1)
                    | sign[node.axis_right()];
                let mut order = ORDER_TABLE[hit_mask * 8 + node_idx];
                while (order & 0x4) == 0 {
                    let cidx = node.children[(order & 0x3) as usize];
                    if !is_empty(cidx) {
                        nodes_to_visit.push(cidx as usize);
                    }
                    order >>= 4;
                }
            }
        } else {
            let start = node.children[0] as usize;
            let end = start + node.children[1] as usize;
            if intersect_primitives_p(&primitives[start..end], r, ftmax) {
                return true;
            }
        }
    }
    return false;
}

#[derive(Clone)]
pub struct QBVHAccel {
    pub primitives: Vec<Arc<Primitive>>,
    pub nodes: Vec<SIMDBVHNode>,
    pub bounds: Bounds3f,
}

impl QBVHAccel {
    pub fn new(
        prims: &[Arc<Primitive>],
        max_prims_in_node: usize,
        split_method: SplitMethod,
    ) -> Self {
        let _p = ProfilePhase::new(Prof::AccelConstruction);

        let max_prims_in_node = usize::min(max_prims_in_node, 255);
        let mut orderd_prims = Vec::new();
        let root = create_bvh_node(&mut orderd_prims, prims, max_prims_in_node, split_method);
        let bounds = root.bounds;
        let mut nodes = Vec::new();
        nodes.reserve(root.node_count());
        flatten_qbvh_tree(&mut nodes, root);
        QBVHAccel {
            primitives: orderd_prims,
            nodes,
            bounds,
        }
    }

    pub fn bounds(&self) -> Bounds3f {
        return self.bounds;
    }

    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let _p = ProfilePhase::new(Prof::AccelIntersect);

        if let Some((tmin, tmax)) = self.bounds.intersect_p(r, t_max) {
            unsafe {
                return intersect_simd(&self.primitives, &self.nodes, r, tmin, tmax, t_max);
            }
        }
        return None;
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        let _p = ProfilePhase::new(Prof::AccelIntersectP);

        if let Some((tmin, tmax)) = self.bounds.intersect_p(r, t_max) {
            unsafe {
                return intersect_simd_p(&self.primitives, &self.nodes, r, tmin, tmax, t_max);
            }
        }
        return false;
    }
}
