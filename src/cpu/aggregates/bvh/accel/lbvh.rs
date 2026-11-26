use super::super::build::*;
use crate::cpu::primitive::*;
use crate::interaction::*;
use crate::util::base::*;
use crate::util::geometry::*;

use std::ops::Range;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct LinearBVHNode {
    pub bounds: [[Float; 3]; 2], //32*3*2
    pub offset: usize,
    pub n_primitives: usize, // 0 -> interior node
    pub axis: u8,            // interior node: xyz
}

fn to_array_bounds3f(b: &Bounds3f) -> [[Float; 3]; 2] {
    return [
        [b.min.x as Float, b.min.y as Float, b.min.z as Float],
        [b.max.x as Float, b.max.y as Float, b.max.z as Float],
    ];
}

impl LinearBVHNode {
    pub fn new(node: &BVHBuildNode) -> Self {
        LinearBVHNode {
            bounds: to_array_bounds3f(&node.bounds),
            offset: 0,
            n_primitives: node.n_primitives, // 0 -> interior node
            axis: node.split_axis as u8,     // interior node: xyz
        }
    }
}

fn flatten_bvh_tree(nodes: &mut Vec<LinearBVHNode>, node: &Box<BVHBuildNode>) -> usize {
    let mut linear_node = LinearBVHNode::new(node);
    let offset = nodes.len();
    if node.n_primitives > 0 {
        linear_node.offset = node.first_prim_offset;
        linear_node.n_primitives = node.n_primitives;
        nodes.push(linear_node);
    } else {
        linear_node.n_primitives = 0;
        nodes.push(linear_node);
        if let Some(c) = node.children[0].as_ref() {
            let _ = flatten_bvh_tree(nodes, c);
        }
        if let Some(c) = node.children[1].as_ref() {
            nodes[offset].offset = flatten_bvh_tree(nodes, c);
        }
    }
    return offset;
}

fn sign(x: Float) -> usize {
    if x.is_sign_negative() {
        1
    } else {
        0
    }
}

#[derive(Clone)]
pub struct LBVHTree {
    pub nodes: Vec<LinearBVHNode>,
    bounds: Bounds3f,
}

impl LBVHTree {
    pub fn new(
        count: usize,
        bounds: impl Fn(usize) -> Bounds3f,
        max_prims_in_node: usize,
        split_method: SplitMethod,
    ) -> (Self, Vec<usize>) {
        let max_prims_in_node = usize::min(max_prims_in_node, 255);
        let mut order = Vec::new();
        let root =
            create_bvh_node_from_bounds(&mut order, count, bounds, max_prims_in_node, split_method);
        let tree_bounds = root.bounds;
        let mut nodes = Vec::with_capacity(root.node_count());
        flatten_bvh_tree(&mut nodes, &root);
        (
            Self {
                nodes,
                bounds: tree_bounds,
            },
            order,
        )
    }

    pub fn bounds(&self) -> Bounds3f {
        self.bounds
    }

    pub fn intersect<H>(
        &self,
        r: &Ray,
        t_max: Float,
        mut leaf: impl FnMut(Range<usize>, Float) -> Option<(Float, H)>,
    ) -> Option<H> {
        let mut result = None;
        let mut cutoff = t_max;
        let mut stack = vec![(0usize, 0.0, t_max)];
        let org = [r.o.x, r.o.y, r.o.z];
        let idir = [
            Float::recip(r.d.x),
            Float::recip(r.d.y),
            Float::recip(r.d.z),
        ];
        let dir_is_neg = [sign(idir[0]), sign(idir[1]), sign(idir[2])];
        while let Some((node_index, t0, t1)) = stack.pop() {
            let t1 = Float::min(cutoff, t1);
            if t1 < t0 {
                continue;
            }
            let node = &self.nodes[node_index];
            if let Some((near, far)) = intersect_box_array_i(
                &node.bounds[0],
                &node.bounds[1],
                &org,
                &idir,
                &dir_is_neg,
                t0,
                t1,
            ) {
                if node.n_primitives > 0 {
                    let start = node.offset;
                    let end = start + node.n_primitives;
                    if let Some((hit_t, hit)) = leaf(start..end, cutoff) {
                        cutoff = hit_t;
                        result = Some(hit);
                    }
                } else {
                    let children = [node_index + 1, node.offset];
                    let indices = [
                        dir_is_neg[node.axis as usize],
                        1 - dir_is_neg[node.axis as usize],
                    ];
                    stack.push((children[indices[1]], near, far));
                    stack.push((children[indices[0]], near, far));
                }
            }
        }
        result
    }

    pub fn intersect_p(
        &self,
        r: &Ray,
        t_max: Float,
        mut leaf: impl FnMut(Range<usize>, Float) -> bool,
    ) -> bool {
        let mut stack = vec![(0usize, 0.0, t_max)];
        let org = [r.o.x, r.o.y, r.o.z];
        let idir = [
            Float::recip(r.d.x),
            Float::recip(r.d.y),
            Float::recip(r.d.z),
        ];
        let dir_is_neg = [sign(idir[0]), sign(idir[1]), sign(idir[2])];
        while let Some((node_index, t0, t1)) = stack.pop() {
            let node = &self.nodes[node_index];
            if let Some((near, far)) = intersect_box_array_i(
                &node.bounds[0],
                &node.bounds[1],
                &org,
                &idir,
                &dir_is_neg,
                t0,
                t1,
            ) {
                if node.n_primitives > 0 {
                    if leaf(node.offset..node.offset + node.n_primitives, t_max) {
                        return true;
                    }
                } else {
                    let children = [node_index + 1, node.offset];
                    let indices = [
                        dir_is_neg[node.axis as usize],
                        1 - dir_is_neg[node.axis as usize],
                    ];
                    stack.push((children[indices[1]], near, far));
                    stack.push((children[indices[0]], near, far));
                }
            }
        }
        false
    }
}

#[derive(Clone)]
pub struct LBVHAccel {
    pub primitives: Vec<Arc<Primitive>>,
    pub tree: LBVHTree,
}

impl LBVHAccel {
    pub fn new(
        prims: &[Arc<Primitive>],
        max_prims_in_node: usize,
        split_method: SplitMethod,
    ) -> Self {
        let (tree, order) = LBVHTree::new(
            prims.len(),
            |i| prims[i].bounds(),
            max_prims_in_node,
            split_method,
        );
        let primitives = order.iter().map(|i| prims[*i].clone()).collect();
        LBVHAccel { primitives, tree }
    }

    pub fn bounds(&self) -> Bounds3f {
        self.tree.bounds()
    }

    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        self.tree.intersect(r, t_max, |range, max| {
            let mut best = None;
            let mut cutoff = max;
            for primitive in &self.primitives[range] {
                if let Some(hit) = primitive.intersect(r, cutoff) {
                    cutoff = hit.t_hit;
                    best = Some((hit.t_hit, hit));
                }
            }
            best
        })
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        self.tree.intersect_p(r, t_max, |range, max| {
            self.primitives[range].iter().any(|p| p.intersect_p(r, max))
        })
    }
}
