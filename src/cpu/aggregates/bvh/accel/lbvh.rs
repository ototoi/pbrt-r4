use super::super::build::*;
use crate::cpu::primitive::*;
use crate::interaction::*;
use crate::util::base::*;
use crate::util::geometry::*;

use log::*;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct LinearBVHNode {
    pub bounds: [[Float; 3]; 2], //32*3*2
    pub offset: usize,
    pub n_primitives: usize, // 0 -> interior node
    pub axis: u8,            // interior node: xyz
}

fn to_bounds3f(b: &[[Float; 3]; 2]) -> Bounds3f {
    return Bounds3f::from((
        (b[0][0] as Float, b[0][1] as Float, b[0][2] as Float),
        (b[1][0] as Float, b[1][1] as Float, b[1][2] as Float),
    ));
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
pub struct LBVHAccel {
    pub primitives: Vec<Arc<Primitive>>,
    pub nodes: Vec<LinearBVHNode>,
}

impl LBVHAccel {
    pub fn new(
        prims: &[Arc<Primitive>],
        max_prims_in_node: usize,
        split_method: SplitMethod,
    ) -> Self {
        let max_prims_in_node = usize::min(max_prims_in_node, 255);
        let mut orderd_prims = Vec::new();
        let root = create_bvh_node(&mut orderd_prims, prims, max_prims_in_node, split_method);

        let total_nodes = root.node_count();
        let allocated_memory = total_nodes * std::mem::size_of::<LinearBVHNode>();
        info!(
            "BVH created with {} nodes for {} primitives ({:.2} MB)",
            total_nodes,
            prims.len(),
            allocated_memory as Float / (1024.0 * 1024.0)
        );

        let mut nodes = Vec::new();
        nodes.reserve(root.node_count());
        flatten_bvh_tree(&mut nodes, &root);
        LBVHAccel {
            primitives: orderd_prims,
            nodes,
        }
    }

    pub fn bounds(&self) -> Bounds3f {
        return to_bounds3f(&(self.nodes[0].bounds));
    }

    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let mut isect = None;
        let mut nodes_to_visit: Vec<(usize, Float, Float)> = Vec::with_capacity(16);
        let mut t_max = t_max;
        let t0: Float = 0.0;
        let t1: Float = t_max;
        let org = [r.o.x, r.o.y, r.o.z];
        let idir = [
            Float::recip(r.d.x),
            Float::recip(r.d.y),
            Float::recip(r.d.z),
        ];
        let dir_is_neg = [sign(idir[0]), sign(idir[1]), sign(idir[2])];
        nodes_to_visit.push((0, t0, t1));
        while let Some((current_node_index, t0, mut t1)) = nodes_to_visit.pop() {
            t1 = Float::min(t_max, t1);
            if t1 < t0 {
                continue;
            }
            assert!(t0 <= t1);
            let node = &self.nodes[current_node_index];
            let min = &node.bounds[0];
            let max = &node.bounds[1];
            if let Some((t0, t1)) =
                intersect_box_array_i(min, max, &org, &idir, &dir_is_neg, t0, t1)
            {
                if node.n_primitives > 0 {
                    let start = node.offset as usize;
                    let end = start + (node.n_primitives as usize);
                    for i in start..end {
                        let prim = self.primitives[i].as_ref();
                        if let Some(isect_n) = prim.intersect(r, t_max) {
                            t_max = isect_n.t_hit;
                            isect = Some(isect_n);
                        }
                    }
                } else {
                    let index_children = [current_node_index + 1, node.offset as usize];
                    let indices = [
                        dir_is_neg[node.axis as usize],
                        1 - dir_is_neg[node.axis as usize],
                    ];
                    nodes_to_visit.push((index_children[indices[1]], t0, t1));
                    nodes_to_visit.push((index_children[indices[0]], t0, t1));
                }
            }
        }
        return isect;
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        let mut nodes_to_visit: Vec<(usize, Float, Float)> = Vec::with_capacity(16);
        let t0: Float = 0.0;
        let t1: Float = t_max;
        let org = [r.o.x, r.o.y, r.o.z];
        let idir = [
            Float::recip(r.d.x),
            Float::recip(r.d.y),
            Float::recip(r.d.z),
        ];
        let dir_is_neg = [sign(idir[0]), sign(idir[1]), sign(idir[2])];
        nodes_to_visit.push((0, t0, t1));
        while let Some((current_node_index, t0, t1)) = nodes_to_visit.pop() {
            //t1 = Float::min(t_max, t1);
            //    continue;
            let node = &self.nodes[current_node_index];
            let min = &node.bounds[0];
            let max = &node.bounds[1];
            if let Some((t0, t1)) =
                intersect_box_array_i(min, max, &org, &idir, &dir_is_neg, t0, t1)
            {
                if node.n_primitives > 0 {
                    let start = node.offset as usize;
                    let end = start + (node.n_primitives as usize);
                    for i in start..end {
                        let p = self.primitives[i].as_ref();
                        if p.intersect_p(r, t_max) {
                            return true;
                        }
                    }
                } else {
                    let index_children = [current_node_index + 1, node.offset as usize];
                    let indices = [
                        dir_is_neg[node.axis as usize],
                        1 - dir_is_neg[node.axis as usize],
                    ];
                    nodes_to_visit.push((index_children[indices[1]], t0, t1));
                    nodes_to_visit.push((index_children[indices[0]], t0, t1));
                }
            }
        }
        return false;
    }
}
