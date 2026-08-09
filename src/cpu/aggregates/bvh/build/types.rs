use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::stats::*;

thread_local!(static INTERIOR_NODES: StatCounter = StatCounter::new("BVH/Interior nodes"));
thread_local!(static LEAF_NODES: StatCounter = StatCounter::new("BVH/Leaf nodes"));

#[derive(Copy, Clone, Debug)]
pub enum SplitMethod {
    SAH,
    HLBVH,
    Middle,
    EqualCounts,
}

#[derive(Copy, Clone)]
pub struct BVHPrimitiveInfo {
    pub primitive_number: usize,
    pub bounds: Bounds3f,
    pub centroid: Point3f,
}

impl BVHPrimitiveInfo {
    pub fn new(i: usize, b: &Bounds3f, c: &Point3f) -> Self {
        BVHPrimitiveInfo {
            primitive_number: i,
            bounds: *b,
            centroid: *c,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BVHBuildNode {
    pub bounds: Bounds3f,
    pub children: [Option<Box<BVHBuildNode>>; 2],
    pub split_axis: u8,
    pub first_prim_offset: usize,
    pub n_primitives: usize,
}

impl BVHBuildNode {
    pub fn new() -> Self {
        BVHBuildNode {
            bounds: Bounds3f::default(),
            children: [None, None],
            split_axis: 0,
            first_prim_offset: 0,
            n_primitives: 0,
        }
    }

    pub fn init_leaf(first: usize, n: usize, b: &Bounds3f) -> Self {
        LEAF_NODES.with(|c| c.inc());

        BVHBuildNode {
            bounds: b.clone(),
            children: [None, None],
            split_axis: 0,
            first_prim_offset: first,
            n_primitives: n,
        }
    }

    fn get_bounds_helper(
        c0: &Option<Box<BVHBuildNode>>,
        c1: &Option<Box<BVHBuildNode>>,
    ) -> Bounds3f {
        match (c0, c1) {
            (Some(n0), Some(n1)) => n0.bounds.union(&n1.bounds),
            (Some(n0), None) => n0.bounds,
            (None, Some(n1)) => n1.bounds,
            (None, None) => {
                log::error!(
                    "BVHBuildNode::init_interior called with no children; using default bounds"
                );
                Bounds3f::default()
            }
        }
    }

    pub fn init_interior(
        axis: usize,
        c0: Option<Box<BVHBuildNode>>,
        c1: Option<Box<BVHBuildNode>>,
    ) -> Self {
        INTERIOR_NODES.with(|c| c.inc());

        let b = Self::get_bounds_helper(&c0, &c1);
        BVHBuildNode {
            bounds: b,
            children: [c0, c1],
            split_axis: axis as u8,
            first_prim_offset: 0,
            n_primitives: 0,
        }
    }

    pub fn node_count(&self) -> usize {
        let mut count = 1;
        if self.n_primitives == 0 {
            if let Some(c) = self.children[0].as_ref() {
                count += c.node_count();
            }
            if let Some(c) = self.children[1].as_ref() {
                count += c.node_count();
            }
        }
        return count;
    }

    pub fn primitive_count(&self) -> usize {
        if self.n_primitives == 0 {
            let mut count = 0;
            if let Some(c0) = self.children[0].as_ref() {
                count += c0.primitive_count();
            }
            if let Some(c1) = self.children[1].as_ref() {
                count += c1.primitive_count();
            }
            return count;
        } else {
            return self.n_primitives as usize;
        }
    }
}
