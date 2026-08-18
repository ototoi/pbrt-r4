use crate::cpu::aggregates::Accel;
use crate::cpu::primitive::*;
use crate::interaction::*;
use crate::paramdict::*;
use crate::shapes::*;
use crate::util::error::*;
use crate::util::geometry::*;
use std::sync::Arc;

// KDTreeAccel Local Declarations
#[derive(Clone, Copy, Debug)]
enum KDAccelNode {
    Leaf {
        primitive_indices_offset: usize,
        n_primitives: usize,
    },
    Interior {
        axis: u8,
        split: Float,
        above_child: usize,
    },
}

impl KDAccelNode {
    fn init_leaf(prim_nums: &[usize], primitive_indices: &mut Vec<usize>) -> Self {
        let primitive_indices_offset = primitive_indices.len();
        let n_primitives = prim_nums.len();
        for i in 0..prim_nums.len() {
            primitive_indices.push(prim_nums[i]);
        }
        KDAccelNode::Leaf {
            primitive_indices_offset,
            n_primitives,
        }
    }
    fn init_interior(axis: u8, split: Float, above_child: usize) -> Self {
        KDAccelNode::Interior {
            axis,
            split,
            above_child,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum EdgeType {
    Start,
    End,
}

impl Default for EdgeType {
    fn default() -> Self {
        EdgeType::Start
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct BoundEdge {
    t: Float,
    prim_num: usize,
    edge_type: EdgeType,
}

struct KDTreeBuilder {
    pub isect_cost: i32,
    pub traversal_cost: i32,
    pub max_prims: usize,
    pub empty_bonus: Float,
}

impl KDTreeBuilder {
    pub fn build_tree(
        &mut self,
        node_bounds: &Bounds3f,
        all_prim_bounds: &[Bounds3f],
        prim_nums: &[usize],
        nodes: &mut Vec<KDAccelNode>,
        primitive_indices: &mut Vec<usize>,
        depth: i32,
        bad_refines: u32,
    ) -> usize {
        let node_num = nodes.len();
        // Initialize leaf node if termination criteria met
        let n_primitives = prim_nums.len();
        if n_primitives <= self.max_prims || depth == 0 {
            nodes.push(KDAccelNode::init_leaf(prim_nums, primitive_indices));
            return node_num;
        }

        // Choose split axis position for interior node
        if let Some((best_axis, t_split, primes0, primes1, bad_refines)) =
            self.choose_split_axis(node_bounds, all_prim_bounds, prim_nums, bad_refines)
        {
            assert!(best_axis < 3);

            // Recursively initialize children nodes
            let mut bounds0 = node_bounds.clone();
            let mut bounds1 = node_bounds.clone();
            bounds0.max[best_axis] = t_split;
            bounds1.min[best_axis] = t_split;
            nodes.push(KDAccelNode::init_interior(best_axis as u8, t_split, 0)); //dummy value

            let left_index = self.build_tree(
                &bounds0,
                all_prim_bounds,
                &primes0,
                nodes,
                primitive_indices,
                depth - 1,
                bad_refines,
            );
            let right_index = self.build_tree(
                &bounds1,
                all_prim_bounds,
                &primes1,
                nodes,
                primitive_indices,
                depth - 1,
                bad_refines,
            );
            assert_eq!(node_num + 1, left_index);
            nodes[node_num] = KDAccelNode::init_interior(best_axis as u8, t_split, right_index);
        } else {
            nodes.push(KDAccelNode::init_leaf(prim_nums, primitive_indices));
        }
        return node_num;
    }

    fn choose_split_axis(
        &self,
        node_bounds: &Bounds3f,
        all_prim_bounds: &[Bounds3f],
        prim_nums: &[usize],
        bad_refines: u32,
    ) -> Option<(usize, Float, Vec<usize>, Vec<usize>, u32)> {
        let n_primitives = prim_nums.len();

        // Choose split axis position for interior node
        let mut best_axis: i32 = -1;
        let mut best_offset: i64 = -1;
        let mut best_cost = Float::INFINITY;
        let old_cost = self.isect_cost as Float * n_primitives as Float;
        let total_sa = node_bounds.surface_area();
        let inv_total_sa = 1.0 / total_sa;
        let traversal_cost = self.traversal_cost as Float;
        let isect_cost = self.isect_cost as Float;
        let empty_bonus = self.empty_bonus;
        let d = node_bounds.max - node_bounds.min;

        // Initialize interior node and continue recursion
        let mut edges = vec![vec![BoundEdge::default(); 2 * n_primitives]; 3];

        // Choose which axis to split along
        let mut axis = node_bounds.maximum_extent();
        let mut retries = 0;
        loop {
            // Initialize edges for _axis_
            for i in 0..n_primitives {
                let pn = prim_nums[i];
                let bounds = &all_prim_bounds[pn];
                edges[axis][2 * i + 0] = BoundEdge {
                    t: bounds.min[axis],
                    prim_num: pn,
                    edge_type: EdgeType::Start,
                };
                edges[axis][2 * i + 1] = BoundEdge {
                    t: bounds.max[axis],
                    prim_num: pn,
                    edge_type: EdgeType::End,
                };
            }

            // Sort _edges_ for _axis_
            edges[axis].sort_by(|a, b| {
                if a.t == b.t {
                    return a.edge_type.cmp(&b.edge_type);
                } else {
                    return a.t.total_cmp(&b.t);
                }
            });

            let n_edges = 2 * n_primitives;

            // Compute cost of all splits for _axis_ to find best
            let mut n_below = 0;
            let mut n_above = n_primitives;
            for i in 0..n_edges {
                if edges[axis][i].edge_type == EdgeType::End {
                    n_above -= 1;
                }
                let edge_t = edges[axis][i].t;
                if edge_t > node_bounds.min[axis] && edge_t < node_bounds.max[axis] {
                    //Compute cost for split at _i_th edge
                    let other_axis0 = (axis + 1) % 3;
                    let other_axis1 = (axis + 2) % 3;
                    let below_sa = 2.0
                        * (d[other_axis0] * d[other_axis1]
                            + (edge_t - node_bounds.min[axis]) * (d[other_axis0] + d[other_axis1]));
                    let above_sa = 2.0
                        * (d[other_axis0] * d[other_axis1]
                            + (node_bounds.max[axis] - edge_t) * (d[other_axis0] + d[other_axis1]));
                    let p_below = below_sa * inv_total_sa;
                    let p_above = above_sa * inv_total_sa;
                    let eb = if n_above == 0 || n_below == 0 {
                        empty_bonus
                    } else {
                        0.0
                    };
                    let cost = traversal_cost
                        * isect_cost
                        * (1.0 - eb)
                        * (p_below * n_below as Float + p_above * n_above as Float);
                    if cost < best_cost {
                        best_cost = cost;
                        best_axis = axis as i32;
                        best_offset = i as i64;
                    }
                }
                if edges[axis][i].edge_type == EdgeType::Start {
                    n_below += 1;
                }
            }

            // Create leaf if no good splits were found
            if best_axis < 0 && retries < 2 {
                retries += 1;
                axis = (axis + 1) % 3;
                continue;
            }
            break;
        }

        let mut bad_refines = bad_refines;
        if best_cost > old_cost {
            bad_refines += 1;
        }
        if (best_cost > 4.0 * old_cost && n_primitives < 16) || bad_refines == 3 {
            return None;
        }

        if best_axis >= 0 && best_offset >= 0 {
            let best_axis = best_axis as usize;
            let best_offset = best_offset as usize;
            let t_split = edges[best_axis][best_offset].t;

            // Classify primitives with respect to split
            let mut primes0 = Vec::new();
            let mut primes1 = Vec::new();
            for i in 0..best_offset {
                if edges[best_axis][i].edge_type == EdgeType::Start {
                    primes0.push(edges[best_axis][i].prim_num);
                }
            }
            for i in (best_offset + 1)..(2 * n_primitives) {
                if edges[best_axis][i].edge_type == EdgeType::End {
                    primes1.push(edges[best_axis][i].prim_num);
                }
            }

            let length0 = primes0.len();
            let length1 = primes1.len();
            if length0 == 0 || length1 == 0 {
                return None;
            }

            return Some((best_axis, t_split, primes0, primes1, bad_refines));
        } else {
            return None;
        }
    }
}

#[derive(Clone)]
pub struct KDTreeAccel {
    pub primitives: Vec<Arc<Primitive>>,
    pub primitive_indices: Vec<usize>,
    nodes: Vec<KDAccelNode>,
    pub bounds: Bounds3f,
}

impl KDTreeAccel {
    pub fn new(
        prims: &[Arc<Primitive>],
        isect_cost: i32,
        traversal_cost: i32,
        empty_bonus: Float,
        max_primes: i32,
        max_depth: i32,
    ) -> Self {
        // Build kd-tree for accelerator
        let primitives: Vec<Arc<Primitive>> = prims.to_vec();
        if primitives.is_empty() {
            let mut primitive_indices = Vec::new();
            let nodes = vec![KDAccelNode::init_leaf(&[], &mut primitive_indices)];
            return KDTreeAccel {
                primitives,
                primitive_indices,
                nodes,
                bounds: Bounds3f::default(),
            };
        }

        let max_depth = if max_depth <= 0 {
            (8.0 + 1.3 * (prims.len() as f32).log2()).round() as i32
        } else {
            max_depth
        };

        // Compute bounds for kd-tree construction
        let mut bounds = prims[0].bounds();
        let mut prim_bounds: Vec<Bounds3f> = Vec::with_capacity(primitives.len());
        for prim in primitives.iter() {
            let b = prim.bounds();
            bounds = Bounds3f::union(&bounds, &b);
            prim_bounds.push(b);
        }

        // Initialize _primNums_ for kd-tree construction
        let mut prim_nums = vec![0; primitives.len()];
        for i in 0..primitives.len() {
            prim_nums[i] = i;
        }

        // Start recursive construction of kd-tree
        let mut builder = KDTreeBuilder {
            isect_cost,
            traversal_cost,
            max_prims: max_primes as usize,
            empty_bonus,
        };
        let mut nodes = Vec::new();
        let mut primitive_indices = Vec::new();
        let root_index = builder.build_tree(
            &bounds,
            &prim_bounds,
            &mut prim_nums,
            &mut nodes,
            &mut primitive_indices,
            max_depth,
            0,
        );
        assert!(root_index == 0);
        return KDTreeAccel {
            primitives,
            primitive_indices,
            nodes,
            bounds,
        };
    }
}

type KDTodo = (usize, Float, Float);

#[inline]
fn safe_recip(x: Float) -> Float {
    if x != 0.0 {
        Float::recip(x)
    } else {
        Float::INFINITY.copysign(x)
    }
}

impl KDTreeAccel {
    pub fn bounds(&self) -> Bounds3f {
        return self.bounds;
    }

    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        // Compute initial parametric range of ray inside kd-tree extent
        let (t_min, t_max) = self.bounds.intersect_p(r, t_max)?;

        let mut isect = None;
        // Prepare to traverse kd-tree for ray
        let inv_dir = [safe_recip(r.d.x), safe_recip(r.d.y), safe_recip(r.d.z)];
        const MAX_STACK: usize = 64;
        let mut todo: [KDTodo; MAX_STACK] = [(0, 0.0, 0.0); MAX_STACK];
        let mut todo_pos: i32 = 0;
        todo[todo_pos as usize] = (0, t_min, t_max);
        while todo_pos >= 0 {
            let (node_num, t_min, t_max) = todo[todo_pos as usize];
            todo_pos -= 1;
            let node = &self.nodes[node_num];
            // Bail out if we found a hit closer than the current node
            if t_max < t_min {
                break;
            }
            match node {
                KDAccelNode::Interior {
                    axis,
                    split,
                    above_child,
                } => {
                    // Process kd-tree interior node

                    // Compute parametric distance along ray to split plane
                    let split = *split;
                    let axis = *axis as usize;
                    let above_child = *above_child as usize;
                    let t_plane = (split - r.o[axis]) * inv_dir[axis];
                    let below_first = r.o[axis] < split || (r.o[axis] == split && r.d[axis] <= 0.0);
                    let (first_index, second_index) = if below_first {
                        (node_num + 1, above_child)
                    } else {
                        (above_child, node_num + 1)
                    };

                    // Advance to next child node, possibly enqueue other child
                    if t_plane > t_max || t_plane < 0.0 {
                        todo_pos += 1;
                        todo[todo_pos as usize] = (first_index, t_min, t_max);
                    } else if t_plane < t_min {
                        todo_pos += 1;
                        todo[todo_pos as usize] = (second_index, t_min, t_max);
                    } else {
                        todo_pos += 1;
                        todo[todo_pos as usize] = (second_index, t_plane, t_max);
                        todo_pos += 1;
                        todo[todo_pos as usize] = (first_index, t_min, t_plane);
                    }
                }
                KDAccelNode::Leaf {
                    primitive_indices_offset,
                    n_primitives,
                } => {
                    // Check for intersections inside leaf node
                    let primitive_indices_offset = *primitive_indices_offset;
                    let n_primitives = *n_primitives;
                    for i in 0..n_primitives {
                        let index = self.primitive_indices[primitive_indices_offset + i];
                        let prim = &self.primitives[index];
                        if let Some(isect_n) = prim.intersect(r, t_max) {
                            isect = Some(isect_n);
                        }
                    }
                }
            }
        }
        return isect;
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        // Compute initial parametric range of ray inside kd-tree extent
        let (t_min, t_max) = if let Some((t_min, t_max)) = self.bounds.intersect_p(r, t_max) {
            (t_min, t_max)
        } else {
            return false;
        };

        // Prepare to traverse kd-tree for ray
        let inv_dir = [safe_recip(r.d.x), safe_recip(r.d.y), safe_recip(r.d.z)];
        const MAX_STACK: usize = 64;
        let mut todo: [KDTodo; MAX_STACK] = [(0, 0.0, 0.0); MAX_STACK];
        let mut todo_pos: i32 = 0;
        todo[todo_pos as usize] = (0, t_min, t_max);
        while todo_pos >= 0 {
            let (node_num, t_min, t_max) = todo[todo_pos as usize];
            todo_pos -= 1;
            let node = &self.nodes[node_num];
            // Bail out if we found a hit closer than the current node
            if t_max < t_min {
                break;
            }
            match node {
                KDAccelNode::Interior {
                    split,
                    axis,
                    above_child,
                } => {
                    // Process kd-tree interior node

                    // Compute parametric distance along ray to split plane
                    let split = *split;
                    let axis = *axis as usize;
                    let above_child = *above_child as usize;
                    let t_plane = (split - r.o[axis]) * inv_dir[axis];
                    let below_first = r.o[axis] < split || (r.o[axis] == split && r.d[axis] <= 0.0);
                    let (first_index, second_index) = if below_first {
                        (node_num + 1, above_child)
                    } else {
                        (above_child, node_num + 1)
                    };

                    // Advance to next child node, possibly enqueue other child
                    if t_plane > t_max || t_plane < 0.0 {
                        todo_pos += 1;
                        todo[todo_pos as usize] = (first_index, t_min, t_max);
                    } else if t_plane < t_min {
                        todo_pos += 1;
                        todo[todo_pos as usize] = (second_index, t_min, t_max);
                    } else {
                        todo_pos += 1;
                        todo[todo_pos as usize] = (second_index, t_plane, t_max);
                        todo_pos += 1;
                        todo[todo_pos as usize] = (first_index, t_min, t_plane);
                    }
                }
                KDAccelNode::Leaf {
                    primitive_indices_offset,
                    n_primitives,
                } => {
                    // Check for intersections inside leaf node

                    let primitive_indices_offset = *primitive_indices_offset;
                    let n_primitives = *n_primitives;
                    for i in 0..n_primitives {
                        let index = self.primitive_indices[primitive_indices_offset + i];
                        if self.primitives[index].intersect_p(r, t_max) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }
}

pub fn create_kdtree_accelerator(
    prims: &[Arc<Primitive>],
    params: &ParameterDictionary,
) -> Result<Primitive, PbrtError> {
    let isect_cost = params.get_one_int("intersectcost", 80);
    let trav_cost = params.get_one_int("traversalcost", 1);
    let empty_bonus = params.get_one_float("emptybonus", 0.5);
    let max_primes = params.get_one_int("maxprims", 1);
    let max_depth = params.get_one_int("maxdepth", -1);
    return Ok(Primitive::Accel(Accel::KdTree(Arc::new(KDTreeAccel::new(
        prims,
        isect_cost,
        trav_cost,
        empty_bonus,
        max_primes,
        max_depth,
    )))));
}
