use super::{LightBounds, LightRecord, INVALID_INDEX};
use crate::util::error::PbrtError;

const N_BUCKETS: usize = 12;

#[derive(Clone, Debug, PartialEq)]
pub struct LightBVH {
    pub all_bounds: Option<super::Bounds3>,
    pub nodes: Vec<LightBVHNode>,
    pub handle_to_leaf: Vec<u32>,
    pub bounded_handles: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LightBVHNode {
    Leaf {
        light_handle: u32,
        bounds: LightBounds,
        parent: u32,
    },
    Interior {
        left_child: u32,
        right_child: u32,
        bounds: LightBounds,
        parent: u32,
    },
}

impl LightBVHNode {
    pub fn bounds(&self) -> LightBounds {
        match self {
            Self::Leaf { bounds, .. } | Self::Interior { bounds, .. } => *bounds,
        }
    }

    pub fn parent(&self) -> u32 {
        match self {
            Self::Leaf { parent, .. } | Self::Interior { parent, .. } => *parent,
        }
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }

    pub fn light_handle(&self) -> Option<u32> {
        match self {
            Self::Leaf { light_handle, .. } => Some(*light_handle),
            Self::Interior { .. } => None,
        }
    }

    pub fn right_child(&self) -> Option<u32> {
        match self {
            Self::Leaf { .. } => None,
            Self::Interior { right_child, .. } => Some(*right_child),
        }
    }
}

pub fn build_light_bvh(
    lights: &[LightRecord],
    light_bounds: &[LightBounds],
) -> Result<LightBVH, PbrtError> {
    if lights.len() != light_bounds.len() {
        return Err(PbrtError::error(
            "Light records and light bounds must have equal lengths.",
        ));
    }
    if lights.len() > u32::MAX as usize {
        return Err(PbrtError::error("Light count exceeds the u32 range."));
    }

    for bounds in light_bounds.iter().copied() {
        bounds.validate()?;
    }

    let mut bounded_handles = Vec::new();
    let mut all_bounds: Option<super::Bounds3> = None;
    for (handle, bounds) in light_bounds.iter().copied().enumerate() {
        if bounds.phi > 0.0 {
            let handle = u32::try_from(handle)
                .map_err(|_| PbrtError::error("Light handle exceeds the u32 range."))?;
            bounded_handles.push(handle);
            all_bounds = Some(match all_bounds {
                Some(existing) => existing.union(bounds.bounds),
                None => bounds.bounds,
            });
        }
    }

    let mut bvh = LightBVH {
        all_bounds,
        nodes: Vec::new(),
        handle_to_leaf: vec![INVALID_INDEX; lights.len()],
        bounded_handles,
    };
    if !bvh.bounded_handles.is_empty() {
        let handles = bvh.bounded_handles.clone();
        build_subtree(&mut bvh, lights, light_bounds, handles, INVALID_INDEX)?;
    }
    Ok(bvh)
}

/// Samples one bounded light using the same one-child traversal as the GPU
/// sampler. The returned PMF is the product of the decisions along the path.
pub fn sample_light_bvh(
    bvh: &LightBVH,
    p: [f32; 3],
    n: [f32; 3],
    mut u: f32,
) -> Result<Option<(u32, f32)>, PbrtError> {
    if !u.is_finite() || !(0.0..=1.0).contains(&u) {
        return Err(PbrtError::error("Light BVH sample coordinate is invalid."));
    }
    if bvh.nodes.is_empty() {
        return Ok(None);
    }
    u = u.min(1.0 - f32::EPSILON);
    let mut node_index = 0_u32;
    let mut pmf = 1.0;
    for _ in 0..bvh.nodes.len() {
        let node = bvh
            .nodes
            .get(node_index as usize)
            .ok_or_else(|| PbrtError::error("Light BVH traversal reached an invalid node."))?;
        if let Some(handle) = node.light_handle() {
            return Ok(Some((handle, pmf)));
        }
        let left = node_index + 1;
        let right = node
            .right_child()
            .ok_or_else(|| PbrtError::error("Light BVH interior node has no right child."))?;
        let left_importance = bvh.nodes[left as usize].bounds().importance(p, n)?;
        let right_importance = bvh.nodes[right as usize].bounds().importance(p, n)?;
        let total = left_importance + right_importance;
        if total <= 0.0 {
            return Ok(None);
        }
        let left_pmf = left_importance / total;
        if u < left_pmf {
            pmf *= left_pmf;
            u = (u / left_pmf).min(1.0 - f32::EPSILON);
            node_index = left;
        } else {
            pmf *= 1.0 - left_pmf;
            u = ((u - left_pmf) / (1.0 - left_pmf)).min(1.0 - f32::EPSILON);
            node_index = right;
        }
    }
    Err(PbrtError::error(
        "Light BVH traversal exceeded its node limit.",
    ))
}

/// Computes the PMF of a global light handle by walking its leaf parents.
pub fn light_bvh_pmf(
    bvh: &LightBVH,
    p: [f32; 3],
    n: [f32; 3],
    handle: u32,
) -> Result<f32, PbrtError> {
    let Some(&leaf_index) = bvh.handle_to_leaf.get(handle as usize) else {
        return Ok(0.0);
    };
    if leaf_index == INVALID_INDEX {
        return Ok(0.0);
    }
    let mut node_index = leaf_index;
    let mut pmf = 1.0;
    for _ in 0..bvh.nodes.len() {
        let node = bvh
            .nodes
            .get(node_index as usize)
            .ok_or_else(|| PbrtError::error("Light BVH PMF reached an invalid leaf."))?;
        let parent = node.parent();
        if parent == INVALID_INDEX {
            return Ok(pmf);
        }
        let parent_node = bvh
            .nodes
            .get(parent as usize)
            .ok_or_else(|| PbrtError::error("Light BVH PMF reached an invalid parent."))?;
        let left = parent + 1;
        let right = parent_node
            .right_child()
            .ok_or_else(|| PbrtError::error("Light BVH PMF parent has no right child."))?;
        let left_importance = bvh.nodes[left as usize].bounds().importance(p, n)?;
        let right_importance = bvh.nodes[right as usize].bounds().importance(p, n)?;
        let total = left_importance + right_importance;
        if total <= 0.0 {
            return Ok(0.0);
        }
        if node_index == left {
            pmf *= left_importance / total;
        } else if node_index == right {
            pmf *= right_importance / total;
        } else {
            return Err(PbrtError::error(
                "Light BVH PMF child is not owned by its parent.",
            ));
        }
        node_index = parent;
    }
    Err(PbrtError::error("Light BVH PMF exceeded its node limit."))
}

fn build_subtree(
    bvh: &mut LightBVH,
    _lights: &[LightRecord],
    light_bounds: &[LightBounds],
    handles: Vec<u32>,
    parent: u32,
) -> Result<u32, PbrtError> {
    if handles.is_empty() {
        return Err(PbrtError::error("Light BVH subtree cannot be empty."));
    }
    if handles.len() == 1 {
        let node_index = node_index(bvh.nodes.len())?;
        let handle = handles[0] as usize;
        let bounds = light_bounds[handle];
        bvh.nodes.push(LightBVHNode::Leaf {
            light_handle: handles[0],
            bounds,
            parent,
        });
        bvh.handle_to_leaf[handle] = node_index;
        return Ok(node_index);
    }

    let split = choose_split(&handles, light_bounds);
    let (left_handles, right_handles) = partition_handles(handles, light_bounds, split);
    if left_handles.is_empty() || right_handles.is_empty() {
        return Err(PbrtError::error("Light BVH split produced an empty child."));
    }

    let node_index = node_index(bvh.nodes.len())?;
    bvh.nodes.push(LightBVHNode::Interior {
        left_child: INVALID_INDEX,
        right_child: INVALID_INDEX,
        bounds: light_bounds[left_handles[0] as usize],
        parent,
    });
    let left_child = build_subtree(bvh, _lights, light_bounds, left_handles, node_index)?;
    let right_child = build_subtree(bvh, _lights, light_bounds, right_handles, node_index)?;
    let bounds = bvh.nodes[left_child as usize]
        .bounds()
        .union(bvh.nodes[right_child as usize].bounds())?;
    bvh.nodes[node_index as usize] = LightBVHNode::Interior {
        left_child,
        right_child,
        bounds,
        parent,
    };
    Ok(node_index)
}

#[derive(Clone, Copy)]
struct Split {
    dimension: usize,
    bucket: usize,
}

fn choose_split(handles: &[u32], light_bounds: &[LightBounds]) -> Split {
    let mut centroid_min = [f32::INFINITY; 3];
    let mut centroid_max = [f32::NEG_INFINITY; 3];
    for &handle in handles {
        let centroid = light_bounds[handle as usize].bounds.centroid();
        for dimension in 0..3 {
            centroid_min[dimension] = centroid_min[dimension].min(centroid[dimension]);
            centroid_max[dimension] = centroid_max[dimension].max(centroid[dimension]);
        }
    }

    let mut combined_bounds = light_bounds[handles[0] as usize].bounds;
    for &handle in &handles[1..] {
        combined_bounds = combined_bounds.union(light_bounds[handle as usize].bounds);
    }

    let mut best = None;
    let mut min_cost = f32::INFINITY;
    for dimension in 0..3 {
        if centroid_min[dimension] == centroid_max[dimension] {
            continue;
        }
        let mut buckets: [Option<LightBounds>; N_BUCKETS] = [None; N_BUCKETS];
        for &handle in handles {
            let bucket = bucket_index(
                light_bounds[handle as usize].bounds.centroid()[dimension],
                centroid_min[dimension],
                centroid_max[dimension],
            );
            buckets[bucket] = Some(match buckets[bucket] {
                Some(existing) => existing.union(light_bounds[handle as usize]).unwrap(),
                None => light_bounds[handle as usize],
            });
        }
        for bucket in 0..N_BUCKETS - 1 {
            let left = union_bucket_range(&buckets, 0, bucket);
            let right = union_bucket_range(&buckets, bucket + 1, N_BUCKETS - 1);
            let cost = left
                .map(|bounds| evaluate_cost(bounds, combined_bounds, dimension))
                .unwrap_or(0.0)
                + right
                    .map(|bounds| evaluate_cost(bounds, combined_bounds, dimension))
                    .unwrap_or(0.0);
            if cost > 0.0 && cost < min_cost {
                min_cost = cost;
                best = Some(Split { dimension, bucket });
            }
        }
    }
    best.unwrap_or(Split {
        dimension: 0,
        bucket: N_BUCKETS / 2 - 1,
    })
}

fn partition_handles(
    mut handles: Vec<u32>,
    light_bounds: &[LightBounds],
    split: Split,
) -> (Vec<u32>, Vec<u32>) {
    let mut centroid_min = f32::INFINITY;
    let mut centroid_max = f32::NEG_INFINITY;
    for &handle in &handles {
        let centroid = light_bounds[handle as usize].bounds.centroid()[split.dimension];
        centroid_min = centroid_min.min(centroid);
        centroid_max = centroid_max.max(centroid);
    }
    let mut left = Vec::new();
    let mut right = Vec::new();
    for handle in handles.drain(..) {
        let centroid = light_bounds[handle as usize].bounds.centroid()[split.dimension];
        if bucket_index(centroid, centroid_min, centroid_max) <= split.bucket {
            left.push(handle);
        } else {
            right.push(handle);
        }
    }
    if left.is_empty() || right.is_empty() {
        let mut sorted = left;
        sorted.extend(right);
        sorted.sort_unstable();
        let mid = sorted.len() / 2;
        (sorted[..mid].to_vec(), sorted[mid..].to_vec())
    } else {
        (left, right)
    }
}

fn bucket_index(value: f32, min: f32, max: f32) -> usize {
    if min == max {
        return 0;
    }
    (((value - min) / (max - min) * N_BUCKETS as f32) as usize).min(N_BUCKETS - 1)
}

fn union_bucket_range(
    buckets: &[Option<LightBounds>; N_BUCKETS],
    start: usize,
    end: usize,
) -> Option<LightBounds> {
    let mut result: Option<LightBounds> = None;
    for bounds in buckets.iter().take(end + 1).skip(start).flatten().copied() {
        result = Some(match result {
            Some(existing) => existing.union(bounds).unwrap(),
            None => bounds,
        });
    }
    result
}

fn evaluate_cost(bounds: LightBounds, parent: super::Bounds3, dimension: usize) -> f32 {
    let diagonal = parent.diagonal();
    if diagonal[dimension] == 0.0 {
        return 0.0;
    }
    let theta_o = bounds.cos_theta_o.acos();
    let theta_e = bounds.cos_theta_e.acos();
    let theta_w = (theta_o + theta_e).min(std::f32::consts::PI);
    let sin_theta_o = safe_sqrt(1.0 - bounds.cos_theta_o * bounds.cos_theta_o);
    let m_omega = 2.0 * std::f32::consts::PI * (1.0 - bounds.cos_theta_o)
        + std::f32::consts::PI / 2.0
            * (2.0 * theta_w * sin_theta_o
                - (theta_o - 2.0 * theta_w).cos()
                - 2.0 * theta_o * sin_theta_o
                + bounds.cos_theta_o);
    let kr = diagonal.into_iter().fold(0.0_f32, f32::max) / diagonal[dimension];
    bounds.phi * m_omega * kr * bounds.bounds.surface_area()
}

fn safe_sqrt(value: f32) -> f32 {
    value.max(0.0).sqrt()
}

fn node_index(index: usize) -> Result<u32, PbrtError> {
    u32::try_from(index).map_err(|_| PbrtError::error("Light BVH node count exceeds u32."))
}
