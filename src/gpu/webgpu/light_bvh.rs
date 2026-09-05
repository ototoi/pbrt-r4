use crate::util::error::PbrtError;

use super::light_sampler::{CompactLightBounds, LIGHT_BVH_INDEX_MAX};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightBvhInput {
    pub handle: u32,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightBvhNode {
    Leaf { handle: u32, parent: u32 },
    Interior { right_child: u32, parent: u32 },
}

impl LightBvhNode {
    pub fn parent(self) -> u32 {
        match self {
            Self::Leaf { parent, .. } | Self::Interior { parent, .. } => parent,
        }
    }

    pub fn is_leaf(self) -> bool {
        matches!(self, Self::Leaf { .. })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LightBvhTopology {
    pub nodes: Vec<LightBvhNode>,
    pub handle_to_leaf: Vec<u32>,
}

impl LightBvhTopology {
    /// Packs one topology entry together with its already-quantized bounds.
    /// The caller supplies bounds in DFS node order; this method only adds the
    /// topology links and therefore cannot silently reorder payload data.
    pub fn pack_node(
        &self,
        node_index: usize,
        bounds: CompactLightBounds,
    ) -> Result<[u32; 8], PbrtError> {
        let node = *self
            .nodes
            .get(node_index)
            .ok_or_else(|| PbrtError::error("Light BVH node index is out of range."))?;
        let (child_or_light_index, is_leaf) = match node {
            LightBvhNode::Leaf { handle, .. } => (handle, true),
            LightBvhNode::Interior { right_child, .. } => (right_child, false),
        };
        let mut words = bounds.to_words(child_or_light_index, is_leaf)?;
        words[7] = node.parent();
        Ok(words)
    }
}

pub fn build_light_bvh(inputs: &[LightBvhInput]) -> Result<LightBvhTopology, PbrtError> {
    if inputs.is_empty() {
        return Ok(LightBvhTopology {
            nodes: Vec::new(),
            handle_to_leaf: Vec::new(),
        });
    }
    let mut items = inputs.to_vec();
    let max_handle = items.iter().map(|item| item.handle).max().unwrap();
    if max_handle > LIGHT_BVH_INDEX_MAX {
        return Err(PbrtError::error(
            "Light BVH handle exceeds the 31-bit limit.",
        ));
    }
    let mut seen_handles = vec![false; max_handle as usize + 1];
    for item in &items {
        if item
            .min
            .iter()
            .chain(item.max.iter())
            .any(|v| !v.is_finite())
            || item
                .min
                .iter()
                .zip(item.max.iter())
                .any(|(min, max)| min > max)
        {
            return Err(PbrtError::error("Light BVH input bounds are invalid."));
        }
        let seen = &mut seen_handles[item.handle as usize];
        if *seen {
            return Err(PbrtError::error("Light BVH light handles must be unique."));
        }
        *seen = true;
    }
    let mut topology = LightBvhTopology {
        nodes: Vec::with_capacity(items.len() * 2 - 1),
        handle_to_leaf: vec![u32::MAX; max_handle as usize + 1],
    };
    let item_count = items.len();
    build_subtree(&mut items, 0, item_count, u32::MAX, &mut topology)?;
    Ok(topology)
}

fn build_subtree(
    items: &mut [LightBvhInput],
    start: usize,
    end: usize,
    parent: u32,
    topology: &mut LightBvhTopology,
) -> Result<u32, PbrtError> {
    let node_index = u32::try_from(topology.nodes.len())
        .map_err(|_| PbrtError::error("Light BVH node count does not fit in u32."))?;
    if node_index > LIGHT_BVH_INDEX_MAX {
        return Err(PbrtError::error(
            "Light BVH node count exceeds the 31-bit limit.",
        ));
    }
    if end - start == 1 {
        let handle = items[start].handle;
        topology.nodes.push(LightBvhNode::Leaf { handle, parent });
        topology.handle_to_leaf[handle as usize] = node_index;
        return Ok(node_index);
    }

    let (centroid_min, centroid_max) = centroid_bounds(items, start, end);
    let split_dimension = largest_extent(centroid_min, centroid_max);
    let middle = if let Some((split_bucket, middle)) = choose_sah_split(
        items,
        start,
        end,
        split_dimension,
        centroid_min,
        centroid_max,
    ) {
        // Partition by the same bucket predicate used for the SAH cost. The
        // centroid tie-break keeps the resulting topology deterministic.
        let extent = centroid_max[split_dimension] - centroid_min[split_dimension];
        items[start..end].sort_by(|a, b| {
            let bucket = |item: &LightBvhInput| {
                ((centroid(item, split_dimension) - centroid_min[split_dimension]) / extent * 12.0)
                    .floor()
                    .min(11.0) as usize
            };
            bucket(a)
                .cmp(&bucket(b))
                .then_with(|| centroid(a, split_dimension).total_cmp(&centroid(b, split_dimension)))
        });
        debug_assert_eq!(
            items[start..end]
                .iter()
                .take_while(|item| {
                    let bucket = ((centroid(item, split_dimension) - centroid_min[split_dimension])
                        / extent
                        * 12.0)
                        .floor()
                        .min(11.0) as usize;
                    bucket <= split_bucket
                })
                .count()
                + start,
            middle
        );
        middle
    } else {
        let middle = (start + end) / 2;
        items[start..end].select_nth_unstable_by(middle - start, |a, b| {
            centroid(a, split_dimension).total_cmp(&centroid(b, split_dimension))
        });
        middle
    };
    topology.nodes.push(LightBvhNode::Interior {
        right_child: u32::MAX,
        parent,
    });
    let left = build_subtree(items, start, middle, node_index, topology)?;
    debug_assert_eq!(left, node_index + 1);
    let right = build_subtree(items, middle, end, node_index, topology)?;
    if let LightBvhNode::Interior { right_child, .. } = &mut topology.nodes[node_index as usize] {
        *right_child = right;
    }
    Ok(node_index)
}

fn choose_sah_split(
    items: &[LightBvhInput],
    start: usize,
    end: usize,
    dimension: usize,
    centroid_min: [f32; 3],
    centroid_max: [f32; 3],
) -> Option<(usize, usize)> {
    let extent = centroid_max[dimension] - centroid_min[dimension];
    if extent == 0.0 {
        return None;
    }
    const BUCKETS: usize = 12;
    let mut counts = [0usize; BUCKETS];
    let mut bounds = [Bounds::empty(); BUCKETS];
    for item in &items[start..end] {
        let bucket = ((centroid(item, dimension) - centroid_min[dimension]) / extent
            * BUCKETS as f32)
            .floor()
            .min((BUCKETS - 1) as f32) as usize;
        counts[bucket] += 1;
        bounds[bucket].union(item);
    }
    let mut best = None;
    let mut best_cost = f32::INFINITY;
    for split in 1..BUCKETS - 1 {
        let mut left = Bounds::empty();
        let mut right = Bounds::empty();
        let mut left_count = 0;
        let mut right_count = 0;
        for bucket in 0..=split {
            left.union_bounds(&bounds[bucket]);
            left_count += counts[bucket];
        }
        for bucket in split + 1..BUCKETS {
            right.union_bounds(&bounds[bucket]);
            right_count += counts[bucket];
        }
        if left_count != 0 && right_count != 0 {
            let cost =
                left.surface_area() * left_count as f32 + right.surface_area() * right_count as f32;
            if cost > 0.0 && cost < best_cost {
                best = Some(split);
                best_cost = cost;
            }
        }
    }
    best.map(|split| {
        let middle = items[start..end]
            .iter()
            .filter(|item| {
                let bucket = ((centroid(item, dimension) - centroid_min[dimension]) / extent
                    * BUCKETS as f32)
                    .floor()
                    .min((BUCKETS - 1) as f32) as usize;
                bucket <= split
            })
            .count()
            + start;
        (split, middle)
    })
}

#[derive(Clone, Copy)]
struct Bounds {
    min: [f32; 3],
    max: [f32; 3],
}

impl Bounds {
    const fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    fn union(&mut self, item: &LightBvhInput) {
        for axis in 0..3 {
            self.min[axis] = self.min[axis].min(item.min[axis]);
            self.max[axis] = self.max[axis].max(item.max[axis]);
        }
    }

    fn union_bounds(&mut self, other: &Self) {
        for axis in 0..3 {
            self.min[axis] = self.min[axis].min(other.min[axis]);
            self.max[axis] = self.max[axis].max(other.max[axis]);
        }
    }

    fn surface_area(self) -> f32 {
        let d = [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ];
        2.0 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2])
    }
}

fn centroid(item: &LightBvhInput, dimension: usize) -> f32 {
    (item.min[dimension] + item.max[dimension]) * 0.5
}

fn centroid_bounds(items: &[LightBvhInput], start: usize, end: usize) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for item in &items[start..end] {
        for axis in 0..3 {
            let value = centroid(item, axis);
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    (min, max)
}

fn largest_extent(min: [f32; 3], max: [f32; 3]) -> usize {
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    if extent[0] > extent[1] && extent[0] > extent[2] {
        0
    } else if extent[1] > extent[2] {
        1
    } else {
        2
    }
}
