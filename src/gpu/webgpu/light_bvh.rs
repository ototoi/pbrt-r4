use crate::gpu::ir::flat;
use crate::util::error::PbrtError;

pub const LIGHT_BVH_NODE_WORDS: usize = 8;
pub const LIGHT_BVH_HEADER_WORDS: usize = 8;
pub const LIGHT_BVH_INDEX_MAX: u32 = 0x7fff_ffff;

#[derive(Clone, Debug, PartialEq)]
pub struct PackedLightBVH {
    pub header_words: [u32; LIGHT_BVH_HEADER_WORDS],
    pub node_words: Vec<[u32; LIGHT_BVH_NODE_WORDS]>,
    pub handle_to_leaf: Vec<u32>,
}

pub fn pack_light_bvh(bvh: &flat::LightBVH) -> Result<Option<PackedLightBVH>, PbrtError> {
    let Some(all_bounds) = bvh.all_bounds else {
        if !bvh.nodes.is_empty() || !bvh.bounded_handles.is_empty() {
            return Err(PbrtError::error(
                "Non-empty Light BVH must have global bounds.",
            ));
        }
        return Ok(None);
    };
    if bvh.nodes.is_empty() || bvh.bounded_handles.is_empty() {
        return Err(PbrtError::error(
            "Light BVH global bounds require non-empty nodes and handles.",
        ));
    }
    if bvh.nodes.len() > LIGHT_BVH_INDEX_MAX as usize {
        return Err(PbrtError::error(
            "Light BVH node count exceeds the 31-bit packed index limit.",
        ));
    }
    all_bounds_check(all_bounds)?;
    let header_words = [
        all_bounds.min[0].to_bits(),
        all_bounds.min[1].to_bits(),
        all_bounds.min[2].to_bits(),
        0,
        all_bounds.max[0].to_bits(),
        all_bounds.max[1].to_bits(),
        all_bounds.max[2].to_bits(),
        0,
    ];
    let mut node_words = Vec::with_capacity(bvh.nodes.len());
    for (index, node) in bvh.nodes.iter().enumerate() {
        let bounds = node.bounds();
        let compact = CompactLightBounds::pack(bounds, all_bounds)?;
        let (payload, is_leaf) = match node {
            flat::LightBVHNode::Leaf { light_handle, .. } => (*light_handle, true),
            flat::LightBVHNode::Interior { right_child, .. } => (*right_child, false),
        };
        if payload > LIGHT_BVH_INDEX_MAX {
            return Err(PbrtError::error(
                "Light BVH payload exceeds the 31-bit packed index limit.",
            ));
        }
        let parent = node.parent();
        if parent != flat::INVALID_INDEX && parent as usize >= bvh.nodes.len() {
            return Err(PbrtError::error(&format!(
                "Light BVH node {index} has an invalid parent."
            )));
        }
        if let flat::LightBVHNode::Interior { left_child, .. } = node {
            if *left_child != index as u32 + 1 {
                return Err(PbrtError::error(
                    "Light BVH nodes do not satisfy the DFS left-child invariant.",
                ));
            }
        }
        node_words.push(compact.to_words(payload, is_leaf, parent));
    }
    Ok(Some(PackedLightBVH {
        header_words,
        node_words,
        handle_to_leaf: bvh.handle_to_leaf.clone(),
    }))
}

#[derive(Clone, Copy)]
struct CompactLightBounds {
    q_min: [u16; 3],
    q_max: [u16; 3],
    direction: [u16; 2],
    phi: f32,
    cos_theta_o: u16,
    cos_theta_e: u16,
    two_sided: bool,
}

impl CompactLightBounds {
    fn pack(bounds: flat::LightBounds, all: flat::Bounds3) -> Result<Self, PbrtError> {
        bounds.validate()?;
        all_bounds_check(all)?;
        let mut q_min = [0; 3];
        let mut q_max = [0; 3];
        for axis in 0..3 {
            q_min[axis] =
                quantize_bounds(bounds.bounds.min[axis], all.min[axis], all.max[axis], false)?;
            q_max[axis] =
                quantize_bounds(bounds.bounds.max[axis], all.min[axis], all.max[axis], true)?;
        }
        let direction = normalize(bounds.direction)?;
        Ok(Self {
            q_min,
            q_max,
            direction: encode_octahedral(direction),
            phi: bounds.phi,
            cos_theta_o: quantize_cosine(bounds.cos_theta_o),
            cos_theta_e: quantize_cosine(bounds.cos_theta_e),
            two_sided: bounds.two_sided,
        })
    }

    fn to_words(self, payload: u32, is_leaf: bool, parent: u32) -> [u32; 8] {
        [
            u32::from(self.q_min[0]) | (u32::from(self.q_min[1]) << 16),
            u32::from(self.q_min[2]) | (u32::from(self.q_max[0]) << 16),
            u32::from(self.q_max[1]) | (u32::from(self.q_max[2]) << 16),
            u32::from(self.direction[0]) | (u32::from(self.direction[1]) << 16),
            self.phi.to_bits(),
            u32::from(self.cos_theta_o)
                | (u32::from(self.cos_theta_e) << 15)
                | (u32::from(self.two_sided) << 30),
            payload | (u32::from(is_leaf) << 31),
            parent,
        ]
    }
}

fn all_bounds_check(bounds: flat::Bounds3) -> Result<(), PbrtError> {
    if !bounds
        .min
        .into_iter()
        .chain(bounds.max)
        .all(|value| value.is_finite())
        || bounds
            .min
            .into_iter()
            .zip(bounds.max)
            .any(|(min, max)| min > max)
    {
        return Err(PbrtError::error("Global Light BVH bounds are invalid."));
    }
    Ok(())
}

fn quantize_bounds(value: f32, min: f32, max: f32, ceil: bool) -> Result<u16, PbrtError> {
    if min == max {
        return Ok(0);
    }
    let normalized = ((value - min) / (max - min)).clamp(0.0, 1.0) * 65535.0;
    let value = if ceil {
        normalized.ceil()
    } else {
        normalized.floor()
    };
    u16::try_from(value as u32)
        .map_err(|_| PbrtError::error("Light BVH bounds quantization overflowed."))
}

fn quantize_cosine(value: f32) -> u16 {
    (32767.0 * ((value + 1.0) / 2.0)).floor() as u16
}

fn normalize(direction: [f32; 3]) -> Result<[f32; 3], PbrtError> {
    let length = direction
        .into_iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if !length.is_finite() || length == 0.0 {
        return Err(PbrtError::error(
            "Light BVH direction must be finite and non-zero.",
        ));
    }
    Ok(direction.map(|value| value / length))
}

fn encode_octahedral(mut direction: [f32; 3]) -> [u16; 2] {
    let sum = direction.into_iter().map(f32::abs).sum::<f32>();
    direction = direction.map(|value| value / sum);
    if direction[2] < 0.0 {
        direction = [
            (1.0 - direction[1].abs()) * sign(direction[0]),
            (1.0 - direction[0].abs()) * sign(direction[1]),
            direction[2],
        ];
    }
    [
        (((direction[0] + 1.0) * 0.5 * 65535.0).round() as u32).min(u16::MAX as u32) as u16,
        (((direction[1] + 1.0) * 0.5 * 65535.0).round() as u32).min(u16::MAX as u32) as u16,
    ]
}

fn sign(value: f32) -> f32 {
    if value.is_sign_negative() {
        -1.0
    } else {
        1.0
    }
}
