use crate::gpu::ir::flat;
use crate::util::error::PbrtError;

pub const LIGHT_BVH_NODE_WORDS: usize = 8;
pub const LIGHT_BVH_NODE_BYTES: usize = LIGHT_BVH_NODE_WORDS * std::mem::size_of::<u32>();
pub const LIGHT_BVH_INDEX_MAX: u32 = 0x7fff_ffff;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LightSamplerKind {
    Uniform,
    Bvh,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactLightBounds {
    pub q_min: [u16; 3],
    pub q_max: [u16; 3],
    pub direction: [u16; 2],
    pub phi: f32,
    pub cos_theta_o: u16,
    pub cos_theta_e: u16,
    pub two_sided: bool,
}

impl CompactLightBounds {
    pub fn pack(
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        all_bounds_min: [f32; 3],
        all_bounds_max: [f32; 3],
        direction: [f32; 3],
        phi: f32,
        cos_theta_o: f32,
        cos_theta_e: f32,
        two_sided: bool,
    ) -> Result<Self, PbrtError> {
        if !bounds_min
            .iter()
            .chain(bounds_max.iter())
            .chain(all_bounds_min.iter())
            .chain(all_bounds_max.iter())
            .chain(direction.iter())
            .copied()
            .chain([phi, cos_theta_o, cos_theta_e])
            .all(f32::is_finite)
        {
            return Err(PbrtError::error(
                "Light BVH bounds contain a non-finite value.",
            ));
        }
        if bounds_min
            .into_iter()
            .zip(bounds_max)
            .any(|(min, max)| min > max)
            || all_bounds_min
                .into_iter()
                .zip(all_bounds_max)
                .any(|(min, max)| min > max)
            || phi < 0.0
            || !(-1.0..=1.0).contains(&cos_theta_o)
            || !(-1.0..=1.0).contains(&cos_theta_e)
        {
            return Err(PbrtError::error(
                "Light BVH bounds contain an invalid range.",
            ));
        }
        let direction_length = direction.into_iter().map(|v| v * v).sum::<f32>().sqrt();
        if direction_length == 0.0 {
            return Err(PbrtError::error(
                "Light BVH bounds direction must be non-zero.",
            ));
        }
        let mut q_min = [0; 3];
        let mut q_max = [0; 3];
        for axis in 0..3 {
            q_min[axis] = quantize_bounds(
                bounds_min[axis],
                all_bounds_min[axis],
                all_bounds_max[axis],
                false,
            )?;
            q_max[axis] = quantize_bounds(
                bounds_max[axis],
                all_bounds_min[axis],
                all_bounds_max[axis],
                true,
            )?;
        }
        let direction = direction.map(|value| value / direction_length);
        Ok(Self {
            q_min,
            q_max,
            direction: encode_octahedral(direction),
            phi,
            cos_theta_o: quantize_cosine(cos_theta_o),
            cos_theta_e: quantize_cosine(cos_theta_e),
            two_sided,
        })
    }

    pub fn to_words(self, child_or_light_index: u32, is_leaf: bool) -> Result<[u32; 8], PbrtError> {
        if child_or_light_index > LIGHT_BVH_INDEX_MAX {
            return Err(PbrtError::error(
                "Light BVH child or light index exceeds the 31-bit limit.",
            ));
        }
        Ok([
            u32::from(self.q_min[0]) | (u32::from(self.q_min[1]) << 16),
            u32::from(self.q_min[2]) | (u32::from(self.q_max[0]) << 16),
            u32::from(self.q_max[1]) | (u32::from(self.q_max[2]) << 16),
            u32::from(self.direction[0]) | (u32::from(self.direction[1]) << 16),
            self.phi.to_bits(),
            u32::from(self.cos_theta_o)
                | (u32::from(self.cos_theta_e) << 15)
                | (u32::from(self.two_sided) << 30),
            child_or_light_index | (u32::from(is_leaf) << 31),
            u32::MAX,
        ])
    }
}

fn quantize_bounds(value: f32, min: f32, max: f32, ceil: bool) -> Result<u16, PbrtError> {
    if min == max {
        return Ok(0);
    }
    let normalized = ((value - min) / (max - min)).clamp(0.0, 1.0) * 65535.0;
    let quantized = if ceil {
        normalized.ceil()
    } else {
        normalized.floor()
    };
    u16::try_from(quantized as u32)
        .map_err(|_| PbrtError::error("Light BVH bounds quantization overflowed."))
}

fn quantize_cosine(value: f32) -> u16 {
    (32767.0 * ((value + 1.0) / 2.0)).floor() as u16
}

fn encode_octahedral(mut direction: [f32; 3]) -> [u16; 2] {
    let sum = direction.into_iter().map(f32::abs).sum::<f32>();
    direction = direction.map(|value| value / sum);
    if direction[2] < 0.0 {
        direction = [
            (1.0 - direction[1].abs()) * octahedral_sign(direction[0]),
            (1.0 - direction[0].abs()) * octahedral_sign(direction[1]),
            direction[2],
        ];
    }
    [
        ((direction[0] + 1.0) / 2.0 * 65535.0)
            .clamp(0.0, 65535.0)
            .round() as u16,
        ((direction[1] + 1.0) / 2.0 * 65535.0)
            .clamp(0.0, 65535.0)
            .round() as u16,
    ]
}

fn octahedral_sign(value: f32) -> f32 {
    if value.is_sign_negative() {
        -1.0
    } else {
        1.0
    }
}

pub fn resolve_light_sampler(
    requested: &str,
    registered_light_count: usize,
) -> Result<LightSamplerKind, PbrtError> {
    if registered_light_count == 1 {
        return Ok(LightSamplerKind::Uniform);
    }

    match requested {
        "uniform" => Ok(LightSamplerKind::Uniform),
        "bvh" => Ok(LightSamplerKind::Bvh),
        "power" | "exhaustive" => Err(PbrtError::error(&format!(
            "WebGPU light sampler \"{requested}\" is not implemented."
        ))),
        unknown => {
            log::error!("Unknown WebGPU light sampler \"{unknown}\"; using bvh.");
            Ok(LightSamplerKind::Bvh)
        }
    }
}

pub fn resolve_scene_light_sampler(
    settings: &flat::RenderSettings,
    registered_lights: &[flat::LightRecord],
) -> Result<LightSamplerKind, PbrtError> {
    resolve_light_sampler(&settings.light_sampler, registered_lights.len())
}
