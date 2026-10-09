pub const WORKGROUP_SIZE: u32 = 8;
pub const RAY_T_MIN: f32 = 0.0;
pub const RAY_T_MAX: f32 = f32::MAX;
pub const LIGHT_KIND_POINT: u32 = 0;
pub const LIGHT_KIND_AREA: u32 = 1;
pub const LIGHT_KIND_SPOT: u32 = 2;
pub const LIGHT_KIND_DISTANT: u32 = 3;
pub const LIGHT_KIND_UNIFORM_INFINITE: u32 = 4;
pub const LIGHT_KIND_IMAGE_INFINITE: u32 = 5;
pub const LIGHT_KIND_PORTAL_IMAGE_INFINITE: u32 = 6;
pub const LIGHT_SAMPLER_KIND_UNIFORM: u32 = 0;
pub const LIGHT_SAMPLER_KIND_BVH: u32 = 1;
pub const INVALID_INDEX: u32 = u32::MAX;
pub const INTERSECTION_NORMAL_KIND_TRIANGLE: u32 = 0;
pub const INTERSECTION_NORMAL_KIND_QUADRIC: u32 = 1;

pub const INSTANCE_ORIENTATION_FLAG_REVERSED: u32 = 1 << 0;
pub const INSTANCE_ORIENTATION_FLAG_TRANSFORM_SWAPS_HANDEDNESS: u32 = 1 << 1;
pub const INSTANCE_ORIENTATION_FLAG_SHAPE_TRANSFORM_SWAPS_HANDEDNESS: u32 = 1 << 2;

pub fn instance_orientation_flags(
    reverse_orientation: bool,
    transform_swaps_handedness: bool,
) -> u32 {
    (if reverse_orientation {
        INSTANCE_ORIENTATION_FLAG_REVERSED
    } else {
        0
    }) | (if transform_swaps_handedness {
        INSTANCE_ORIENTATION_FLAG_TRANSFORM_SWAPS_HANDEDNESS
    } else {
        0
    })
}
pub const TEXTURE_OPERATION_IMAGE: u32 = 0;
pub const TEXTURE_OPERATION_CONSTANT: u32 = 1;
pub const TEXTURE_OPERATION_SCALE: u32 = 2;
pub const TEXTURE_OPERATION_MIX: u32 = 3;
pub const TEXTURE_OPERATION_CHECKERBOARD: u32 = 4;
pub const TEXTURE_OPERATION_DIRECTION_MIX: u32 = 5;
pub const TEXTURE_OPERATION_DOTS: u32 = 6;
pub const TEXTURE_OPERATION_FBM: u32 = 7;
pub const TEXTURE_OPERATION_WRINKLED: u32 = 8;
pub const TEXTURE_OPERATION_WINDY: u32 = 9;
pub const TEXTURE_OPERATION_BILERP: u32 = 10;
pub const TEXTURE_OPERATION_MARBLE: u32 = 11;
