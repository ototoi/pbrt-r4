#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::webgpu::wavefront::{
    validate_wavefront_abi, BxDFWorkItem, DirectLightingContribution, IntersectionRecord,
    QueueHeader, ShadowWorkItem, SurfaceInteractionRecord, WavefrontControl, BXDF_WORK_ITEM_STRIDE,
    DIRECT_LIGHTING_CONTRIBUTION_STRIDE, INTERSECTION_RECORD_STRIDE, QUEUE_HEADER_STRIDE,
    SHADOW_WORK_ITEM_STRIDE, SURFACE_INTERACTION_RECORD_STRIDE, WAVEFRONT_ARENA_HEADER_SIZE,
    WAVEFRONT_CONTROL_SIZE, WAVEFRONT_QUEUE_HEADER_COUNT,
};

#[test]
fn wavefront_records_match_the_declared_storage_strides() {
    assert!(validate_wavefront_abi());
    assert_eq!(
        std::mem::size_of::<WavefrontControl>(),
        WAVEFRONT_CONTROL_SIZE as usize
    );
    assert_eq!(
        std::mem::size_of::<QueueHeader>(),
        QUEUE_HEADER_STRIDE as usize
    );
    assert_eq!(WAVEFRONT_QUEUE_HEADER_COUNT, 6);
    assert_eq!(
        WAVEFRONT_ARENA_HEADER_SIZE,
        WAVEFRONT_CONTROL_SIZE + WAVEFRONT_QUEUE_HEADER_COUNT * QUEUE_HEADER_STRIDE
    );
    assert_eq!(
        std::mem::size_of::<IntersectionRecord>(),
        INTERSECTION_RECORD_STRIDE as usize
    );
    assert_eq!(
        std::mem::size_of::<SurfaceInteractionRecord>(),
        SURFACE_INTERACTION_RECORD_STRIDE as usize
    );
    assert_eq!(
        std::mem::size_of::<BxDFWorkItem>(),
        BXDF_WORK_ITEM_STRIDE as usize
    );
    assert_eq!(
        std::mem::size_of::<DirectLightingContribution>(),
        DIRECT_LIGHTING_CONTRIBUTION_STRIDE as usize
    );
    assert_eq!(
        std::mem::size_of::<ShadowWorkItem>(),
        SHADOW_WORK_ITEM_STRIDE as usize
    );
}
