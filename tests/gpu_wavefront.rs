#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::webgpu::wavefront::{
    WavefrontLayout, WAVEFRONT_CONTROL_SIZE, WAVEFRONT_SLOT_STRIDE,
};

#[test]
fn fixed_slot_layout_is_checked_and_contiguous() {
    let layout = WavefrontLayout::for_pixel_count(3).unwrap();

    assert_eq!(layout.capacity, 3);
    assert_eq!(
        layout.byte_len,
        WAVEFRONT_CONTROL_SIZE + 3 * WAVEFRONT_SLOT_STRIDE
    );
    assert_eq!(layout.slot_offset(0), Some(WAVEFRONT_CONTROL_SIZE));
    assert_eq!(
        layout.slot_offset(2),
        Some(WAVEFRONT_CONTROL_SIZE + 2 * WAVEFRONT_SLOT_STRIDE)
    );
    assert_eq!(layout.slot_offset(3), None);
    assert!(WavefrontLayout::for_pixel_count(u32::MAX).is_none());
}
