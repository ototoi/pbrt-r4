use pbrt_r4::gpu::webgpu::queue::{packed_wavefront_layout, QUEUE_STATE_WORDS};

#[test]
fn packed_wavefront_regions_follow_the_shader_layout() {
    let layout = packed_wavefront_layout(3, 2).unwrap();

    assert_eq!(QUEUE_STATE_WORDS, 24);
    assert_eq!(layout.sample_state_offset_words, 24);
    assert_eq!(layout.ray_data_offset_words, 48);
    assert_eq!(layout.shadow_data_offset_words, 168);
    assert_eq!(layout.material_data_offset_words, 228);
    assert_eq!(layout.hit_area_data_offset_words, 237);
    assert_eq!(layout.escaped_data_offset_words, 246);
    assert_eq!(layout.total_words, 255);
    assert_eq!(layout.state_readback_size_bytes(), 96);
    assert_eq!(layout.wavefront_size_bytes().unwrap(), 1020);
}

#[test]
fn packed_wavefront_layout_must_fit_shader_u32_word_offsets() {
    assert!(packed_wavefront_layout(u64::from(u32::MAX), 1).is_err());
}
