use pbrt_r4::gpu::ir::flat;
use pbrt_r4::gpu::webgpu::render_settings::RenderSettings;

#[test]
fn render_settings_keep_only_webgpu_render_loop_values() {
    let settings = RenderSettings::from_flat(flat::RenderSettings {
        samples_per_pixel: 8,
        max_depth: 5,
        seed: 17,
        light_sampler: "bvh".to_string(),
        disable_wavelength_jitter: true,
    });

    assert_eq!(
        settings,
        RenderSettings {
            samples_per_pixel: 8,
            max_depth: 5,
        }
    );
}
