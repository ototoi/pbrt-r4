#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::ir::{GpuRenderConfig, GpuRenderRequest};
use pbrt_r4::gpu::webgpu::{PrepareOptions, Renderer};
use pbrt_r4::parser::{parse_string, SceneBuilder};

#[test]
fn wavefront_renders_direct_diffuse_lighting_with_hit_and_miss_pixels() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"
            Film "rgb" "integer xresolution" [4] "integer yresolution" [2]
            PixelFilter "box"
            Sampler "independent" "integer pixelsamples" [1]
            Integrator "volpath" "integer maxdepth" [1]
            LookAt 0 0 2 0 0 0 0 1 0
            Camera "perspective" "float fov" [45]
            WorldBegin
            Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
            AttributeBegin
                Translate 0 0 2
                LightSource "point" "rgb I" [10 10 10]
            AttributeEnd
            Shape "trianglemesh"
                "point3 P" [-1 -1 0 1 -1 0 0 1 0]
                "integer indices" [0 1 2]
            WorldEnd
        "#,
        &mut builder,
    )
    .unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    let mut renderer = Renderer::new(&PrepareOptions::default()).unwrap_or_else(|error| {
        panic!("Hardware Ray Query is required for this WebGPU test: {error}")
    });
    let executable = renderer.prepare(&compiled).unwrap();
    let request = GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap();
    let output = renderer.render(&executable, &request).unwrap();

    assert_eq!(output.rgb.len(), 8);
    assert!(output
        .rgb
        .iter()
        .flatten()
        .all(|component| component.is_finite() && (0.0..=1.0).contains(component)));
    assert!(output.rgb.iter().any(|pixel| *pixel == [0.0; 3]));
    assert!(output.rgb.iter().any(|pixel| *pixel != [0.0; 3]));
}

#[test]
fn wavefront_shadow_ray_rejects_occluded_direct_lighting() {
    let visible = render_center_pixel("");
    let occluded = render_center_pixel(
        r#"
            Shape "trianglemesh"
                "point3 P" [0.9 -10 -10 0.9 10 -10 0.9 0 10]
                "integer indices" [0 1 2]
        "#,
    );

    let visible_luminance: f32 = visible.iter().sum();
    let occluded_luminance: f32 = occluded.iter().sum();
    assert!(visible_luminance > 0.01, "unoccluded pixel: {visible:?}");
    assert!(
        occluded_luminance < visible_luminance * 0.01,
        "occluded pixel {occluded:?} should be dark relative to {visible:?}",
    );
}

#[test]
fn wavefront_uniformly_samples_multiple_point_lights() {
    let one_light = render_center_pixel("");
    let two_identical_lights = render_center_pixel(
        r#"
            AttributeBegin
                Translate 1 0 2
                LightSource "point" "rgb I" [10 10 10]
            AttributeEnd
        "#,
    );

    for (one, two) in one_light.into_iter().zip(two_identical_lights) {
        assert!(
            (two - 2.0 * one).abs() < 1.0e-5,
            "two identical point lights should double direct lighting: {one}, {two}",
        );
    }
}

#[test]
fn wavefront_accumulates_multiple_camera_samples() {
    let pixel = render_center_pixel_with_sample_count("", 4);
    assert!(pixel.into_iter().all(f32::is_finite));
    assert!(pixel.into_iter().any(|component| component > 0.01));
}

fn render_center_pixel(extra_geometry_or_lights: &str) -> [f32; 3] {
    render_center_pixel_with_sample_count(extra_geometry_or_lights, 1)
}

fn render_center_pixel_with_sample_count(
    extra_geometry_or_lights: &str,
    sample_count: u32,
) -> [f32; 3] {
    let mut builder = SceneBuilder::new();
    let scene = format!(
        r#"
            Film "rgb" "integer xresolution" [1] "integer yresolution" [1]
            PixelFilter "box"
            Sampler "independent" "integer pixelsamples" [{sample_count}]
            Integrator "volpath" "integer maxdepth" [1]
            LookAt 0 0 2 0 0 0 0 1 0
            Camera "perspective" "float fov" [45]
            WorldBegin
            Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
            AttributeBegin
                Translate 1 0 2
                LightSource "point" "rgb I" [10 10 10]
            AttributeEnd
            Shape "trianglemesh"
                "point3 P" [-2 -2 0 2 -2 0 0 2 0]
                "integer indices" [0 1 2]
            {extra_geometry_or_lights}
            WorldEnd
        "#,
    );
    parse_string(&scene, &mut builder).unwrap();

    let compiled = builder.build_gpu_ir().unwrap();
    let mut renderer = Renderer::new(&PrepareOptions::default()).unwrap_or_else(|error| {
        panic!("Hardware Ray Query is required for this WebGPU test: {error}")
    });
    let executable = renderer.prepare(&compiled).unwrap();
    let mut render_config = GpuRenderConfig::default();
    render_config.sampler.samples_per_pixel = sample_count;
    let request = GpuRenderRequest::new(&render_config, 0, sample_count).unwrap();
    renderer.render(&executable, &request).unwrap().rgb[0]
}
