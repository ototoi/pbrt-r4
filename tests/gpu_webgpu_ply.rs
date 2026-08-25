#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::ir::{GpuRenderConfig, GpuRenderRequest};
use pbrt_r4::gpu::webgpu::{PrepareOptions, Renderer};
use pbrt_r4::parser::{parse_string, SceneBuilder};

#[test]
fn hardware_wavefront_renders_a_plymesh_from_the_cpu_loader() {
    let file = tempfile::Builder::new().suffix(".ply").tempfile().unwrap();
    std::fs::write(
        file.path(),
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n-0.5 -0.5 0\n0.5 -0.5 0\n0 0.5 0\n3 0 1 2\n",
    )
    .unwrap();

    let mut builder = SceneBuilder::new();
    parse_string(
        &format!(
            r#"
                Film "rgb" "integer xresolution" [1] "integer yresolution" [1]
                Sampler "independent" "integer pixelsamples" [1]
                PixelFilter "box"
                Integrator "volpath" "integer maxdepth" [1]
                LookAt 0 0 2 0 0 0 0 1 0
                Camera "perspective" "float fov" [45]
                WorldBegin
                Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
                LightSource "point" "point3 from" [0 0 2] "rgb I" [10 10 10]
                Shape "plymesh" "string filename" ["{}"]
                WorldEnd
            "#,
            file.path().display()
        ),
        &mut builder,
    )
    .unwrap();

    let scene = builder.build_gpu_ir().unwrap();
    let mut renderer = Renderer::new(&PrepareOptions::default()).unwrap_or_else(|error| {
        panic!("Hardware Ray Query is required for this WebGPU test: {error}")
    });
    let executable = renderer.prepare(&scene).unwrap();
    let output = renderer
        .render(
            &executable,
            &GpuRenderRequest::new(&GpuRenderConfig::default(), 0, 1).unwrap(),
        )
        .unwrap();

    assert_eq!(output.rgb.len(), 1);
}
