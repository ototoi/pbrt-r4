#![cfg(feature = "webgpu")]

use std::process::Command;

use image::GenericImageView;

#[test]
fn gpu_cli_renders_multiple_samples_through_cpu_film() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let scene_path = directory.path().join("scene.pbrt");
    let output_path = directory.path().join("image.png");
    std::fs::write(
        &scene_path,
        r#"
            Film "rgb" "integer xresolution" [32] "integer yresolution" [32]
            PixelFilter "box"
            Sampler "independent" "integer pixelsamples" [4]
            Integrator "volpath" "integer maxdepth" [1]
            LookAt 0 0 3 0 0 0 0 1 0
            Camera "perspective" "float fov" [45]
            WorldBegin
            Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
            AttributeBegin
                Translate 1 0 2
                LightSource "point" "rgb I" [1 1 1]
            AttributeEnd
            Shape "trianglemesh"
                "point3 P" [-1 -1 0 1 -1 0 0 1 0]
                "integer indices" [0 1 2]
            WorldEnd
        "#,
    )
    .expect("write scene");

    let status = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
        .args([
            "--gpu",
            "--infile",
            scene_path.to_str().unwrap(),
            "--outfile",
            output_path.to_str().unwrap(),
            "--quiet",
        ])
        .status()
        .expect("run pbrt-r4");
    assert!(status.success());

    let image = image::open(&output_path).expect("GPU output image");
    assert_eq!(image.dimensions(), (32, 32));
}
