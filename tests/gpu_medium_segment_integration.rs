use std::process::Command;

use pbrt_r4::util::imageio::read_image::read_image;

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn another_pixels_medium_boundary_does_not_resample_completed_segments() {
    let directory = tempfile::tempdir().unwrap();
    let render = |boundary: bool| {
        let scene = directory.path().join(if boundary {
            "boundary.pbrt"
        } else {
            "plain.pbrt"
        });
        let output = directory.path().join(if boundary {
            "boundary.exr"
        } else {
            "plain.exr"
        });
        let mut input = String::from(
            r#"MakeNamedMedium "fog" "string type" "homogeneous"
    "rgb sigma_a" [0.1 0.2 0.4] "rgb sigma_s" [0 0 0]
MediumInterface "fog" "fog"
LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" [35]
Film "rgb" "integer xresolution" [8] "integer yresolution" [8]
Sampler "independent" "integer pixelsamples" [1]
Integrator "volpath" "integer maxdepth" [1]
WorldBegin
AttributeBegin
    AreaLightSource "diffuse" "rgb L" [1 1 1]
    Material "diffuse" "rgb reflectance" [0 0 0]
    Shape "trianglemesh"
        "point3 P" [-5 -5 0  5 -5 0  5 5 0  -5 5 0]
        "integer indices" [0 1 2  0 2 3]
AttributeEnd
"#,
        );
        if boundary {
            input.push_str(
                r#"AttributeBegin
    Material ""
    MediumInterface "fog" "fog"
    Shape "trianglemesh"
        "point3 P" [-3 -3 2  0 -3 2  0 3 2  -3 3 2]
        "integer indices" [0 1 2  0 2 3]
AttributeEnd
"#,
            );
        }
        std::fs::write(&scene, input).unwrap();
        let status = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
            .args([
                "--use-gpu",
                "--quiet",
                "--outfile",
                output.to_str().unwrap(),
                scene.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(status.success());
        let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
        assert_eq!([resolution.x, resolution.y], [8, 8]);
        pixels
    };

    let plain = render(false);
    let boundary = render(true);
    let mut changed_energy = 0.0;
    for y in 0..8 {
        for x in 0..8 {
            let index = y * 8 + x;
            for (plain, boundary) in plain[index]
                .to_rgb()
                .into_iter()
                .zip(boundary[index].to_rgb())
            {
                let difference = (plain - boundary).abs();
                if x < 4 {
                    assert!(
                        difference < 1e-6,
                        "unrelated ray at ({x}, {y}) changed by {difference}"
                    );
                } else {
                    changed_energy += difference;
                }
            }
        }
    }
    assert!(
        changed_energy > 0.01,
        "the boundary did not exercise segment continuation"
    );
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn transparent_boundary_preserves_direct_light_visibility() {
    let directory = tempfile::tempdir().unwrap();
    let render = |boundary: bool| {
        let scene = directory.path().join(if boundary {
            "shadow-boundary.pbrt"
        } else {
            "shadow-plain.pbrt"
        });
        let output = directory.path().join(if boundary {
            "shadow-boundary.exr"
        } else {
            "shadow-plain.exr"
        });
        let mut input = String::from(
            r#"LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" [35]
Film "rgb" "integer xresolution" [8] "integer yresolution" [8]
Sampler "independent" "integer pixelsamples" [1]
Integrator "volpath" "integer maxdepth" [1]
WorldBegin
LightSource "point" "point3 from" [3 0 2] "rgb I" [20 20 20]
Material "diffuse" "rgb reflectance" [0.8 0.8 0.8]
Shape "trianglemesh"
    "point3 P" [-5 -5 0  5 -5 0  5 5 0  -5 5 0]
    "integer indices" [0 1 2  0 2 3]
"#,
        );
        if boundary {
            input.push_str(
                r#"AttributeBegin
    Material ""
    Shape "trianglemesh"
        "point3 P" [1 -3 -1  1 3 -1  1 3 3  1 -3 3]
        "integer indices" [0 1 2  0 2 3]
AttributeEnd
"#,
            );
        }
        std::fs::write(&scene, input).unwrap();
        let status = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
            .args([
                "--use-gpu",
                "--quiet",
                "--outfile",
                output.to_str().unwrap(),
                scene.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(status.success());
        read_image(output.to_str().unwrap()).unwrap().0
    };

    let plain = render(false);
    let boundary = render(true);
    let mut energy = 0.0;
    for (plain, boundary) in plain.iter().zip(&boundary) {
        for (plain, boundary) in plain.to_rgb().into_iter().zip(boundary.to_rgb()) {
            energy += plain;
            assert!((plain - boundary).abs() < 1e-5);
        }
    }
    assert!(energy > 0.0);
}
