use std::process::Command;

use pbrt_r4::util::base::Float;
use pbrt_r4::util::imageio::read_image::read_image;
use pbrt_r4::util::spectrum::RGBSpectrum;

#[path = "support/nanovdb_fixture.rs"]
mod nanovdb_fixture;

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn another_pixels_crossing_multiple_medium_boundaries_does_not_resample_completed_segments() {
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
AttributeBegin
    Material ""
    MediumInterface "fog" "fog"
    Shape "trianglemesh"
        "point3 P" [-3 -3 3  0 -3 3  0 3 3  -3 3 3]
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
fn nanovdb_camera_medium_attenuates_primary_rays() {
    let directory = tempfile::tempdir().unwrap();
    let nvdb_path = directory.path().join("camera-medium.nvdb");
    std::fs::write(
        &nvdb_path,
        nanovdb_fixture::file_bytes(
            &[nanovdb_fixture::root_tile("density", 2.0, ([0; 3], [7; 3]))],
            false,
        ),
    )
    .unwrap();

    let render = |camera_medium: bool| {
        let scene_path = directory.path().join(if camera_medium {
            "nanovdb-camera-medium.pbrt"
        } else {
            "clear-camera.pbrt"
        });
        let output_path = directory.path().join(if camera_medium {
            "nanovdb-camera-medium.exr"
        } else {
            "clear-camera.exr"
        });
        let medium_setup = if camera_medium {
            format!(
                r#"MakeNamedMedium "fog" "string type" "nanovdb"
    "string filename" "{}" "rgb sigma_a" [0.25 0.25 0.25] "rgb sigma_s" [0 0 0]
MediumInterface "" "fog"
"#,
                nvdb_path.display()
            )
        } else {
            String::new()
        };
        let scene = format!(
            r#"{medium_setup}LookAt 3 3 6  3 3 0  0 1 0
Camera "perspective" "float fov" [12]
Film "rgb" "integer xresolution" [8] "integer yresolution" [8]
Sampler "independent" "integer pixelsamples" [128]
Integrator "volpath" "integer maxdepth" [2]
WorldBegin
AreaLightSource "diffuse" "rgb L" [10 10 10]
Material "diffuse" "rgb reflectance" [0.8 0.8 0.8]
Shape "trianglemesh"
    "point3 P" [0 0 0  7 0 0  7 7 0  0 7 0]
    "integer indices" [0 1 2  0 2 3]
"#
        );
        std::fs::write(&scene_path, scene).unwrap();
        let status = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
            .args([
                "--use-gpu",
                "--quiet",
                "--outfile",
                output_path.to_str().unwrap(),
                scene_path.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(status.success());
        let (pixels, resolution) = read_image(output_path.to_str().unwrap()).unwrap();
        assert_eq!([resolution.x, resolution.y], [8, 8]);
        pixels
    };

    let clear = render(false);
    let attenuated = render(true);
    let mean_luminance = |pixels: &[RGBSpectrum]| {
        pixels
            .iter()
            .map(|pixel| {
                let [r, g, b] = pixel.to_rgb();
                0.2126 * r + 0.7152 * g + 0.0722 * b
            })
            .sum::<Float>()
            / pixels.len() as Float
    };
    let clear_mean = mean_luminance(&clear);
    let attenuated_mean = mean_luminance(&attenuated);
    assert!(clear_mean.is_finite() && attenuated_mean.is_finite());
    assert!(clear_mean > 0.0);
    assert!(attenuated_mean > 0.0 && attenuated_mean < clear_mean * 0.25);
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn mirrored_medium_boundary_uses_v4_geometric_side() {
    let directory = tempfile::tempdir().unwrap();
    let render = |boundary: bool| {
        let scene = directory.path().join(if boundary {
            "mirrored-boundary.pbrt"
        } else {
            "mirrored-plain.pbrt"
        });
        let output = directory.path().join(if boundary {
            "mirrored-boundary.exr"
        } else {
            "mirrored-plain.exr"
        });
        let mut input = String::from(
            r#"MakeNamedMedium "fog" "string type" "homogeneous"
    "rgb sigma_a" [0.2 0.3 0.4] "rgb sigma_s" [0 0 0]
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
    Scale -1 1 1
    Material ""
    MediumInterface "fog" ""
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
                    changed_energy += difference;
                }
            }
        }
    }
    assert!(
        changed_energy > 0.01,
        "the mirrored boundary did not select its inside medium"
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

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn shadow_transmittance_applies_homogeneous_absorption() {
    let directory = tempfile::tempdir().unwrap();
    let render = |with_medium: bool| {
        let scene = directory.path().join(if with_medium {
            "shadow-medium.pbrt"
        } else {
            "shadow-vacuum.pbrt"
        });
        let output = directory.path().join(if with_medium {
            "shadow-medium.exr"
        } else {
            "shadow-vacuum.exr"
        });
        let mut input = String::from(
            r#"MakeNamedMedium "fog" "string type" "homogeneous"
    "rgb sigma_a" [0.3 0.3 0.3] "rgb sigma_s" [0 0 0]
LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" [35]
Film "rgb" "integer xresolution" [8] "integer yresolution" [8]
Sampler "independent" "integer pixelsamples" [128]
Integrator "volpath" "integer maxdepth" [1]
WorldBegin
LightSource "point" "point3 from" [3 0 2] "rgb I" [20 20 20]
"#,
        );
        if with_medium {
            input.push_str("MediumInterface \"\" \"fog\"\n");
        }
        input.push_str(
            r#"Material "diffuse" "rgb reflectance" [0.8 0.8 0.8]
Shape "trianglemesh"
    "point3 P" [-5 -5 0  5 -5 0  5 5 0  -5 5 0]
    "integer indices" [0 1 2  0 2 3]
"#,
        );
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
            .iter()
            .map(|pixel| pixel.to_rgb().into_iter().sum::<f32>())
            .sum::<f32>()
    };

    let vacuum_energy = render(false);
    let medium_energy = render(true);
    assert!(vacuum_energy > 0.0);
    assert!(
        medium_energy < vacuum_energy * 0.8,
        "medium shadow transmittance was not applied: vacuum={vacuum_energy}, medium={medium_energy}"
    );
    assert!(
        medium_energy > vacuum_energy * 0.1,
        "medium shadow transmittance removed too much energy: vacuum={vacuum_energy}, medium={medium_energy}"
    );
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn primary_and_shadow_medium_samples_are_independent() {
    let directory = tempfile::tempdir().unwrap();
    let render = |with_medium: bool| {
        let scene = directory.path().join(if with_medium {
            "coupled-medium.pbrt"
        } else {
            "coupled-vacuum.pbrt"
        });
        let output = directory.path().join(if with_medium {
            "coupled-medium.exr"
        } else {
            "coupled-vacuum.exr"
        });
        let mut input = String::from(
            r#"MakeNamedMedium "fog" "string type" "homogeneous"
    "rgb sigma_a" [0.5 0.5 0.5] "rgb sigma_s" [0 0 0]
LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" [35]
Film "rgb" "integer xresolution" [8] "integer yresolution" [8]
Sampler "independent" "integer pixelsamples" [128]
Integrator "volpath" "integer maxdepth" [1]
WorldBegin
LightSource "point" "point3 from" [0 0 3] "rgb I" [20 20 20]
AttributeBegin
    Material ""
"#,
        );
        input.push_str(if with_medium {
            "    MediumInterface \"fog\" \"\"\n"
        } else {
            "    MediumInterface \"\" \"\"\n"
        });
        input.push_str(
            r#"    Shape "trianglemesh"
        "point3 P" [-5 -5 4  5 -5 4  5 5 4  -5 5 4]
        "integer indices" [0 1 2  0 2 3]
AttributeEnd
AttributeBegin
    Material "diffuse" "rgb reflectance" [0.8 0.8 0.8]
"#,
        );
        input.push_str(if with_medium {
            "    MediumInterface \"\" \"fog\"\n"
        } else {
            "    MediumInterface \"\" \"\"\n"
        });
        input.push_str(
            r#"    Shape "trianglemesh"
        "point3 P" [-5 -5 2  5 -5 2  5 5 2  -5 5 2]
        "integer indices" [0 1 2  0 2 3]
AttributeEnd
"#,
        );
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
            .iter()
            .map(|pixel| pixel.to_rgb().into_iter().sum::<f32>())
            .sum::<f32>()
    };

    let vacuum_energy = render(false);
    let medium_energy = render(true);
    let ratio = medium_energy / vacuum_energy;
    assert!(
        (0.16..0.29).contains(&ratio),
        "two independent 0.5-unit absorption segments should retain about exp(-1.5) energy; observed ratio={ratio}"
    );
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn homogeneous_scattering_lights_a_bounded_medium_and_matches_cpu() {
    let directory = tempfile::tempdir().unwrap();
    let render = |scattering: bool, gpu: bool| {
        let scene = directory
            .path()
            .join(format!("medium-{scattering}-{gpu}.pbrt"));
        let output = scene.with_extension("exr");
        let input = r#"MakeNamedMedium "fog" "string type" "homogeneous"
    "rgb sigma_a" [0.1 0.1 0.1] "rgb sigma_s" [SIGMA_S SIGMA_S SIGMA_S]
    "float g" [0.4]
MediumInterface "" "fog"
LookAt 0 0 0  0 0 1  0 1 0
Camera "perspective" "float fov" [40]
Film "rgb" "integer xresolution" [8] "integer yresolution" [8]
Sampler "independent" "integer pixelsamples" [512]
Integrator "volpath" "integer maxdepth" [4]
WorldBegin
AttributeBegin
    MediumInterface "" ""
    Translate 0 0 3
    LightSource "point" "rgb I" [10 10 10]
AttributeEnd
Material ""
MediumInterface "fog" ""
Shape "sphere" "float radius" [1]
AttributeBegin
    MediumInterface "" ""
    Translate 1000 0 0
    Material "diffuse" "rgb reflectance" [0 0 0]
    Shape "sphere" "float radius" [0.1]
AttributeEnd
"#
        .replace("SIGMA_S", if scattering { "1" } else { "0" });
        std::fs::write(&scene, input).unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"));
        command
            .arg("--quiet")
            .arg("--outfile")
            .arg(&output)
            .arg(&scene);
        if gpu {
            command.arg("--use-gpu");
        }
        assert!(command.status().unwrap().success());
        let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
        assert_eq!([resolution.x, resolution.y], [8, 8]);
        assert!(pixels
            .iter()
            .all(|pixel| pixel.to_rgb().iter().all(|v| v.is_finite() && *v >= 0.0)));
        pixels.iter().map(|pixel| pixel.y() as f64).sum::<f64>() / pixels.len() as f64
    };
    assert_eq!(render(false, true), 0.0);
    let cpu = render(true, false);
    let gpu = render(true, true);
    assert!(
        cpu > 0.0 && gpu > 0.0,
        "scattering must receive point-light radiance: CPU={cpu}, GPU={gpu}"
    );
    assert!((gpu / cpu - 1.0).abs() < 0.15, "CPU={cpu}, GPU={gpu}");
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn homogeneous_emission_matches_absorbing_segment_solution() {
    let directory = tempfile::tempdir().unwrap();
    let render = |gpu: bool, sigma_a: f32, max_depth: u32, spp: u32| {
        let scene = directory
            .path()
            .join(format!("emission-{gpu}-{sigma_a}-{max_depth}-{spp}.pbrt"));
        let output = scene.with_extension("exr");
        std::fs::write(
            &scene,
            format!(
                r#"MakeNamedMedium "emitter" "string type" "homogeneous"
    "rgb sigma_a" [{sigma_a} {sigma_a} {sigma_a}] "rgb sigma_s" [0 0 0]
    "rgb Le" [1 1 1]
MediumInterface "" "emitter"
LookAt 0 0 0  0 0 1  0 1 0
Camera "perspective" "float fov" [40]
Film "rgb" "integer xresolution" [1] "integer yresolution" [1]
Sampler "independent" "integer pixelsamples" [{spp}]
Integrator "volpath" "integer maxdepth" [{max_depth}]
WorldBegin
Material ""
MediumInterface "emitter" ""
Shape "sphere" "float radius" [1]
"#
            ),
        )
        .unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"));
        command
            .arg("--quiet")
            .arg("--outfile")
            .arg(&output)
            .arg(&scene);
        if gpu {
            command.arg("--use-gpu");
        }
        assert!(command.status().unwrap().success());
        let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
        assert_eq!([resolution.x, resolution.y], [1, 1]);
        let luminance = pixels[0].y();
        assert!(luminance.is_finite() && luminance >= 0.0);
        luminance
    };

    let expected = 1.0 - (-0.1_f32).exp();
    let cpu = render(false, 0.1, 1, 16_384);
    let gpu = render(true, 0.1, 1, 16_384);
    assert!(
        (cpu - expected).abs() < 0.01,
        "CPU={cpu}, analytical absorbing-segment radiance={expected}"
    );
    assert!(
        (gpu - expected).abs() < 0.01,
        "GPU={gpu}, analytical absorbing-segment radiance={expected}"
    );
    assert_eq!(render(true, 0.0, 1, 8), 0.0);
    assert_eq!(render(true, 0.1, 0, 8), 0.0);
}
