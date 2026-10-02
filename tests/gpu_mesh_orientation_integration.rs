use std::path::Path;
use std::process::Command;

use pbrt_r4::util::imageio::read_image::read_image;
use pbrt_r4::util::spectrum::RGBSpectrum;

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn dielectric_mesh_orientation_matches_cpu() {
    let directory = tempfile::tempdir().unwrap();
    // Transmission throughput distinguishes entering from exiting a dielectric.
    // Opposed authored normals must remain opposed, even with mirrored instances.
    for normals in ["absent", "outward", "opposed"] {
        for reverse in [false, true] {
            for mirrored in [false, true] {
                let name = format!("{normals}-reverse-{reverse}-mirror-{mirrored}");
                let scene = directory.path().join(format!("{name}.pbrt"));
                std::fs::write(&scene, dielectric_scene(normals, reverse, mirrored)).unwrap();
                let cpu = render_energy(
                    &scene,
                    &directory.path().join(format!("{name}-cpu.exr")),
                    false,
                );
                let gpu = render_energy(
                    &scene,
                    &directory.path().join(format!("{name}-gpu.exr")),
                    true,
                );
                let error = (gpu - cpu).abs() / cpu;
                eprintln!("{name}: CPU={cpu:.6}, GPU={gpu:.6}, relative error={error:.4}");
                assert!(cpu > 0.0 && gpu > 0.0);
                assert!(error < 0.10, "{name}: CPU={cpu}, GPU={gpu}, error={error}");
            }
        }
    }
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn tangent_only_mesh_orientation_matches_cpu() {
    let directory = tempfile::tempdir().unwrap();
    for reverse in [false, true] {
        for mirrored in [false, true] {
            let name = format!("tangent-only-reverse-{reverse}-mirror-{mirrored}");
            let scene = directory.path().join(format!("{name}.pbrt"));
            let text = dielectric_scene("absent", reverse, mirrored).replace(
                r#""point2 uv""#,
                r#""vector3 S" [ 1 0 0  1 0 0  1 0 0  1 0 0 ] "point2 uv""#,
            );
            std::fs::write(&scene, text).unwrap();
            let cpu = render_energy(
                &scene,
                &directory.path().join(format!("{name}-cpu.exr")),
                false,
            );
            let gpu = render_energy(
                &scene,
                &directory.path().join(format!("{name}-gpu.exr")),
                true,
            );
            assert!(cpu > 0.0 && gpu > 0.0);
            assert!(
                (gpu - cpu).abs() / cpu < 0.10,
                "{name}: CPU={cpu}, GPU={gpu}"
            );
        }
    }
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn oriented_normal_derivatives_match_cpu_with_bump_mapping() {
    let directory = tempfile::tempdir().unwrap();
    for degenerate_uv in [false, true] {
        for reverse in [false, true] {
            let name = format!("bump-degenerate-{degenerate_uv}-reverse-{reverse}");
            let uv = if degenerate_uv {
                "0 0  0 0  0 0  0 0"
            } else {
                "0 0  1 0  1 1  0 1"
            };
            let mut text = dielectric_scene("outward", reverse, false)
                .replace(
                    r#"LightSource "infinite" "rgb L" [ 1 1 1 ]"#,
                    r#"LightSource "point" "point3 from" [ 0 2 4 ] "rgb I" [ 40 40 40 ]"#,
                )
                .replace(
                    r#"Material "dielectric" "float eta" [ 1.5 ]"#,
                    r#"Texture "height" "float" "constant" "float value" [ 0.5 ]
Material "diffuse" "rgb reflectance" [ 0.8 0.7 0.6 ] "texture displacement" [ "height" ]"#,
                )
                .replace(
                    r#""normal N" [ 0 0 1  0 0 1  0 0 1  0 0 1 ]"#,
                    r#""normal N" [ -0.2 -0.1 1  0.3 -0.1 1  0.2 0.3 1  -0.1 0.2 1 ]"#,
                );
            text = text.replace(
                r#""point2 uv" [ 0 0  1 0  1 1  0 1 ]"#,
                &format!(r#""point2 uv" [ {uv} ]"#),
            );
            let scene = directory.path().join(format!("{name}.pbrt"));
            std::fs::write(&scene, text).unwrap();
            let cpu = render_energy(
                &scene,
                &directory.path().join(format!("{name}-cpu.exr")),
                false,
            );
            let gpu = render_energy(
                &scene,
                &directory.path().join(format!("{name}-gpu.exr")),
                true,
            );
            let error = (gpu - cpu).abs() / cpu;
            eprintln!("{name}: CPU={cpu:.6}, GPU={gpu:.6}, relative error={error:.4}");
            assert!(cpu > 0.0 && gpu > 0.0);
            assert!(error < 0.10, "{name}: CPU={cpu}, GPU={gpu}, error={error}");
        }
    }
}

#[test]
#[ignore = "requires a WebGPU adapter with experimental ray-query support"]
fn oriented_area_light_emission_matches_cpu() {
    let directory = tempfile::tempdir().unwrap();
    for reverse in [false, true] {
        for mirrored in [false, true] {
            let name = format!("emitter-reverse-{reverse}-mirror-{mirrored}");
            let text = dielectric_scene("outward", reverse, false)
                .replace(r#"LightSource "infinite" "rgb L" [ 1 1 1 ]"#, "")
                .replace(
                    r#"Material "dielectric" "float eta" [ 1.5 ]"#,
                    r#"Material "diffuse" "rgb reflectance" [ 0.5 0.5 0.5 ]"#,
                )
                // CPU area lights do not support ObjectInstance; transform the shape directly.
                .replace(
                    r#"ObjectBegin "plane""#,
                    &format!(
                        "{}\nAreaLightSource \"diffuse\" \"rgb L\" [ 1 1 1 ]",
                        if mirrored { "Scale -1 1 1" } else { "" },
                    ),
                )
                .replace("ObjectEnd\n", "")
                .replace("ObjectInstance \"plane\"\n", "");
            let scene = directory.path().join(format!("{name}.pbrt"));
            std::fs::write(&scene, text).unwrap();
            let cpu = render_energy(
                &scene,
                &directory.path().join(format!("{name}-cpu.exr")),
                false,
            );
            let gpu = render_energy(
                &scene,
                &directory.path().join(format!("{name}-gpu.exr")),
                true,
            );
            eprintln!("{name}: CPU={cpu:.6}, GPU={gpu:.6}");
            if reverse {
                assert_eq!(cpu, 0.0, "{name}");
                assert_eq!(gpu, 0.0, "{name}");
            } else {
                assert!(cpu > 0.0 && gpu > 0.0);
                assert!(
                    (gpu - cpu).abs() / cpu < 0.03,
                    "{name}: CPU={cpu}, GPU={gpu}"
                );
            }
        }
    }
}

fn dielectric_scene(normals: &str, reverse: bool, mirrored: bool) -> String {
    let normals = match normals {
        "absent" => "",
        "outward" => r#""normal N" [ 0 0 1  0 0 1  0 0 1  0 0 1 ]"#,
        "opposed" => r#""normal N" [ 0 0 -1  0 0 -1  0 0 -1  0 0 -1 ]"#,
        _ => unreachable!(),
    };
    let orientation = if reverse { "ReverseOrientation" } else { "" };
    let transform = if mirrored { "Scale -1 1 1" } else { "" };
    format!(
        r#"Integrator "volpath" "integer maxdepth" [ 2 ]
Sampler "independent" "integer pixelsamples" [ 128 ]
Film "rgb" "integer xresolution" [ 8 ] "integer yresolution" [ 8 ]
LookAt 0 0 4  0 0 0  0 1 0
Camera "perspective" "float fov" [ 15 ]
WorldBegin
LightSource "infinite" "rgb L" [ 1 1 1 ]
Material "dielectric" "float eta" [ 1.5 ]
{orientation}
ObjectBegin "plane"
Shape "trianglemesh"
    "point3 P" [ -10 -10 0  10 -10 0  10 10 0  -10 10 0 ]
    "point2 uv" [ 0 0  1 0  1 1  0 1 ]
    "integer indices" [ 0 1 2  0 2 3 ]
    {normals}
ObjectEnd
{transform}
ObjectInstance "plane"
WorldEnd
"#
    )
}

fn render_energy(scene: &Path, output: &Path, gpu: bool) -> f32 {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"));
    if gpu {
        command.arg("--use-gpu");
    }
    let result = command
        .arg("--nthreads")
        .arg("2")
        .arg("--outfile")
        .arg(output)
        .arg(scene)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "render failed: {}\n{}",
        scene.display(),
        String::from_utf8_lossy(&result.stderr)
    );
    let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
    assert_eq!([resolution.x, resolution.y], [8, 8]);
    assert!(pixels.iter().all(RGBSpectrum::is_valid));
    pixels
        .iter()
        .map(|p| p.to_rgb().into_iter().sum::<f32>())
        .sum::<f32>()
        / (pixels.len() * 3) as f32
}
