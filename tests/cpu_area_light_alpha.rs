use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;

use pbrt_r4::base::light::Light;
use pbrt_r4::base::shape::{Shape, ShapeSampleContext};
use pbrt_r4::media::MediumInterface;
use pbrt_r4::paramdict::{ParameterDictionary, TextureParameterDictionary};
use pbrt_r4::prelude::*;
use pbrt_r4::util::imageio::read_image::read_image;
use pbrt_r4::util::spectrum::SampledWavelengths;
use pbrt_r4::util::transform::Transform;

#[test]
fn fractional_alpha_area_light_emits_for_approximately_half_its_samples() {
    let directory = tempfile::tempdir().unwrap();
    let scene = directory.path().join("area-light-alpha.pbrt");
    let render = |with_alpha: bool, texture_alpha: bool, uv_texture: bool| {
        let scene_text = format!(
            r#"LookAt 0 0 5  0 0 0  0 1 0
Camera "perspective" "float fov" [25]
Film "rgb" "integer xresolution" [8] "integer yresolution" [8]
Sampler "independent" "integer pixelsamples" [512] "integer seed" [17]
Integrator "path" "integer maxdepth" [1]
WorldBegin
{alpha_texture}
AttributeBegin
    AreaLightSource "diffuse" "rgb L" [20 20 20]
    Shape "trianglemesh"
        "point3 P" [1 -1 2  2 1 2  3 -1 2]
        "integer indices" [0 1 2]
{uv_attribute}{alpha_attribute}
AttributeEnd
Material "diffuse" "rgb reflectance" [0.8 0.8 0.8]
Shape "trianglemesh"
    "point3 P" [-3 -3 0  3 -3 0  0 3 0]
    "integer indices" [0 1 2]
"#,
            alpha_texture = if uv_texture {
                "Texture \"mask\" \"float\" \"checkerboard\" \"integer dimension\" [2] \"string mapping\" \"uv\" \"string aamode\" \"none\" \"float tex1\" [1] \"float tex2\" [0]"
            } else if texture_alpha {
                "Texture \"mask\" \"float\" \"constant\" \"float value\" [0.5]"
            } else {
                ""
            },
            alpha_attribute = if texture_alpha || uv_texture {
                "        \"texture alpha\" [\"mask\"]"
            } else if with_alpha {
                "        \"float alpha\" [0.5]"
            } else {
                ""
            },
            uv_attribute = if uv_texture {
                "        \"point2 uv\" [0 0 4 0 0 4]\n"
            } else {
                ""
            },
        );
        std::fs::write(&scene, scene_text).unwrap();
        let output = directory.path().join(if uv_texture {
            "uv-texture-alpha.exr"
        } else if texture_alpha {
            "texture-alpha.exr"
        } else if with_alpha {
            "scalar-alpha.exr"
        } else {
            "opaque.exr"
        });
        let status = Command::new(env!("CARGO_BIN_EXE_pbrt-r4"))
            .args([
                "--outfile",
                output.to_str().unwrap(),
                scene.to_str().unwrap(),
            ])
            .status()
            .unwrap();
        assert!(
            status.success(),
            "CPU area-light render failed: alpha={with_alpha}, texture={texture_alpha}"
        );
        let (pixels, resolution) = read_image(output.to_str().unwrap()).unwrap();
        assert_eq!([resolution.x, resolution.y], [8, 8]);
        pixels
            .iter()
            .map(|pixel| pixel.to_rgb().into_iter().sum::<f32>())
            .sum::<f32>()
            / pixels.len() as f32
    };

    let opaque = render(false, false, false);
    let scalar_alpha = render(true, false, false);
    let texture_alpha = render(true, true, false);
    let uv_texture_alpha = render(false, false, true);
    assert!(
        opaque > 0.1,
        "opaque reference did not sample the light: {opaque}"
    );
    assert!(
        (0.4..0.6).contains(&(scalar_alpha / opaque)),
        "scalar alpha 0.5 should accept about half of the area-light samples: fractional={scalar_alpha}, opaque={opaque}"
    );
    assert!(
        (0.4..0.6).contains(&(texture_alpha / opaque)),
        "texture alpha 0.5 should accept about half of the area-light samples: fractional={texture_alpha}, opaque={opaque}"
    );
    assert!(
        (0.4..0.6).contains(&(uv_texture_alpha / opaque)),
        "UV checkerboard alpha should accept about half of the area-light samples: fractional={uv_texture_alpha}, opaque={opaque}"
    );
}

#[test]
fn fractional_alpha_area_light_emission_sampling_accepts_half_its_points() {
    let opaque = create_triangle_area_light(None);
    let masked = create_triangle_area_light(Some(0.5));
    let lambda = SampledWavelengths::sample_visible(0.5);
    let samples = 1024;
    let mut opaque_samples = 0;
    let mut masked_samples = 0;

    for index in 0..samples {
        let u = Point2f::new(
            (index as Float + 0.5) / samples as Float,
            ((index * 619) % samples) as Float / samples as Float,
        );
        let sample = opaque.sample_le(u, Point2f::new(0.5, 0.5), &lambda, 0.0);
        assert!(
            sample.is_some(),
            "opaque area-light emission sample was rejected"
        );
        opaque_samples += 1;

        if masked
            .sample_le(u, Point2f::new(0.5, 0.5), &lambda, 0.0)
            .is_some()
        {
            masked_samples += 1;
        }
    }

    let coverage = masked_samples as Float / opaque_samples as Float;
    assert!(
        (0.4..0.6).contains(&coverage),
        "alpha 0.5 should accept about half of light-emission samples: accepted={masked_samples}, total={opaque_samples}"
    );
}

#[test]
fn fractional_alpha_area_light_radiance_is_zero_for_masked_points() {
    let light = create_triangle_area_light(Some(0.5));
    let Light::DiffuseArea(area_light) = light.as_ref() else {
        panic!("expected diffuse area light");
    };
    let lambda = SampledWavelengths::sample_visible(0.5);
    let normal = Normal3f::new(0.0, 0.0, -1.0);
    let direction = Vector3f::new(0.0, 0.0, -1.0);
    let samples = 1024;
    let mut emitting = 0;

    for index in 0..samples {
        let x = 1.5 + (index as Float + 0.5) / samples as Float;
        let position = Point3f::new(x, -0.5, 2.0);
        let radiance = area_light.l(position, normal, Point2f::zero(), direction, &lambda);
        if !radiance.is_black() {
            emitting += 1;
        }
    }

    let coverage = emitting as Float / samples as Float;
    assert!(
        (0.4..0.6).contains(&coverage),
        "alpha 0.5 should emit at about half of sampled points: emitting={emitting}, total={samples}"
    );
}

fn create_triangle_area_light(alpha: Option<Float>) -> Arc<Light> {
    let identity = Transform::identity();
    let mut shape_params = ParameterDictionary::new();
    shape_params.add_ints("indices", &[0, 1, 2]);
    shape_params.add_point("P", &[1.0, -1.0, 2.0, 2.0, 1.0, 2.0, 3.0, -1.0, 2.0]);
    if let Some(alpha) = alpha {
        shape_params.add_float("alpha", alpha);
    }
    let shape = Shape::create(
        "trianglemesh",
        &identity,
        &identity,
        false,
        &shape_params,
        &HashMap::new(),
    )
    .expect("triangle area-light shape")
    .remove(0);
    Light::create_area(
        "diffuse",
        &identity,
        &MediumInterface::default(),
        &ParameterDictionary::new(),
        &Arc::new(shape),
    )
    .expect("diffuse area light")
}

#[test]
fn sampled_area_light_shapes_preserve_uv_for_alpha_textures() {
    let identity = Transform::identity();
    let reference = ShapeSampleContext {
        p: Point3f::new(3.0, 0.0, 3.0),
        ..Default::default()
    };
    let u = Point2f::new(0.25, 0.75);

    for kind in ["disk", "cylinder"] {
        let shape = Shape::create(
            kind,
            &identity,
            &identity,
            false,
            &ParameterDictionary::new(),
            &HashMap::new(),
        )
        .unwrap()
        .remove(0);
        for sampled in [shape.sample(&u), shape.sample_from(&reference, &u)] {
            let (intr, _) = sampled.expect("sampled area-light point");
            let uv = intr.get_uv();
            assert_ne!(uv, Point2f::zero(), "{kind} sample lost its UV");
        }
    }

    for kind in ["cone", "paraboloid", "hyperboloid", "curve"] {
        let mut params = ParameterDictionary::new();
        if kind == "hyperboloid" {
            params.add_point("p1", &[0.5, 0.0, 0.0]);
            params.add_point("p2", &[1.0, 0.0, 1.0]);
        } else if kind == "curve" {
            params.add_point(
                "P",
                &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0],
            );
        }
        let shape = Shape::create(kind, &identity, &identity, false, &params, &HashMap::new())
            .unwrap()
            .remove(0);
        let (intr, _) = shape.sample(&u).expect("sampled area-light point");
        assert_ne!(intr.get_uv(), Point2f::zero(), "{kind} sample lost its UV");
    }

    let sphere = Shape::create(
        "sphere",
        &identity,
        &identity,
        false,
        &ParameterDictionary::new(),
        &HashMap::new(),
    )
    .unwrap()
    .remove(0);
    let (intr, _) = sphere
        .sample_from(&reference, &u)
        .expect("sphere sampled from an outside reference point");
    assert_ne!(intr.get_uv(), Point2f::zero(), "sphere sample lost its UV");
}

#[test]
fn disk_and_cylinder_sample_alpha_at_their_own_uv() {
    let identity = Transform::identity();
    let mut texture_params = ParameterDictionary::new();
    texture_params.add_float("uscale", 4.0);
    texture_params.add_float("vscale", 4.0);
    texture_params.add_string("aamode", "none");
    let float_textures = HashMap::new();
    let spectrum_textures = HashMap::new();
    let texture_params =
        TextureParameterDictionary::new(&texture_params, &float_textures, &spectrum_textures);
    let mask = FloatTexture::create("checkerboard", &identity, &texture_params).unwrap();
    let mut float_textures = HashMap::new();
    float_textures.insert("mask".to_string(), Arc::new(mask));

    let mut shape_params = ParameterDictionary::new();
    shape_params.add_string("alpha", "mask");
    let reference = ShapeSampleContext {
        p: Point3f::new(3.0, 0.0, 3.0),
        ..Default::default()
    };
    for kind in ["disk", "cylinder"] {
        let shape = Shape::create(
            kind,
            &identity,
            &identity,
            false,
            &shape_params,
            &float_textures,
        )
        .unwrap()
        .remove(0);
        for from_reference in [false, true] {
            let mut masked = 0;
            let mut visible = 0;
            for i in 0..128 {
                let u = Point2f::new(
                    (i as Float + 0.5) / 128.0,
                    ((i * 37) % 128) as Float / 128.0,
                );
                let sampled = if from_reference {
                    shape.sample_from(&reference, &u)
                } else {
                    shape.sample(&u)
                };
                let Some((intr, _)) = sampled else { continue };
                match shape.alpha(&intr) {
                    Some(0.0) => masked += 1,
                    Some(1.0) => visible += 1,
                    other => panic!("unexpected {kind} alpha: {other:?}"),
                }
            }
            assert!(
                masked > 0 && visible > 0,
                "{kind} samples all used one alpha texel"
            );
        }
    }
}
