use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::prelude::*;
use pbrt_r4::shapes::{get_alpha_texture, get_shadow_alpha_texture};
use pbrt_r4::textures::FloatTexture;
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn missing_alpha_texture_is_an_error() {
    let params = ParameterDictionary::new();
    let textures: HashMap<String, Arc<FloatTexture>> = HashMap::new();

    assert!(get_alpha_texture(&params, &textures).unwrap().is_none());
    assert!(get_shadow_alpha_texture(&params, &textures)
        .unwrap()
        .is_none());
}

#[test]
fn scalar_alpha_value_is_preserved() {
    let mut params = ParameterDictionary::new();
    params.add_float("alpha", 0.0);
    let textures: HashMap<String, Arc<FloatTexture>> = HashMap::new();

    let alpha = get_alpha_texture(&params, &textures)
        .unwrap()
        .expect("scalar alpha should be preserved");
    match alpha {
        pbrt_r4::shapes::AlphaMaskInfo::Value { value } => assert_eq!(value, 0.0),
        _ => panic!("expected scalar alpha value"),
    }
}

#[test]
fn fractional_alpha_stochastically_masks_single_hit_triangles() {
    let identity = Transform::identity();
    let mut params = ParameterDictionary::new();
    params.add_ints("indices", &[0, 1, 2]);
    params.add_point("P", &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    params.add_float("alpha", 0.5);

    let shapes = Shape::create(
        "trianglemesh",
        &identity,
        &identity,
        false,
        &params,
        &HashMap::new(),
    )
    .expect("triangle mesh");
    assert_eq!(shapes.len(), 1);

    let mut accepted = 0;
    let mut rejected = 0;
    for i in 0..128 {
        let u = (i % 16) as Float / 32.0 + 0.01;
        let v = (i / 16) as Float / 32.0 + 0.01;
        let ray = Ray::new(
            &Point3f::new(u, v, -1.0),
            &Vector3f::new(0.0, 0.0, 1.0),
            Float::INFINITY,
            0.0,
        );
        if shapes[0].intersect(&ray, Float::INFINITY).is_some() {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }

    assert!(accepted > 0);
    assert!(rejected > 0);
}

#[test]
fn fractional_alpha_retries_the_next_sphere_intersection() {
    let identity = Transform::identity();
    let mut params = ParameterDictionary::new();
    params.add_float("radius", 1.0);
    params.add_float("alpha", 0.5);

    let shapes = Shape::create(
        "sphere",
        &identity,
        &identity,
        false,
        &params,
        &HashMap::new(),
    )
    .expect("sphere shape");

    let mut back_intersections = 0;
    for i in 0..128 {
        let x = (i % 16) as Float / 20.0 - 0.375;
        let y = (i / 16) as Float / 20.0 - 0.375;
        let ray = Ray::new(
            &Point3f::new(x, y, -3.0),
            &Vector3f::new(0.0, 0.0, 1.0),
            Float::INFINITY,
            0.0,
        );
        if let Some(isect) = shapes[0].intersect(&ray, Float::INFINITY) {
            if isect.t_hit > 3.0 {
                back_intersections += 1;
            }
        }
    }

    assert!(back_intersections > 0);
}
