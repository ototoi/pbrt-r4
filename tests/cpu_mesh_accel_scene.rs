#![cfg(feature = "mesh-accel")]

use pbrt_r4::parser::{parse_string, SceneBuilder};

#[test]
fn emissive_triangle_mesh_keeps_one_area_light_per_triangle() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"WorldBegin
AreaLightSource "diffuse" "rgb L" [ 1 1 1 ]
Shape "trianglemesh"
    "point3 P" [ 0 0 0  1 0 0  1 1 0  0 1 0 ]
    "integer indices" [ 0 1 2  0 2 3 ]
WorldEnd"#,
        &mut builder,
    )
    .expect("scene should parse");

    let scene = builder.build_scene().expect("scene should build");
    assert_eq!(scene.lights.len(), 2);
}

#[test]
fn three_mesh_instances_are_hit_after_transform() {
    let mut builder = SceneBuilder::new();
    parse_string(
        r#"LookAt 0 0 5  0 0 0  0 1 0
WorldBegin
ObjectBegin "quad"
Shape "trianglemesh"
    "point3 P" [ -0.5 -0.5 0  0.5 -0.5 0  0.5 0.5 0  -0.5 0.5 0 ]
    "integer indices" [ 0 1 2  0 2 3 ]
ObjectEnd
AttributeBegin
Translate -1 0 0
ObjectInstance "quad"
AttributeEnd
AttributeBegin
Translate 0 0 0
ObjectInstance "quad"
AttributeEnd
AttributeBegin
Translate 1 0 0
ObjectInstance "quad"
AttributeEnd
WorldEnd"#,
        &mut builder,
    )
    .expect("scene should parse");

    let scene = builder
        .build_scene()
        .expect("instanced mesh scene should build");
    for x in [-1.0, 0.0, 1.0] {
        let ray = pbrt_r4::util::geometry::ray::Ray::new(
            &pbrt_r4::util::base::Point3f::new(x, 0.0, 5.0),
            &pbrt_r4::util::base::Vector3f::new(0.0, 0.0, -1.0),
            10.0,
            0.0,
        );
        assert!(
            scene.intersect(&ray, f32::INFINITY).is_some(),
            "instance at x={x} was missed"
        );
    }
}
