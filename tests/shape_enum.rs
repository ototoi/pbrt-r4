//! Simple test to verify Shape works correctly

use pbrt_r4::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn test_shape_enum_sphere() {
    // Create a simple sphere
    let o2w = Transform::identity();
    let w2o = Transform::identity();
    let sphere = Sphere::new(&o2w, &w2o, false, 1.0, -1.0, 1.0, 360.0);

    // Wrap in Shape
    let shape = Shape::Sphere(Box::new(sphere));

    // Test object_bound
    let bounds = shape.object_bound();
    assert!(bounds.min.x < -0.99);
    assert!(bounds.max.x > 0.99);

    // Test world_bound
    let world_bounds = shape.world_bound();
    assert!(world_bounds.min.x < -0.99);

    // Test area
    let area = shape.area();
    assert!(area > 0.0);

    println!("Shape::Sphere test passed!");
}

#[test]
fn test_shape_enum_cylinder() {
    let o2w = Transform::identity();
    let w2o = Transform::identity();
    let cylinder = Cylinder::new(&o2w, &w2o, false, 1.0, -1.0, 1.0, 360.0);
    let shape = Shape::Cylinder(Box::new(cylinder));

    let bounds = shape.object_bound();
    assert!(bounds.min.x < -0.99);
    assert!(bounds.max.x > 0.99);

    let area = shape.area();
    assert!(area > 0.0);
}

#[test]
fn test_shape_enum_disk() {
    let o2w = Transform::identity();
    let w2o = Transform::identity();
    let disk = Disk::new(&o2w, &w2o, false, 0.0, 1.0, 0.5, 360.0);
    let shape = Shape::Disk(Box::new(disk));

    let area = shape.area();
    assert!(area > 0.0);
}

#[test]
fn test_shape_enum_cone() {
    let o2w = Transform::identity();
    let w2o = Transform::identity();
    let cone = Cone::new(&o2w, &w2o, false, 1.0, 1.0, 360.0);
    let shape = Shape::Cone(Box::new(cone));

    let area = shape.area();
    assert!(area > 0.0);
}

#[test]
fn test_shape_enum_ray_intersection() {
    let o2w = Transform::identity();
    let w2o = Transform::identity();
    let sphere = Sphere::new(&o2w, &w2o, false, 1.0, -1.0, 1.0, 360.0);
    let shape = Shape::Sphere(Box::new(sphere));

    // Ray pointing at the sphere from outside
    let ray = Ray::new(
        &Point3f::new(0.0, 0.0, -5.0),
        &Vector3f::new(0.0, 0.0, 1.0),
        Float::INFINITY,
        0.0,
    );

    // Should intersect
    let intersection = shape.intersect(&ray, Float::INFINITY);
    assert!(intersection.is_some());

    if let Some(si) = intersection {
        assert!(si.t_hit > 0.0);
        assert!(si.t_hit < 5.0); // Should hit before reaching origin
    }

    // Ray pointing away should not intersect
    let ray_away = Ray::new(
        &Point3f::new(0.0, 0.0, -5.0),
        &Vector3f::new(0.0, 0.0, -1.0),
        Float::INFINITY,
        0.0,
    );

    let no_intersection = shape.intersect(&ray_away, Float::INFINITY);
    assert!(no_intersection.is_none());

    println!("Shape::Sphere intersection test passed!");
}

#[test]
fn test_shape_create_bilinear_mesh() {
    let mut params = ParameterDictionary::new();
    params.add_point(
        "P",
        &[
            -1.0, 0.0, -1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        ],
    );

    let textures: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let shapes = Shape::create(
        "bilinearmesh",
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
        &textures,
    )
    .unwrap();

    assert_eq!(shapes.len(), 1);
    assert!(matches!(shapes[0].as_ref(), Shape::BilinearPatch(_)));
}

#[test]
fn test_shape_create_bilinear_mesh_intersects_ray() {
    let mut params = ParameterDictionary::new();
    params.add_point(
        "P",
        &[
            -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        ],
    );

    let textures: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let shapes = Shape::create(
        "bilinearmesh",
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
        &textures,
    )
    .unwrap();

    let ray = Ray::new(
        &Point3f::new(0.0, 0.0, -2.0),
        &Vector3f::new(0.0, 0.0, 1.0),
        Float::INFINITY,
        0.0,
    );

    let hit = shapes
        .iter()
        .any(|shape| shape.intersect(&ray, Float::INFINITY).is_some());
    assert!(hit);
}
