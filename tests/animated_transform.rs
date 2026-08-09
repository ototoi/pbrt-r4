// Imported from animatedtransform.cpp
use pbrt_r4::prelude::*;

fn random_transform(rng: &mut RNG) -> Transform {
    let mut t = Transform::identity();
    let r = |rng: &mut RNG| -10.0 + 20.0 * rng.uniform_float();
    for _ in 0..10 {
        match rng.uniform_uint32_threshold(3) {
            0 => {
                t = t * Transform::scale(r(rng).abs(), r(rng).abs(), r(rng).abs());
            }
            1 => {
                t = t * Transform::translate(r(rng), r(rng), r(rng));
            }
            2 => {
                let theta = r(rng) * 20.0;
                let uv = Point2f::new(rng.uniform_float(), rng.uniform_float());
                let axis = uniform_sample_sphere(&uv);
                t = t * Transform::rotate(theta, axis.x, axis.y, axis.z);
            }
            _ => unreachable!(),
        }
    }
    return t;
}

fn random_rotation(rng: &mut RNG) -> (Float, Vector3f) {
    let theta = 20.0 * rng.uniform_float();
    let uv = Point2f::new(rng.uniform_float(), rng.uniform_float());
    let axis = uniform_sample_sphere(&uv);
    return (theta, axis);
}

#[test]
fn animated_transform_convert() {
    let mut rng = RNG::new();

    for _ in 0..200 {
        let (theta1, axis1) = random_rotation(&mut rng);
        let q1 = Quaternion::from_angle_axis(theta1, &axis1).normalize();
        let m1 = q1.to_matrix();
        let q1p = Quaternion::from_matrix(&m1);
        assert!(
            q1.dot(&q1p) > 0.99999,
            "{:?} {:?} - {} {} - {} {}",
            q1,
            q1p,
            Vector3::new(q1.x, q1.y, q1.z).length(),
            Vector3::new(q1p.x, q1p.y, q1p.z).length(),
            (1.0 - q1.w * q1.w).sqrt(),
            (1.0 - q1p.w * q1p.w).sqrt()
        );
    }
}

#[test]
fn animated_transform_randoms() {
    let mut rng = RNG::new();
    let r = |rng: &mut RNG| -10.0 + 20.0 * rng.uniform_float();

    for _ in 0..200 {
        // Generate a pair of random transformation matrices.
        let t0 = random_transform(&mut rng);
        let t1 = random_transform(&mut rng);
        let at = AnimatedTransform::new(&t0, 0.0, &t1, 1.0).unwrap();

        for _ in 0..5 {
            // Generate a pair of random transformation matrices.
            let bounds = Bounds3f::from((
                (r(&mut rng), r(&mut rng), r(&mut rng)),
                (r(&mut rng), r(&mut rng), r(&mut rng)),
            ));
            assert!(bounds.diagonal().x >= 0.0);
            assert!(bounds.diagonal().y >= 0.0);
            assert!(bounds.diagonal().z >= 0.0);

            let motion_bounds = at.motion_bounds(&bounds);

            let mut t = 0.0;
            while t <= 1.0 {
                // Now, interpolate the transformations at a bunch of times
                // along the time range and then transform the bounding box
                // with the result.
                assert!(t >= 0.0 && t <= 1.0);
                let tr = at.interpolate(t);
                let mut tb = tr.transform_bounds(&bounds);

                // Add a little slop to allow for floating-point round-off
                // error in computing the motion extrema times.
                let diagonal = tb.diagonal();
                tb.min += 1e-4 * diagonal;
                tb.max -= 1e-4 * diagonal;

                // Now, the transformed bounds should be inside the motion
                // bounds.
                assert!(tb.min.x >= motion_bounds.min.x);
                assert!(tb.min.y >= motion_bounds.min.y);
                assert!(tb.min.z >= motion_bounds.min.z);

                assert!(tb.max.x <= motion_bounds.max.x);
                assert!(tb.max.y <= motion_bounds.max.y);
                assert!(tb.max.z <= motion_bounds.max.z);

                t += 1e-3 * rng.uniform_float();
            }
        }
    }
}
