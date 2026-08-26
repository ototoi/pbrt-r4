use pbrt_r4::base::shape::ShapeSampleContext;
use pbrt_r4::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

fn rectangular_patch() -> Shape {
    let mut params = ParameterDictionary::new();
    params.add_point(
        "P",
        &[
            -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        ],
    );

    let textures: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    Shape::create(
        "bilinearmesh",
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
        &textures,
    )
    .unwrap()
    .pop()
    .unwrap()
}

#[test]
fn rectangular_patch_uses_solid_angle_sampling() {
    let patch = rectangular_patch();
    let context = ShapeSampleContext {
        p: Point3f::new(0.0, 0.0, 2.0),
        ..Default::default()
    };

    // Solid angle of an axis-aligned rectangle with half extents a and b,
    // viewed at perpendicular distance d.
    let a = 1.0;
    let b = 1.0;
    let d = 2.0;
    let solid_angle = 4.0 * Float::atan2(a * b, d * Float::sqrt(d * d + a * a + b * b));
    let expected_pdf = 1.0 / solid_angle;

    for u in [Point2f::new(0.2, 0.3), Point2f::new(0.8, 0.7)] {
        let (sample, sample_pdf) = patch.sample_from(&context, &u).unwrap();
        assert!((sample_pdf - expected_pdf).abs() < 1e-5);

        let wi = (sample.get_p() - context.p).normalize();
        let evaluated_pdf = patch.pdf_from(&context, &wi);
        assert!((evaluated_pdf - expected_pdf).abs() < 1e-5);
        assert!((evaluated_pdf - sample_pdf).abs() < 1e-5);
    }
}

#[test]
fn rectangular_patch_cosine_warp_sample_and_pdf_agree() {
    let patch = rectangular_patch();
    let normal = Normal3f::new(1.0, 0.0, 1.0).normalize();
    let context = ShapeSampleContext {
        p: Point3f::new(0.7, 0.0, 2.0),
        n: normal,
        ns: normal,
        time: 0.5,
    };

    for u in [Point2f::new(0.17, 0.31), Point2f::new(0.73, 0.89)] {
        let (sample, sample_pdf) = patch.sample_from(&context, &u).unwrap();
        let wi = (sample.get_p() - context.p).normalize();
        let evaluated_pdf = patch.pdf_from(&context, &wi);
        assert!((sample.get_time() - context.time).abs() < 1e-6);
        assert!(sample_pdf.is_finite() && sample_pdf > 0.0);
        assert!(
            (evaluated_pdf - sample_pdf).abs() < 2e-4,
            "sample_pdf={sample_pdf} evaluated_pdf={evaluated_pdf}"
        );
    }
}
