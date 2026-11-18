use pbrt_r4::base::shape::Shape;
use pbrt_r4::lights::DiffuseAreaLight;
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::prelude::*;
use pbrt_r4::util::imageio::read_image_with_metadata;
use pbrt_r4::util::spectrum::SampledWavelengths;
use pbrt_r4::util::transform::Transform;
use std::collections::HashMap;
use std::sync::Arc;

fn shape() -> Arc<Shape> {
    let mut params = ParameterDictionary::new();
    params.add_point3f("P", &Point3f::new(0.0, 0.0, 0.0));
    Shape::create(
        "sphere",
        &Transform::identity(),
        &Transform::identity(),
        false,
        &params,
        &HashMap::new(),
    )
    .unwrap()
    .pop()
    .unwrap()
    .into()
}

fn create(params: &ParameterDictionary) -> Result<DiffuseAreaLight, PbrtError> {
    let medium: Option<Arc<pbrt_r4::media::Medium>> = None;
    DiffuseAreaLight::create(&Transform::identity(), &medium, params, &shape())
}

fn png(dir: &std::path::Path, name: &str, pixel: [u8; 3]) -> String {
    let path = dir.join(name);
    let image = image::RgbImage::from_pixel(2, 2, image::Rgb(pixel));
    image.save(&path).unwrap();
    path.to_str().unwrap().to_string()
}

#[test]
fn filename_only_uses_general_reader() {
    let dir = tempfile::tempdir().unwrap();
    let filename = png(dir.path(), "image.png", [64, 128, 192]);
    let mut params = ParameterDictionary::new();
    params.add_string("filename", &filename);
    let light = create(&params).expect("filename-only area light must be accepted");

    let (texels, resolution, color_space) = read_image_with_metadata(&filename).unwrap();
    assert_eq!(texels.len(), 4);
    assert_eq!(resolution, Point2i::new(2, 2));
    assert_eq!(
        color_space.name,
        pbrt_r4::util::spectrum::rgb_to_spectrum::SRGB.name
    );
    let lambda = SampledWavelengths::sample_visible(0.5);
    let emitted = light.l(
        Point3f::new(0.0, 0.0, 1.0),
        Normal3f::new(0.0, 0.0, 1.0),
        Point2f::new(0.5, 0.5),
        Vector3f::new(0.0, 0.0, 1.0),
        &lambda,
    );
    assert!(emitted.values().iter().all(|value| value.is_finite()));
    assert!(emitted.values().iter().any(|value| *value > 0.0));
}

#[test]
fn explicit_l_conflicts_with_filename_even_when_one() {
    let dir = tempfile::tempdir().unwrap();
    let filename = png(dir.path(), "image.png", [64, 128, 192]);
    for add_l in [
        |params: &mut ParameterDictionary| params.add_float("L", 1.0),
        |params: &mut ParameterDictionary| params.add_rgb("L", &[1.0, 1.0, 1.0]),
    ] {
        let mut params = ParameterDictionary::new();
        add_l(&mut params);
        params.add_string("filename", &filename);
        let error = create(&params).err().expect("L and filename must conflict");
        assert!(error.to_string().contains("Both"), "{}", error);
    }
}

#[test]
fn missing_rgb_channels_and_nonfinite_pixels_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let luma_path = dir.path().join("luma.png");
    image::GrayImage::from_pixel(2, 2, image::Luma([128]))
        .save(&luma_path)
        .unwrap();
    let mut params = ParameterDictionary::new();
    params.add_string("filename", luma_path.to_str().unwrap());
    assert!(create(&params).is_err());

    let nan_path = dir.path().join("nan.pfm");
    std::fs::write(&nan_path, b"Pf\n1 1\n-1.0\n\x00\x00\xc0\x7f").unwrap();
    let mut params = ParameterDictionary::new();
    params.add_string("filename", nan_path.to_str().unwrap());
    let error = create(&params).err().expect("NaN pixels must be rejected");
    assert!(error.to_string().to_lowercase().contains("not-a-number"));
}

#[test]
fn power_normalizes_emission_by_v4_formula() {
    let dir = tempfile::tempdir().unwrap();
    let filename = png(dir.path(), "image.png", [64, 128, 192]);

    let mut base_params = ParameterDictionary::new();
    base_params.add_string("filename", &filename);
    let base = create(&base_params).unwrap();

    let power = 7.0;
    let mut power_params = ParameterDictionary::new();
    power_params.add_string("filename", &filename);
    power_params.add_float("power", power);
    let normalized = create(&power_params).unwrap();

    let area = shape().area();
    let (texels, _, color_space) = read_image_with_metadata(&filename).unwrap();
    let luminance = color_space.luminance_vector();
    let mut k_e = 0.0;
    for rgb in &texels {
        let rgb = rgb.to_rgb();
        for channel in 0..3 {
            k_e += rgb[channel] * luminance[channel];
        }
    }
    k_e /= texels.len() as Float;
    k_e *= area * PI;

    let lambda = SampledWavelengths::sample_visible(0.5);
    let point = Point3f::new(0.0, 0.0, 1.0);
    let n = Normal3f::new(0.0, 0.0, 1.0);
    let w = Vector3f::new(0.0, 0.0, 1.0);
    let uv = Point2f::new(0.5, 0.5);
    let base_l = base.l(point, n, uv, w, &lambda).values().to_vec();
    let normalized_l = normalized.l(point, n, uv, w, &lambda).values().to_vec();
    let expected_ratio = power / k_e;
    for (before, after) in base_l.iter().zip(normalized_l.iter()) {
        let actual = *after / *before;
        assert!(
            (actual - expected_ratio).abs() / expected_ratio < 1e-4,
            "ratio {} != {}",
            actual,
            expected_ratio
        );
    }
}
