use pbrt_r4::base::light::Light;
use pbrt_r4::media::MediumInterface;
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::util::base::{Float, Point3f, Vector3f};
use pbrt_r4::util::geometry::Ray;
use pbrt_r4::util::imageio::{read_image_with_encoding, ColorEncoding};
use pbrt_r4::util::spectrum::rgb_to_spectrum::SRGB;
use pbrt_r4::util::spectrum::{
    spectrum_to_photometric, SampledSpectrum, SampledWavelengths, Spectrum,
};
use pbrt_r4::util::transform::Transform;

fn infinite_light_params() -> ParameterDictionary {
    ParameterDictionary::new()
}

#[test]
fn infinite_light_accepts_filename() {
    let params = {
        let mut params = infinite_light_params();
        params.add_string(
            "filename",
            "tests/scenes/crown-step13-textured-material-panels.exr",
        );
        params
    };
    let light = Light::create(
        "infinite",
        &Transform::identity(),
        &MediumInterface::new(),
        &params,
        &Transform::identity(),
    )
    .expect("infinite light with filename should create");
    assert!(matches!(&*light, Light::Infinite(_)));
}

#[test]
fn infinite_light_accepts_mapname_when_filename_is_missing() {
    let params = {
        let mut params = infinite_light_params();
        params.add_string(
            "mapname",
            "tests/scenes/crown-step13-textured-material-panels.exr",
        );
        params
    };
    let light = Light::create(
        "infinite",
        &Transform::identity(),
        &MediumInterface::new(),
        &params,
        &Transform::identity(),
    )
    .expect("infinite light with mapname should create");
    assert!(matches!(&*light, Light::Infinite(_)));
}

#[test]
fn infinite_light_without_filename_or_mapname_stays_uniform() {
    let params = infinite_light_params();
    let light = Light::create(
        "infinite",
        &Transform::identity(),
        &MediumInterface::new(),
        &params,
        &Transform::identity(),
    )
    .expect("default infinite light should create");
    assert!(matches!(&*light, Light::Infinite(_)));
}

#[test]
fn infinite_light_png_uses_srgb_encoding_by_default() {
    let file = tempfile::Builder::new()
        .prefix("pbrt-r4-infinite-light-")
        .suffix(".png")
        .tempfile()
        .expect("temporary PNG should be created");
    image::RgbImage::from_pixel(1, 1, image::Rgb([128, 96, 32]))
        .save(file.path())
        .expect("temporary PNG should be written");
    let filename = file.path().to_str().unwrap();

    let mut params = infinite_light_params();
    params.add_string("filename", filename);
    let light = Light::create(
        "infinite",
        &Transform::identity(),
        &MediumInterface::new(),
        &params,
        &Transform::identity(),
    )
    .expect("infinite light with PNG should create");

    let lambda = SampledWavelengths::sample_visible(0.5);
    let ray = Ray::new(
        &Point3f::zero(),
        &Vector3f::new(0.0, 0.0, 1.0),
        Float::INFINITY,
        0.0,
    );
    let actual = light.le(&ray, &lambda);

    let (pixels, _) = read_image_with_encoding(filename, ColorEncoding::SRgb).unwrap();
    let expected_rgb = pixels[0].to_rgb();
    let unit_illuminant = Spectrum::from_rgb_illuminant(&[1.0, 1.0, 1.0]);
    let scale = 1.0 / spectrum_to_photometric(&unit_illuminant);
    let expected = SRGB.illuminant_to_sampled_spectrum(expected_rgb, &lambda) * scale;

    assert!(
        SampledSpectrum::near_equal(&actual, &expected, 1e-6),
        "PNG infinite-light radiance should decode the default sRGB encoding"
    );
}

#[test]
fn infinite_light_16bit_png_uses_srgb_encoding_by_default() {
    let file = tempfile::Builder::new()
        .prefix("pbrt-r4-infinite-light-16bit-")
        .suffix(".png")
        .tempfile()
        .expect("temporary PNG should be created");
    let encoded_rgb = [32768u16, 16384, 65535];
    image::ImageBuffer::from_pixel(1, 1, image::Rgb(encoded_rgb))
        .save(file.path())
        .expect("temporary 16-bit PNG should be written");
    let filename = file.path().to_str().unwrap();

    let (linear_pixels, resolution) =
        read_image_with_encoding(filename, ColorEncoding::Linear).unwrap();
    assert_eq!(resolution.x, 1);
    assert_eq!(resolution.y, 1);
    let stored_rgb = linear_pixels[0].to_rgb();
    for (stored, encoded) in stored_rgb.into_iter().zip(encoded_rgb) {
        assert!((stored - encoded as Float / 65535.0).abs() < 1e-6);
    }

    let mut params = infinite_light_params();
    params.add_string("filename", filename);
    let light = Light::create(
        "infinite",
        &Transform::identity(),
        &MediumInterface::new(),
        &params,
        &Transform::identity(),
    )
    .expect("infinite light with 16-bit PNG should create");

    let lambda = SampledWavelengths::sample_visible(0.5);
    let ray = Ray::new(
        &Point3f::zero(),
        &Vector3f::new(0.0, 0.0, 1.0),
        Float::INFINITY,
        0.0,
    );
    let actual = light.le(&ray, &lambda);
    let expected_rgb =
        encoded_rgb.map(|sample| ColorEncoding::SRgb.to_linear(sample as Float / 65535.0));
    let unit_illuminant = Spectrum::from_rgb_illuminant(&[1.0, 1.0, 1.0]);
    let scale = 1.0 / spectrum_to_photometric(&unit_illuminant);
    let expected = SRGB.illuminant_to_sampled_spectrum(expected_rgb, &lambda) * scale;

    assert!(
        SampledSpectrum::near_equal(&actual, &expected, 1e-6),
        "16-bit PNG infinite-light radiance should decode the default sRGB encoding"
    );
}
