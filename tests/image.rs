use pbrt_r4::util::base::Point2i;
use pbrt_r4::util::image::Image;
use pbrt_r4::util::spectrum::rgb_to_spectrum::SRGB;
use pbrt_r4::util::spectrum::RGBSpectrum;

#[test]
fn try_with_color_space_rejects_mismatched_texel_count() {
    let result = Image::try_with_color_space(
        Point2i::new(2, 2),
        vec![RGBSpectrum::from([0.0, 0.0, 0.0])],
        &SRGB,
    );

    let err = match result {
        Ok(_) => panic!("mismatched texel count should return an error"),
        Err(err) => err,
    };

    assert!(
        err.to_string().contains("image texel count mismatch"),
        "unexpected error: {err}"
    );
}
