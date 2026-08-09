use pbrt_r4::options::PbrtOptions;

#[test]
fn v4_mse_and_pixelstats_options_are_parsed() {
    PbrtOptions::set(PbrtOptions::default());
    PbrtOptions::apply_option("msereferenceimage", "\"reference.exr\"").unwrap();
    PbrtOptions::apply_option("msereferenceout", "\"mse.exr\"").unwrap();
    PbrtOptions::apply_option("pixelstats", "true").unwrap();
    let options = PbrtOptions::get();
    assert_eq!(
        options.mse_reference_image.as_deref(),
        Some("reference.exr")
    );
    assert_eq!(options.mse_reference_output.as_deref(), Some("mse.exr"));
    assert!(options.record_pixel_statistics);
}
