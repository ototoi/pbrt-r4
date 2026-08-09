use pbrt_r4::util::tensor::{TensorFile, TensorType};
use std::path::Path;

#[test]
fn type_sizes_match_v4_table() {
    assert_eq!(TensorType::UInt8.size(), 1);
    assert_eq!(TensorType::Int16.size(), 2);
    assert_eq!(TensorType::Float32.size(), 4);
    assert_eq!(TensorType::Float64.size(), 8);
    assert_eq!(TensorType::Invalid.size(), 0);
}

#[test]
fn opens_real_bsdf_file_when_available() {
    let path =
        "/mnt/hdd1/src/other/pbrt-r4-devkit/pbrt-v4-scenes/sportscar/bsdfs/paper_white_spec.bsdf";
    if !Path::new(path).exists() {
        eprintln!("(skipping: {} not present in this checkout)", path);
        return;
    }
    let tf = TensorFile::open(path).expect("paper_white_spec.bsdf should parse");
    for required in &[
        "theta_i",
        "phi_i",
        "wavelengths",
        "ndf",
        "sigma",
        "vndf",
        "luminance",
        "spectra",
        "description",
        "jacobian",
    ] {
        assert!(
            tf.has_field(required),
            "expected field `{}` in paper_white_spec.bsdf",
            required
        );
    }
    let wavelengths = tf
        .field("wavelengths")
        .and_then(|f| f.as_f32_slice())
        .expect("wavelengths should be float32 1D");
    assert!(!wavelengths.is_empty());
    assert!(wavelengths[0] >= 350.0 && wavelengths[0] <= 850.0);
}
