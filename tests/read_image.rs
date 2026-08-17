use pbrt_r4::util::base::inverse_gamma_correct;
use pbrt_r4::util::base::Point2i;
use pbrt_r4::util::imageio::read_image::{
    read_image_with_encoding, read_raw_image_with_encoding, RawImage, RawImageData,
};
use pbrt_r4::util::imageio::ColorEncoding;

fn write_rgba_png(bytes: &[u8], width: u32, height: u32) -> tempfile::NamedTempFile {
    let file = tempfile::Builder::new()
        .prefix("pbrt-r4-rgba-")
        .suffix(".png")
        .tempfile()
        .expect("temporary PNG should be created");
    let image = image::RgbaImage::from_raw(width, height, bytes.to_vec()).unwrap();
    image
        .save(file.path())
        .expect("temporary PNG should be written");
    file
}

#[test]
fn raw_image_exposes_conventional_channel_names() {
    let raw = RawImage {
        data: RawImageData::F32(vec![0.0; 4]),
        resolution: Point2i::new(1, 1),
        channels: 4,
    };
    assert_eq!(raw.channel_names(), ["R", "G", "B", "A"]);
}

#[test]
fn raw_image_exposes_luma_alpha_channel_names() {
    let raw = RawImage {
        data: RawImageData::F32(vec![0.0; 2]),
        resolution: Point2i::new(1, 1),
        channels: 2,
    };
    assert_eq!(raw.channel_names(), ["Y", "A"]);
}

#[test]
fn raw_rgba_read_preserves_alpha() {
    let file = write_rgba_png(&[10, 20, 30, 40, 50, 60, 70, 80], 2, 1);
    let raw =
        read_raw_image_with_encoding(file.path().to_str().unwrap(), ColorEncoding::Linear).unwrap();

    assert_eq!(raw.channels, 4);
    assert_eq!(raw.resolution, Point2i::new(2, 1));
    assert_eq!(
        raw.data_f32(),
        vec![
            10.0 / 255.0,
            20.0 / 255.0,
            30.0 / 255.0,
            40.0 / 255.0,
            50.0 / 255.0,
            60.0 / 255.0,
            70.0 / 255.0,
            80.0 / 255.0,
        ]
    );
}

#[test]
fn rgba_spectrum_read_ignores_alpha() {
    let file = write_rgba_png(&[10, 20, 30, 40], 1, 1);
    let (spectra, resolution) =
        read_image_with_encoding(file.path().to_str().unwrap(), ColorEncoding::Linear).unwrap();
    let rgb = spectra[0].to_rgb();

    assert_eq!(resolution, Point2i::new(1, 1));
    assert_eq!(rgb, [10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0]);
}

#[test]
fn raw_rgba_gamma_correction_applies_to_alpha_as_v4_does() {
    let file = write_rgba_png(&[128, 128, 128, 128], 1, 1);
    let raw =
        read_raw_image_with_encoding(file.path().to_str().unwrap(), ColorEncoding::SRgb).unwrap();
    let expected = inverse_gamma_correct(128.0 / 255.0);

    assert!((raw.channel(0, 0) - expected).abs() < 1e-6);
    assert!((raw.channel(0, 3) - expected).abs() < 1e-6);
}
