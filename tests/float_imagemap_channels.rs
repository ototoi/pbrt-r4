use pbrt_r4::prelude::*;

use std::collections::HashMap;
use std::sync::Arc;

fn write_rgba_png(bytes: &[u8], width: u32, height: u32) -> tempfile::NamedTempFile {
    let file = tempfile::Builder::new()
        .prefix("pbrt-r4-float-imagemap-")
        .suffix(".png")
        .tempfile()
        .expect("temporary PNG should be created");
    let image = image::RgbaImage::from_raw(width, height, bytes.to_vec()).unwrap();
    image
        .save(file.path())
        .expect("temporary PNG should be written");
    file
}

fn create_float_imagemap(file: &tempfile::NamedTempFile) -> FloatTexture {
    let mut geom_params = ParameterDictionary::new();
    geom_params.add_string("filename", file.path().to_str().unwrap());
    geom_params.add_string("encoding", "linear");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&geom_params, &f_tex, &s_tex);

    FloatTexture::create("imagemap", &Transform::identity(), &tp).unwrap()
}

#[test]
fn opaque_rgba_float_imagemap_uses_rgb_average() {
    let file = write_rgba_png(
        &[
            0, 127, 254, 255, // lower-left
            0, 127, 254, 255, // lower-right
            0, 127, 254, 255, // upper-left
            0, 127, 254, 255, // upper-right
        ],
        2,
        2,
    );
    let texture = create_float_imagemap(&file);
    let mut ctx = TextureEvalContext::default();
    ctx.uv = Point2f::new(0.5, 0.5);

    let value = texture.evaluate(&ctx);
    assert!((value - 0.49803922).abs() < 1e-5);
}

#[test]
fn nonopaque_rgba_float_imagemap_uses_alpha_for_bilinear() {
    let file = write_rgba_png(
        &[
            26, 51, 77, 51, // lower-left
            26, 51, 77, 102, // lower-right
            26, 51, 77, 153, // upper-left
            26, 51, 77, 204, // upper-right
        ],
        2,
        2,
    );
    let texture = create_float_imagemap(&file);
    let mut ctx = TextureEvalContext::default();
    ctx.uv = Point2f::new(0.5, 0.5);

    let value = texture.evaluate(&ctx);
    assert!((value - 0.5).abs() < 1e-5);
}

#[test]
fn two_channel_float_imagemap_uses_luma_and_ignores_alpha() {
    let file = tempfile::Builder::new()
        .prefix("pbrt-r4-two-channel-imagemap-")
        .suffix(".png")
        .tempfile()
        .unwrap();
    let image = image::GrayAlphaImage::from_raw(1, 1, vec![51, 255]).unwrap();
    image.save(file.path()).unwrap();

    let mut geom_params = ParameterDictionary::new();
    geom_params.add_string("filename", file.path().to_str().unwrap());
    geom_params.add_string("encoding", "linear");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&geom_params, &f_tex, &s_tex);

    let texture = FloatTexture::create("imagemap", &Transform::identity(), &tp).unwrap();
    let value = texture.evaluate(&TextureEvalContext::default());
    assert!((value - 51.0 / 255.0).abs() < 1e-5);
    match texture {
        FloatTexture::ImageMap(image_map) => {
            assert_eq!(image_map.mipmap_channel_count(), 1);
        }
        _ => panic!("expected imagemap texture"),
    }
}

#[test]
fn two_channel_spectrum_imagemap_uses_luma_and_ignores_alpha() {
    let file = tempfile::Builder::new()
        .prefix("pbrt-r4-two-channel-spectrum-imagemap-")
        .suffix(".png")
        .tempfile()
        .unwrap();
    let image = image::GrayAlphaImage::from_raw(1, 1, vec![51, 255]).unwrap();
    image.save(file.path()).unwrap();

    let mut geom_params = ParameterDictionary::new();
    geom_params.add_string("filename", file.path().to_str().unwrap());
    geom_params.add_string("encoding", "linear");
    let f_tex: HashMap<String, Arc<FloatTexture>> = HashMap::new();
    let s_tex: HashMap<String, Arc<SpectrumTexture>> = HashMap::new();
    let tp = TextureParameterDictionary::new(&geom_params, &f_tex, &s_tex);

    let texture = SpectrumTexture::create(
        "imagemap",
        &Transform::identity(),
        &tp,
        SpectrumType::Albedo,
    )
    .unwrap();
    let value = texture
        .evaluate(
            &TextureEvalContext::default(),
            &SampledWavelengths::sample_visible(0.5),
        )
        .max_component_value();
    assert!((value - 51.0 / 255.0).abs() < 1e-5);
}
