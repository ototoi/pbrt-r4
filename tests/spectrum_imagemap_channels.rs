use pbrt_r4::prelude::*;

use std::collections::HashMap;
use std::sync::Arc;

fn write_rgba_png(bytes: &[u8]) -> tempfile::NamedTempFile {
    let file = tempfile::Builder::new()
        .prefix("pbrt-r4-spectrum-imagemap-")
        .suffix(".png")
        .tempfile()
        .unwrap();
    let image = image::RgbaImage::from_raw(2, 2, bytes.to_vec()).unwrap();
    image.save(file.path()).unwrap();
    file
}

fn create_spectrum_imagemap(file: &tempfile::NamedTempFile) -> SpectrumTexture {
    let mut parameters = ParameterDictionary::new();
    parameters.add_string("filename", file.path().to_str().unwrap());
    parameters.add_string("encoding", "linear");
    let float_textures = HashMap::<String, Arc<FloatTexture>>::new();
    let spectrum_textures = HashMap::<String, Arc<SpectrumTexture>>::new();
    let texture_parameters =
        TextureParameterDictionary::new(&parameters, &float_textures, &spectrum_textures);
    SpectrumTexture::create(
        "imagemap",
        &Transform::identity(),
        &texture_parameters,
        SpectrumType::Albedo,
    )
    .unwrap()
}

fn channel_count(texture: &SpectrumTexture) -> usize {
    match texture {
        SpectrumTexture::ImageMap(texture) => texture.mipmap_channel_count(),
        _ => panic!("expected an imagemap texture"),
    }
}

#[test]
fn spectrum_opaque_rgba_imagemap_shrinks_to_rgb() {
    let file = write_rgba_png(&[
        26, 51, 77, 255, 26, 51, 77, 255, 26, 51, 77, 255, 26, 51, 77, 255,
    ]);
    let texture = create_spectrum_imagemap(&file);

    assert_eq!(channel_count(&texture), 3);
}

#[test]
fn spectrum_nonopaque_rgba_imagemap_preserves_alpha() {
    let file = write_rgba_png(&[
        26, 51, 77, 255, 26, 51, 77, 254, 26, 51, 77, 255, 26, 51, 77, 255,
    ]);
    let texture = create_spectrum_imagemap(&file);

    assert_eq!(channel_count(&texture), 4);
}
