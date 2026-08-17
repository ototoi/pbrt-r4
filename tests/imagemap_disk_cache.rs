use pbrt_r4::options::PbrtOptions;
use pbrt_r4::prelude::*;

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use std::thread;

fn write_rgba_png() -> tempfile::NamedTempFile {
    let file = tempfile::Builder::new()
        .prefix("pbrt-r4-imagemap-disk-cache-")
        .suffix(".png")
        .tempfile()
        .unwrap();
    let image = image::RgbaImage::from_raw(
        2,
        2,
        vec![
            26, 51, 77, 255, 26, 51, 77, 254, 26, 51, 77, 255, 26, 51, 77, 255,
        ],
    )
    .unwrap();
    image.save(file.path()).unwrap();
    file
}

fn channel_count(texture: &FloatTexture) -> usize {
    match texture {
        FloatTexture::ImageMap(texture) => texture.mipmap_channel_count(),
        _ => panic!("expected an imagemap texture"),
    }
}

fn evaluate(texture: &FloatTexture) -> Float {
    let mut ctx = TextureEvalContext::default();
    ctx.uv = Point2f::new(0.37, 0.63);
    texture.evaluate(&ctx)
}

#[test]
fn imagemap_rebuilds_a_corrupt_disk_cache() {
    PbrtOptions::set(PbrtOptions::default());
    let file = write_rgba_png();
    let mut parameter_values = ParameterDictionary::new();
    parameter_values.add_string("filename", file.path().to_str().unwrap());
    parameter_values.add_string("encoding", "linear");
    let float_textures = HashMap::<String, Arc<FloatTexture>>::new();
    let spectrum_textures = HashMap::<String, Arc<SpectrumTexture>>::new();
    let parameters =
        TextureParameterDictionary::new(&parameter_values, &float_textures, &spectrum_textures);

    let texture = FloatTexture::create("imagemap", &Transform::identity(), &parameters).unwrap();
    assert_eq!(channel_count(&texture), 4);
    let first_value = evaluate(&texture);

    let texinfo = create_texinfo(&parameters).unwrap().unwrap();
    let mut mipmap_texinfo = texinfo.clone();
    mipmap_texinfo.scale = 1.0;
    let cache_dir = std::path::PathBuf::from(get_mipmap_cache_dir(&mipmap_texinfo)).join("float");
    assert!(cache_dir.join("COMPLETE").exists());

    let hit = FloatTexture::create("imagemap", &Transform::identity(), &parameters).unwrap();
    assert_eq!(channel_count(&hit), 4);
    assert_eq!(evaluate(&hit), first_value);

    fs::write(cache_dir.join("level-000.bin"), [0_u8]).unwrap();
    let rebuilt = FloatTexture::create("imagemap", &Transform::identity(), &parameters).unwrap();
    assert_eq!(channel_count(&rebuilt), 4);
    assert!(cache_dir.join("COMPLETE").exists());
    assert!(fs::metadata(cache_dir.join("level-000.bin")).unwrap().len() > 1);

    let off_file = write_rgba_png();
    let off_path = off_file.path().to_path_buf();
    let (off_value, off_cache_exists) = thread::spawn(move || {
        let mut parameter_values = ParameterDictionary::new();
        parameter_values.add_string("filename", off_path.to_str().unwrap());
        parameter_values.add_string("encoding", "linear");
        let float_textures = HashMap::<String, Arc<FloatTexture>>::new();
        let spectrum_textures = HashMap::<String, Arc<SpectrumTexture>>::new();
        let parameters =
            TextureParameterDictionary::new(&parameter_values, &float_textures, &spectrum_textures);
        let mut options = PbrtOptions::default();
        options.texture_cache = false;
        PbrtOptions::set(options);
        let texture =
            FloatTexture::create("imagemap", &Transform::identity(), &parameters).unwrap();
        let texinfo = create_texinfo(&parameters).unwrap().unwrap();
        let cache_dir = std::path::PathBuf::from(get_mipmap_cache_dir(&texinfo)).join("float");
        (evaluate(&texture), cache_dir.exists())
    })
    .join()
    .unwrap();
    assert_eq!(off_value, first_value);
    assert!(!off_cache_exists);
}
