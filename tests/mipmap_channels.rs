use pbrt_r4::prelude::*;
use pbrt_r4::textures::{MIPMap, MIPMapStorage};

#[test]
fn float_view_preserves_rgb_channels_and_averages_for_bilerp() {
    let mipmap = MIPMap::<Float>::new_with_raw_channels(
        &Point2i::new(2, 2),
        &[
            1.0, 0.0, 0.0, // lower-left
            0.0, 1.0, 0.0, // lower-right
            0.0, 0.0, 1.0, // upper-left
            1.0, 1.0, 1.0, // upper-right
        ],
        3,
        ImageFilter::Bilinear,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
    );
    assert_eq!(mipmap.storage.pyramid[0].channels, 3);
    let value = mipmap.triangle(0, &Point2f::new(0.5, 0.5));
    assert!((value - 0.5).abs() < 1e-6);
}

#[test]
fn float_view_uses_alpha_for_rgba_bilerp_but_texel_uses_first_channel() {
    let mipmap = MIPMap::<Float>::new_with_raw_channels(
        &Point2i::new(2, 2),
        &[
            0.1, 0.0, 0.0, 0.2, // lower-left
            0.1, 0.0, 0.0, 0.4, // lower-right
            0.1, 0.0, 0.0, 0.6, // upper-left
            0.1, 0.0, 0.0, 0.8, // upper-right
        ],
        4,
        ImageFilter::Bilinear,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
    );
    assert_eq!(mipmap.storage.pyramid[0].channels, 4);
    assert!((mipmap.triangle(0, &Point2f::new(0.5, 0.5)) - 0.5).abs() < 1e-6);
    assert!((mipmap.texel(0, 0, 0) - 0.1).abs() < 1e-6);
}

#[test]
fn spectrum_view_ignores_alpha_for_rgba_bilerp() {
    let mipmap = MIPMap::<RGBSpectrum>::new_with_raw_channels(
        &Point2i::new(2, 2),
        &[
            0.1, 0.2, 0.3, 0.9, // lower-left
            0.1, 0.2, 0.3, 0.8, // lower-right
            0.1, 0.2, 0.3, 0.7, // upper-left
            0.1, 0.2, 0.3, 0.6, // upper-right
        ],
        4,
        ImageFilter::Bilinear,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
    );
    let rgb = mipmap.triangle(0, &Point2f::new(0.5, 0.5)).to_rgb();
    assert!((rgb[0] - 0.1).abs() < 1e-6);
    assert!((rgb[1] - 0.2).abs() < 1e-6);
    assert!((rgb[2] - 0.3).abs() < 1e-6);
}

#[test]
fn mipmap_storage_contains_only_pyramid_data() {
    let mipmap = MIPMap::<Float>::new(
        &Point2i::new(1, 1),
        &[0.5],
        ImageFilter::Bilinear,
        8.0,
        ImageWrap::Repeat,
        ImageWrap::Clamp,
    );
    let _: &MIPMapStorage = &mipmap.storage;
    assert_eq!(mipmap.storage.pyramid.len(), 1);
    assert_eq!(mipmap.filter, ImageFilter::Bilinear);
    assert_eq!(mipmap.swrap_mode, ImageWrap::Repeat);
    assert_eq!(mipmap.twrap_mode, ImageWrap::Clamp);
}
