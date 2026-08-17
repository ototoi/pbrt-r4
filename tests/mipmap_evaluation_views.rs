use pbrt_r4::prelude::*;
use pbrt_r4::textures::{MIPMap, MIPMapStorageKind};

fn rgba_mipmap(filter: ImageFilter) -> MIPMap<Float> {
    MIPMap::new_with_raw_channels_and_storage(
        &Point2i::new(2, 2),
        &[
            0.1, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.4, 0.1, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.8,
        ],
        4,
        filter,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
        MIPMapStorageKind::F32,
    )
}

#[test]
fn float_point_lookup_uses_r_while_bilinear_uses_alpha() {
    let point = rgba_mipmap(ImageFilter::Point);
    let bilinear = rgba_mipmap(ImageFilter::Bilinear);
    let st = Point2f::new(0.5, 0.5);
    let zero = Vector2f::new(0.0, 0.0);

    assert!((point.lookup_delta(&st, &zero, &zero) - 0.1).abs() < 1e-6);
    assert!((bilinear.lookup_delta(&st, &zero, &zero) - 0.5).abs() < 1e-6);
}

#[test]
fn float_trilinear_lookup_uses_alpha() {
    let mipmap = rgba_mipmap(ImageFilter::Trilinear);
    let st = Point2f::new(0.5, 0.5);
    let zero = Vector2f::new(0.0, 0.0);

    assert!((mipmap.lookup_delta(&st, &zero, &zero) - 0.5).abs() < 1e-6);
}

#[test]
fn float_ewa_lookup_uses_r() {
    let mipmap = rgba_mipmap(ImageFilter::EWA);
    let st = Point2f::new(0.5, 0.5);
    let dst0 = Vector2f::new(0.5, 0.0);
    let dst1 = Vector2f::new(0.0, 0.5);

    assert!((mipmap.lookup_delta(&st, &dst0, &dst1) - 0.1).abs() < 1e-6);
}
