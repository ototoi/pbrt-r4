use pbrt_r4::textures::{ImageFilter, ImageWrap, MIPMap, MIPMapLevelStorage};
use pbrt_r4::util::base::{Float, Point2i};
use pbrt_r4::util::imageio::{ColorEncoding, RawImage, RawImageData};

fn storage_for(raw: RawImage) -> MIPMap<Float> {
    MIPMap::new_from_raw_image(
        &raw,
        ImageFilter::Point,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
    )
}

#[test]
fn raw_u8_image_selects_u8_mipmap_storage_and_round_trips_encoding() {
    let mipmap = storage_for(RawImage {
        data: RawImageData::U8 {
            data: vec![128],
            encoding: ColorEncoding::SRgb,
        },
        resolution: Point2i::new(1, 1),
        channels: 1,
    });

    match &mipmap.storage.pyramid[0].data {
        MIPMapLevelStorage::U8 { data, encoding } => {
            assert_eq!(data, &[128]);
            assert_eq!(*encoding, ColorEncoding::SRgb);
        }
        _ => panic!("expected U8 MIPMap storage"),
    }
    assert!((mipmap.texel(0, 0, 0) - 0.2158605).abs() < 1e-5);
}

#[test]
fn raw_f16_and_f32_images_select_matching_mipmap_storage() {
    let f16 = storage_for(RawImage {
        data: RawImageData::F16(vec![half::f16::from_f32(0.5)]),
        resolution: Point2i::new(1, 1),
        channels: 1,
    });
    assert!(matches!(
        f16.storage.pyramid[0].data,
        MIPMapLevelStorage::F16(_)
    ));

    let f32 = storage_for(RawImage {
        data: RawImageData::F32(vec![0.5]),
        resolution: Point2i::new(1, 1),
        channels: 1,
    });
    assert!(matches!(
        f32.storage.pyramid[0].data,
        MIPMapLevelStorage::F32(_)
    ));
}

#[test]
fn u8_mipmap_downsamples_in_linear_space_before_reencoding() {
    let mipmap = storage_for(RawImage {
        data: RawImageData::U8 {
            data: vec![0, 255],
            encoding: ColorEncoding::SRgb,
        },
        resolution: Point2i::new(2, 1),
        channels: 1,
    });

    let averaged = mipmap.texel(1, 0, 0);
    assert!((averaged - 0.5).abs() < 0.01);
}
