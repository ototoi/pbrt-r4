#![cfg(feature = "webgpu")]

use std::sync::Arc;

use pbrt_r4::gpu::flat::texture::{
    ColorSpace, ImageFilterMode, ImageValueType, ImageView, ImageWrapMode, Mipmap, MipmapEncoding,
};
use pbrt_r4::gpu::flat::{self, INVALID_INDEX};
use pbrt_r4::gpu::webgpu::scene::resolve_infinite_image_bindings;

fn view(mipmap: u32) -> ImageView {
    ImageView {
        mipmap,
        value_type: ImageValueType::LinearRgb,
        swrap: ImageWrapMode::Clamp,
        twrap: ImageWrapMode::Clamp,
        filter: ImageFilterMode::Nearest,
        scale: 1.0,
        invert: false,
    }
}

fn light(image_index: u32, sampling_model: u32) -> flat::Light {
    flat::Light {
        kind: flat::LightKind::ImageInfinite,
        attributes: Vec::new(),
        image_index,
        sampling_model,
    }
}

fn mipmap(color_space: ColorSpace) -> Arc<Mipmap> {
    Arc::new(Mipmap {
        levels: Vec::new(),
        color_space,
        encoding: MipmapEncoding::Linear,
    })
}

#[test]
fn resolves_shared_images_and_color_spaces_to_binding_slots() {
    let mipmaps = [
        ColorSpace::Unknown,
        ColorSpace::Srgb,
        ColorSpace::Aces2065,
        ColorSpace::DciP3,
        ColorSpace::Rec2020,
    ]
    .map(mipmap);
    let mut views = (0..5).map(view).collect::<Vec<_>>();
    let mut duplicate = view(0);
    duplicate.filter = ImageFilterMode::Bilinear;
    views.push(duplicate);
    let bindings = [4, 1, 3, 0, 2];
    let mut lights = (0..5)
        .map(|index| light(index, 10 + index))
        .collect::<Vec<_>>();
    lights.push(light(0, 20));
    lights.push(light(INVALID_INDEX, 30));
    let payloads = resolve_infinite_image_bindings(&lights, &mipmaps, &views, &bindings).unwrap();
    assert_eq!(payloads.len(), 6);
    for (key, expected) in [
        (10, 3),
        (11, 1),
        (12, 0x1000_0004),
        (13, 0x2000_0002),
        (14, 0x3000_0000),
        (20, 3),
    ] {
        assert_eq!(payloads[&key], expected);
    }
    assert!(!payloads.contains_key(&30));
}

#[test]
fn preserves_view_binding_and_mipmap_validation_order() {
    let lights = [light(0, 4)];
    for (views, bindings, expected) in [
        (
            Vec::new(),
            Vec::new(),
            "Infinite light image view was not registered.",
        ),
        (
            vec![view(0)],
            Vec::new(),
            "Infinite light image binding was not generated.",
        ),
        (
            vec![view(0)],
            vec![0],
            "Infinite light references an invalid mipmap.",
        ),
    ] {
        let error = resolve_infinite_image_bindings(&lights, &[], &views, &bindings)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn skips_lights_without_images_and_accepts_empty_input() {
    assert!(resolve_infinite_image_bindings(&[], &[], &[], &[])
        .unwrap()
        .is_empty());
    assert!(
        resolve_infinite_image_bindings(&[light(INVALID_INDEX, 0)], &[], &[], &[])
            .unwrap()
            .is_empty()
    );
}
