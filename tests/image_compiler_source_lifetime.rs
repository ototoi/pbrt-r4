use pbrt_r4::gpu::flat::texture::{
    ColorSpace, ImageCompiler, ImageOptimizationPolicy, ImageValueType, Mipmap, MipmapEncoding,
    MipmapLevel, MipmapLevelData,
};
use std::sync::Arc;

fn source(value: f32) -> Arc<Mipmap> {
    Arc::new(Mipmap {
        levels: vec![MipmapLevel {
            resolution: [1, 1],
            channels: 1,
            data: MipmapLevelData::F32(vec![value]),
        }],
        color_space: ColorSpace::Srgb,
        encoding: MipmapEncoding::Linear,
    })
}

fn scalar(mipmap: &Arc<Mipmap>) -> f32 {
    let MipmapLevelData::F32(values) = &mipmap.levels[0].data else {
        panic!("expected F32 storage");
    };
    values[0]
}

fn policy() -> ImageOptimizationPolicy {
    ImageOptimizationPolicy {
        allow_f16: false,
        max_absolute_error: 0.0,
        max_relative_error: 0.0,
    }
}

#[test]
fn same_source_and_usage_is_shared() {
    let mut compiler = ImageCompiler::new(policy());
    let source = source(1.0);
    let first = compiler.compile(&source, ImageValueType::Float).unwrap();
    let second = compiler.compile(&source, ImageValueType::Float).unwrap();
    assert!(Arc::ptr_eq(&first, &second));
}

#[test]
fn different_usage_has_its_own_format() {
    let mut compiler = ImageCompiler::new(policy());
    let source = source(0.25);
    let float = compiler.compile(&source, ImageValueType::Float).unwrap();
    let rgb = compiler
        .compile(&source, ImageValueType::LinearRgb)
        .unwrap();
    assert!(!Arc::ptr_eq(&float, &rgb));
    assert_eq!(scalar(&float), 0.25);
    assert_eq!(scalar(&rgb), 0.25);
    let MipmapLevelData::F32(values) = &rgb.levels[0].data else {
        panic!("expected F32 storage");
    };
    assert_eq!(values.len(), 3);
}

#[test]
fn compiled_entry_keeps_temporary_source_alive() {
    let mut compiler = ImageCompiler::new(policy());
    let first = source(1.0);
    let weak = Arc::downgrade(&first);
    let compiled = compiler.compile(&first, ImageValueType::Float).unwrap();
    assert_eq!(scalar(&compiled), 1.0);
    drop(first);
    assert!(
        weak.upgrade().is_some(),
        "cache must retain the source allocation"
    );

    // Past the drop, allocations may reuse the address of `first`.  Every
    // new source must still be compiled from its own values.
    for n in 0..256 {
        let expected = n as f32 + 2.0;
        let temporary = source(expected);
        let compiled = compiler.compile(&temporary, ImageValueType::Float).unwrap();
        assert_eq!(scalar(&compiled), expected, "iteration {}", n);
    }
}
