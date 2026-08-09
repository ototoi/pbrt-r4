use pbrt_r4::cpu::aggregates::create_accelerator::create_accelerator;
use pbrt_r4::cpu::aggregates::Accel;
use pbrt_r4::cpu::primitive::Primitive;
use pbrt_r4::paramdict::ParameterDictionary;

#[test]
fn empty_bvh_accelerator_returns_an_error() {
    let prims: Vec<std::sync::Arc<Primitive>> = Vec::new();
    let params = ParameterDictionary::new();

    match create_accelerator("bvh", &prims, &params) {
        Err(error) => assert!(error.msg.contains("at least one primitive")),
        Ok(_) => panic!("empty BVH construction should return an error"),
    }
}

#[test]
fn empty_kdtree_accelerator_builds_without_panicking() {
    let prims: Vec<std::sync::Arc<Primitive>> = Vec::new();
    let params = ParameterDictionary::new();

    let accel = create_accelerator("kdtree", &prims, &params).expect("empty scene should build");
    match accel {
        Primitive::Accel(Accel::KdTree(_)) => {}
        other => panic!(
            "expected kdtree accelerator, got {:?}",
            std::mem::discriminant(&other)
        ),
    }
}

#[test]
fn unknown_accelerator_name_returns_error() {
    let prims: Vec<std::sync::Arc<Primitive>> = Vec::new();
    let params = ParameterDictionary::new();

    assert!(create_accelerator("not-an-accelerator", &prims, &params).is_err());
}
