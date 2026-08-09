use pbrt_r4::base::medium::Medium;

#[test]
fn medium_enum_stays_pointer_sized() {
    assert!(std::mem::size_of::<Medium>() <= 16);
}
