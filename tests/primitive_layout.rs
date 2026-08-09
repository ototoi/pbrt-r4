use pbrt_r4::cpu::aggregates::Accel;
use pbrt_r4::cpu::primitive::{
    AnimatedPrimitive, Primitive, SimplePrimitive, TransformedPrimitive,
};

#[test]
fn primitive_enum_stays_small_for_triangle_heavy_scenes() {
    assert!(std::mem::size_of::<Primitive>() <= 24);
    assert_eq!(std::mem::size_of::<SimplePrimitive>(), 16);
    assert_eq!(std::mem::size_of::<TransformedPrimitive>(), 16);
    assert_eq!(std::mem::size_of::<AnimatedPrimitive>(), 16);
    assert_eq!(std::mem::size_of::<Accel>(), 16);
}
