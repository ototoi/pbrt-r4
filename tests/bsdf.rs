use pbrt_r4::base::bxdf::*;
use pbrt_r4::bsdf::BSDF;

#[test]
fn bsdf_layout_stays_compact() {
    assert!(std::mem::size_of::<BxDF>() <= 16);
    #[cfg(not(feature = "float-as-double"))]
    assert!(std::mem::size_of::<BSDF>() <= 64);
    #[cfg(feature = "float-as-double")]
    assert!(std::mem::size_of::<BSDF>() <= 128);
}
