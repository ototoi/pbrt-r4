use pbrt_r4::textures::{ImageWrap, MIPMap};
use pbrt_r4::util::base::Float;

#[test]
fn octahedral_wrap_handles_one_by_one_images() {
    let image = pbrt_r4::textures::mipmap::F32MIPMapImage::new(vec![7.0], (1, 1));
    assert_eq!(
        MIPMap::<Float>::texel_static(
            &image,
            -100,
            100,
            ImageWrap::OctahedralSphere,
            ImageWrap::OctahedralSphere,
        ),
        7.0
    );
}
