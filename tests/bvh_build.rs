use pbrt_r4::cpu::bvh::BVHBuildNode;
use pbrt_r4::prelude::*;

#[test]
fn bvh_build_node_interior_uses_union_bounds() {
    let b0 = Bounds3f::from(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)));
    let b1 = Bounds3f::from(((-2.0, -1.0, -3.0), (0.5, 2.0, 4.0)));
    let c0 = Box::new(BVHBuildNode::init_leaf(0, 1, &b0));
    let c1 = Box::new(BVHBuildNode::init_leaf(1, 1, &b1));

    let interior = BVHBuildNode::init_interior(0, Some(c0), Some(c1));
    let expected = b0.union(&b1);
    assert_eq!(interior.bounds, expected);
}

#[test]
fn bvh_build_node_interior_without_children_does_not_panic() {
    let interior = BVHBuildNode::init_interior(0, None, None);
    assert_eq!(interior.bounds, Bounds3f::default());
    assert!(interior.children[0].is_none());
    assert!(interior.children[1].is_none());
}
