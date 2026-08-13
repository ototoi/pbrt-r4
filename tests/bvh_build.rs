use pbrt_r4::cpu::bvh::sah::split_sah;
use pbrt_r4::cpu::bvh::{BVHBuildNode, BVHPrimitiveInfo, SplitMethod};
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

#[test]
fn sah_matches_v4_split_decision_even_below_max_primitives() {
    let bounds = [
        Bounds3f::from(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
        Bounds3f::from(((10.0, 0.0, 0.0), (11.0, 1.0, 1.0))),
        Bounds3f::from(((20.0, 0.0, 0.0), (21.0, 1.0, 1.0))),
    ];
    let mut primitive_info: Vec<_> = bounds
        .iter()
        .enumerate()
        .map(|(index, bound)| {
            let centroid = (bound.min + bound.max) * 0.5;
            BVHPrimitiveInfo::new(index, bound, &centroid)
        })
        .collect();
    let mut ordered_indices = Vec::new();

    let node = split_sah(
        0,
        &mut primitive_info,
        &mut ordered_indices,
        8,
        SplitMethod::SAH,
    );

    assert_eq!(node.n_primitives, 0);
    assert!(node.children[0].is_some());
    assert!(node.children[1].is_some());
    assert_eq!(node.primitive_count(), 3);
    assert_eq!(ordered_indices.len(), 3);
}
