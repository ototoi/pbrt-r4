use pbrt_r4::cpu::aggregates::bvh::accel::BVHTree;
use pbrt_r4::cpu::aggregates::bvh::build::SplitMethod;
use pbrt_r4::prelude::*;

fn test_bounds() -> Vec<Bounds3f> {
    (0..500)
        .map(|i| {
            let x = (i % 25) as Float * 0.2;
            let y = (i / 25) as Float * 0.2;
            Bounds3f::from(((x, y, 2.0), (x + 0.1, y + 0.1, 2.1)))
        })
        .collect()
}

fn test_ray(x: Float, y: Float) -> Ray {
    Ray::new(
        &Point3f::new(x, y, 0.0),
        &Vector3f::new(0.0, 0.0, 1.0),
        Float::INFINITY,
        0.0,
    )
}

#[test]
fn tree_order_is_a_permutation() {
    let bounds = test_bounds();
    let (_, order) = BVHTree::new(&bounds, 8, SplitMethod::HLBVH);
    let mut sorted = order;
    sorted.sort_unstable();
    assert_eq!(sorted, (0..bounds.len()).collect::<Vec<_>>());
}

#[test]
fn tree_nearest_hit_matches_brute_force() {
    let bounds = test_bounds();
    let (tree, order) = BVHTree::new(&bounds, 8, SplitMethod::HLBVH);
    for j in 0..50 {
        for i in 0..50 {
            let ray = test_ray(i as Float * 0.1, j as Float * 0.1);
            let expected = bounds
                .iter()
                .enumerate()
                .filter_map(|(index, bound)| {
                    bound
                        .intersect_p(&ray, Float::INFINITY)
                        .map(|(t, _)| (t, index))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0));
            let actual = tree.intersect(&ray, Float::INFINITY, |mut range, limit| {
                range
                    .filter_map(|slot| {
                        let index = order[slot];
                        bounds[index]
                            .intersect_p(&ray, limit)
                            .map(|(t, _)| (t, index))
                    })
                    .min_by(|a, b| a.0.total_cmp(&b.0))
            });
            assert_eq!(actual, expected.map(|v| v.1));
            assert_eq!(
                tree.intersect_p(&ray, Float::INFINITY, |mut range, limit| {
                    range.any(|slot| bounds[order[slot]].intersect_p(&ray, limit).is_some())
                }),
                expected.is_some()
            );
        }
    }
}

#[test]
fn tree_respects_t_max() {
    let bounds = test_bounds();
    let (tree, _) = BVHTree::new(&bounds, 8, SplitMethod::HLBVH);
    let ray = test_ray(0.05, 0.05);
    assert!(tree.intersect(&ray, 1.0, |_, _| Some((2.0, ()))).is_none());
    assert!(!tree.intersect_p(&ray, 1.0, |_, _| true));
}

#[test]
fn tree_clamps_leaf_size() {
    let bounds = test_bounds();
    let (tree, order) = BVHTree::new(&bounds, 1000, SplitMethod::HLBVH);
    let ray = test_ray(0.05, 0.05);
    let hit = tree.intersect(&ray, Float::INFINITY, |mut range, limit| {
        range
            .filter_map(|slot| {
                bounds[order[slot]]
                    .intersect_p(&ray, limit)
                    .map(|(t, _)| (t, ()))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
    });
    assert!(hit.is_some());
}
