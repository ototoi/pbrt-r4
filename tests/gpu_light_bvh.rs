use pbrt_r4::gpu::ir::flat::{
    build_light_bounds, build_light_bvh, light_bvh_pmf, sample_light_bvh, LightBVHNode,
    LightBoundInput, LightKind, LightRecord,
};

fn point_inputs() -> Vec<LightBoundInput> {
    vec![
        LightBoundInput::Point {
            handle: 0,
            world_position: [-4.0, 0.0, 0.0],
            intensity_max: 1.0,
            scale: 1.0,
        },
        LightBoundInput::Point {
            handle: 1,
            world_position: [0.0, 0.0, 0.0],
            intensity_max: 2.0,
            scale: 1.0,
        },
        LightBoundInput::Point {
            handle: 2,
            world_position: [4.0, 0.0, 0.0],
            intensity_max: 3.0,
            scale: 1.0,
        },
    ]
}

#[test]
fn empty_and_zero_power_lights_produce_empty_bvh() {
    let empty = build_light_bvh(&[], &[]).unwrap();
    assert!(empty.nodes.is_empty());
    assert!(empty.all_bounds.is_none());

    let zero = build_light_bounds(&[LightBoundInput::Point {
        handle: 0,
        world_position: [0.0; 3],
        intensity_max: 0.0,
        scale: 1.0,
    }])
    .unwrap();
    let records = [LightRecord {
        kind: LightKind::Point,
        payload: 0,
    }];
    let bvh = build_light_bvh(&records, &zero).unwrap();
    assert!(bvh.nodes.is_empty());
    assert_eq!(bvh.handle_to_leaf, vec![u32::MAX]);
}

#[test]
fn bvh_has_dfs_layout_and_handle_mapping() {
    let bounds = build_light_bounds(&point_inputs()).unwrap();
    let records = vec![
        LightRecord {
            kind: LightKind::Point,
            payload: 0,
        };
        3
    ];
    let bvh = build_light_bvh(&records, &bounds).unwrap();
    assert_eq!(bvh.nodes.len(), 5);
    assert_eq!(bvh.bounded_handles, vec![0, 1, 2]);
    assert!(matches!(bvh.nodes[0], LightBVHNode::Interior { .. }));
    assert_eq!(bvh.nodes[1].parent(), 0);
    assert_eq!(bvh.nodes[2].parent(), 0);
    for (handle, &leaf) in bvh.handle_to_leaf.iter().enumerate() {
        assert!(
            matches!(bvh.nodes[leaf as usize], LightBVHNode::Leaf { light_handle, .. } if light_handle == handle as u32)
        );
    }
    assert_eq!(bvh.all_bounds.unwrap().min, [-4.0, 0.0, 0.0]);
    assert_eq!(bvh.all_bounds.unwrap().max, [4.0, 0.0, 0.0]);
}

#[test]
fn mismatched_light_and_bounds_are_rejected() {
    let bounds = build_light_bounds(&point_inputs()).unwrap();
    let records = [LightRecord {
        kind: LightKind::Point,
        payload: 0,
    }];
    assert!(build_light_bvh(&records, &bounds).is_err());
}

#[test]
fn reference_sampling_and_pmf_are_consistent() {
    let inputs = [
        LightBoundInput::Point {
            handle: 0,
            world_position: [-2.0, 0.0, 2.0],
            intensity_max: 1.0,
            scale: 1.0,
        },
        LightBoundInput::Point {
            handle: 1,
            world_position: [2.0, 0.0, 2.0],
            intensity_max: 3.0,
            scale: 1.0,
        },
        LightBoundInput::Point {
            handle: 2,
            world_position: [0.0, 2.0, 2.0],
            intensity_max: 2.0,
            scale: 1.0,
        },
    ];
    let bounds = build_light_bounds(&inputs).unwrap();
    let records = vec![
        LightRecord {
            kind: LightKind::Point,
            payload: 0,
        };
        3
    ];
    let bvh = build_light_bvh(&records, &bounds).unwrap();
    let mut sum = 0.0;
    for handle in 0..3 {
        let pmf = light_bvh_pmf(&bvh, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], handle).unwrap();
        assert!(pmf > 0.0);
        sum += pmf;
    }
    assert!((sum - 1.0).abs() < 1e-6);
    let (handle, pmf) = sample_light_bvh(&bvh, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.5)
        .unwrap()
        .unwrap();
    assert!(pmf > 0.0);
    assert!(
        (light_bvh_pmf(&bvh, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], handle).unwrap() - pmf).abs() < 1e-6
    );
}
