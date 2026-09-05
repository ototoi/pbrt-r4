use pbrt_r4::gpu::ir::flat::{
    build_light_bounds, build_light_bvh, LightBoundInput, LightKind, LightRecord,
};
use pbrt_r4::gpu::webgpu::light_bvh::pack_light_bvh;

fn records(count: usize) -> Vec<LightRecord> {
    (0..count)
        .map(|index| LightRecord {
            kind: LightKind::Point,
            payload: index as u32,
        })
        .collect()
}

#[test]
fn empty_bvh_has_no_packed_region() {
    let bvh = build_light_bvh(&[], &[]).unwrap();
    assert_eq!(pack_light_bvh(&bvh).unwrap(), None);
}

#[test]
fn packed_bvh_contains_header_nodes_handles_and_parents() {
    let inputs = [
        LightBoundInput::Point {
            handle: 0,
            world_position: [-1.0, 0.0, 0.0],
            intensity_max: 1.0,
            scale: 1.0,
        },
        LightBoundInput::Point {
            handle: 1,
            world_position: [1.0, 0.0, 0.0],
            intensity_max: 2.0,
            scale: 1.0,
        },
    ];
    let bounds = build_light_bounds(&inputs).unwrap();
    let bvh = build_light_bvh(&records(2), &bounds).unwrap();
    let packed = pack_light_bvh(&bvh).unwrap().unwrap();

    assert_eq!(packed.header_words[0], (-1.0f32).to_bits());
    assert_eq!(packed.header_words[4], 1.0f32.to_bits());
    assert_eq!(packed.node_words.len(), 3);
    assert_eq!(packed.handle_to_leaf, bvh.handle_to_leaf);
    assert_eq!(packed.node_words[0][7], u32::MAX);
    assert_eq!(packed.node_words[1][7], 0);
    assert_eq!(packed.node_words[2][7], 0);
    assert_eq!(packed.node_words[0][6] >> 31, 0);
    assert_eq!(packed.node_words[1][6] >> 31, 1);
    assert_eq!(packed.node_words[2][6] >> 31, 1);
}
