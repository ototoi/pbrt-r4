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

fn unpack_bounds(words: &[u32; 8], all_min: [f32; 3], all_max: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let q_min = [words[0] & 0xffff, words[0] >> 16, words[1] & 0xffff];
    let q_max = [words[1] >> 16, words[2] & 0xffff, words[2] >> 16];
    let decode = |axis: usize, value: u32| {
        if all_min[axis] == all_max[axis] {
            all_min[axis]
        } else {
            all_min[axis] + (all_max[axis] - all_min[axis]) * value as f32 / u16::MAX as f32
        }
    };
    (
        [
            decode(0, q_min[0]),
            decode(1, q_min[1]),
            decode(2, q_min[2]),
        ],
        [
            decode(0, q_max[0]),
            decode(1, q_max[1]),
            decode(2, q_max[2]),
        ],
    )
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

#[test]
fn packed_bounds_contain_flat_bounds_after_quantization() {
    let inputs = [
        LightBoundInput::Point {
            handle: 0,
            world_position: [-1.25, 2.0, 0.0],
            intensity_max: 1.0,
            scale: 1.0,
        },
        LightBoundInput::Point {
            handle: 1,
            world_position: [2.75, 2.0, 3.5],
            intensity_max: 2.0,
            scale: 1.0,
        },
        LightBoundInput::Point {
            handle: 2,
            world_position: [0.5, 2.0, 1.25],
            intensity_max: 3.0,
            scale: 1.0,
        },
    ];
    let bounds = build_light_bounds(&inputs).unwrap();
    let bvh = build_light_bvh(&records(3), &bounds).unwrap();
    let packed = pack_light_bvh(&bvh).unwrap().unwrap();
    let all_min = [
        f32::from_bits(packed.header_words[0]),
        f32::from_bits(packed.header_words[1]),
        f32::from_bits(packed.header_words[2]),
    ];
    let all_max = [
        f32::from_bits(packed.header_words[4]),
        f32::from_bits(packed.header_words[5]),
        f32::from_bits(packed.header_words[6]),
    ];

    for (flat_node, packed_node) in bvh.nodes.iter().zip(&packed.node_words) {
        let (decoded_min, decoded_max) = unpack_bounds(packed_node, all_min, all_max);
        let original = flat_node.bounds();
        for axis in 0..3 {
            let epsilon = 1e-5 * (1.0 + original.bounds.min[axis].abs());
            assert!(decoded_min[axis] <= original.bounds.min[axis] + epsilon);
            assert!(decoded_max[axis] + epsilon >= original.bounds.max[axis]);
        }
    }
}
