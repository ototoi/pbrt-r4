#![cfg(feature = "webgpu")]
#[path = "support/nanovdb_fixture.rs"]
mod fixture;
use fixture::*;
use pbrt_r4::gpu::flat::flatten::nanovdb::{
    build_majorant, create_nanovdb, decode_density_grid, world_to_index_f,
};
use pbrt_r4::gpu::flat::sparse_grid::{SparseGridEntry, SparseGridRoot, SparseScalarGrid};
use pbrt_r4::gpu::flat::NanoVDBMedium;
use pbrt_r4::paramdict::ParameterDictionary;

#[test]
fn converts_all_tiles_and_inactive_voxels_exactly() {
    for negative_links in [false, true] {
        let (bytes, _) = full_tree(negative_links);
        let grid = decode_density_grid(&bytes).unwrap().density;
        let mut source = nanovdb_rs::ReadAccessor::<f32>::from_grid_bytes(&bytes).unwrap();
        let mut compare = |coord| {
            assert_eq!(
                grid.get_value(coord).unwrap().to_bits(),
                source.get_value(coord).to_bits(),
                "{coord:?}"
            )
        };
        for x in 0..32 {
            for y in 0..32 {
                for z in 0..32 {
                    compare([x * 128, y * 128, z * 128]);
                }
            }
        }
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    compare([x * 8, y * 8, z * 8]);
                }
            }
        }
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    compare([x, y, z]);
                }
            }
        }
        for coord in [
            [-1, 0, 0],
            [-4096, 0, 0],
            [-4097, 0, 0],
            [4096, 0, 0],
            [i32::MIN, 0, 0],
            [i32::MAX, 0, 0],
        ] {
            compare(coord);
        }
        assert_eq!(grid.get_value([7, 7, 7]).unwrap(), 511.0 / 512.0);
    }
}

#[test]
fn rejects_corrupt_tree_and_scalar_values() {
    let (valid, root) = full_tree(false);
    for (offset, value) in [
        (root + 72, u64::MAX),
        (root + 72, (ROOT - 672) as u64),
        (UPPER + 8256, (UPPER as i64 - UPPER as i64) as u64),
        (LOWER + 1088, i64::MIN as u64),
    ] {
        let mut bytes = valid.clone();
        put64(&mut bytes, offset, value);
        assert!(decode_density_grid(&bytes).is_err());
    }
    for value in [f32::NAN, f32::INFINITY, -1.0] {
        for offset in [
            root + 28,
            root + 116,
            UPPER + 8264,
            LOWER + 1096,
            LEAF + 96 + 511 * 4,
        ] {
            let mut bytes = valid.clone();
            putf(&mut bytes, offset, value);
            assert!(decode_density_grid(&bytes).is_err(), "{offset}");
        }
    }
    for (offset, value) in [(16, 31 << 21), (632, 1), (636, 2)] {
        let mut bytes = valid.clone();
        put32(&mut bytes, offset, value);
        assert!(decode_density_grid(&bytes).is_err());
    }
    let mut duplicate = valid.clone();
    put64(&mut duplicate, root + 96, 0);
    assert!(decode_density_grid(&duplicate).is_err());
    let mut overlap = valid.clone();
    put64(
        &mut overlap,
        LOWER + 1088,
        (UPPER as i64 - LOWER as i64) as u64,
    );
    assert!(decode_density_grid(&overlap).is_err());
    for length in [0, 671, 736, valid.len() - 1] {
        assert!(decode_density_grid(&valid[..length]).is_err());
    }
}

#[test]
fn validates_public_flat_structure() {
    let (bytes, _) = full_tree(false);
    let grid = decode_density_grid(&bytes).unwrap().density;
    let mut bad = grid.clone();
    bad.leaf_values.pop();
    assert!(bad.validate().is_err());
    let mut bad = grid.clone();
    bad.roots[0].entry.child_index = Some(1);
    assert!(bad.validate().is_err());
    assert!(bad.get_value([0, 0, 0]).is_err());
    let mut bad = grid.clone();
    bad.upper_entries[0].value = 1.0;
    assert!(bad.validate().is_err());
    let mut bad = grid.clone();
    bad.lower_entries[0].child_index = Some(u32::MAX);
    assert!(bad.validate().is_err());
    let mut bad = grid.clone();
    bad.roots[1].block[0] = -524289;
    assert!(bad.validate().is_err());
    assert!(SparseScalarGrid::default().validate().is_ok());
    assert_eq!(SparseScalarGrid::default().get_value([0; 3]).unwrap(), 0.0);
    let mut bad = grid;
    bad.lower_entries[0] = SparseGridEntry {
        child_index: Some(0),
        value: f32::NAN,
    };
    assert!(bad.validate().is_err());
}

fn load(
    grids: &[Vec<u8>],
    zip: bool,
    lescale: Option<f32>,
) -> Result<pbrt_r4::gpu::flat::NanoVDBMedium, pbrt_r4::util::error::PbrtError> {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(&mut file, &file_bytes(grids, zip)).unwrap();
    let mut params = ParameterDictionary::new();
    params.add_string("filename", file.path().to_str().unwrap());
    if let Some(value) = lescale {
        params.add_float("Lescale", value);
    }
    create_nanovdb(&params)
}

#[test]
fn loads_zip_and_uncompressed_and_unions_temperature_bounds() {
    let density = root_tile("density", 2.0, ([0; 3], [7; 3]));
    for zip in [false, true] {
        let medium = load(&[density.clone()], zip, None).unwrap();
        assert_eq!(medium.majorant.len(), 64usize.pow(3));
        assert!(medium.majorant.iter().all(|v| *v == 2.0));
        let temperature = root_tile("temperature", 2000.0, ([0; 3], [8; 3]));
        assert!(load(&[density.clone(), temperature.clone()], zip, None).is_err());
        let medium = load(&[density.clone(), temperature], zip, Some(0.0)).unwrap();
        assert_eq!(medium.bounds_max, [8.0; 3]);
        let mut temperature = root_tile("temperature", 1.0, ([0; 3], [8; 3]));
        put32(&mut temperature, 636, 2);
        assert!(load(&[density.clone(), temperature], zip, Some(0.0)).is_err());
    }
    assert!(load(&[root_tile("other", 1.0, ([0; 3], [7; 3]))], false, None).is_err());
    assert!(create_nanovdb(&ParameterDictionary::new()).is_err());
}

#[test]
fn rejects_truncated_file_without_panicking() {
    let bytes = file_bytes(&[root_tile("density", 2.0, ([0; 3], [7; 3]))], false);
    for length in [0, 8, 100, bytes.len() - 1] {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut file, &bytes[..length]).unwrap();
        let mut params = ParameterDictionary::new();
        params.add_string("filename", file.path().to_str().unwrap());
        assert!(create_nanovdb(&params).is_err());
    }
}

#[test]
fn uses_f32_map_slots_and_rejects_rotation() {
    let mut bytes = root_tile("density", 2.0, ([0; 3], [7; 3]));
    putf(&mut bytes, 332, 2.0);
    putf(&mut bytes, 368, 0.25);
    let medium = decode_density_grid(&bytes).unwrap();
    assert_eq!(world_to_index_f(&medium, [0.5, 1.0, 2.0]), [0.5, 1.0, 2.0]);
    putf(&mut bytes, 336, 0.1);
    assert!(decode_density_grid(&bytes).is_err());
    putf(&mut bytes, 336, 0.0);
    putf(&mut bytes, 332, -1.0);
    assert!(decode_density_grid(&bytes).is_err());
}

#[test]
fn majorant_rejects_density_outside_clipped_bbox() {
    let bytes = root_tile("density", 2.0, ([0; 3], [7; 3]));
    let mut medium = decode_density_grid(&bytes).unwrap();
    medium.bounds_max = [16.0; 3];
    assert!(build_majorant(&medium).is_err());
    medium.bounds_max = [7.0; 3];
    medium.inverse_map_f[0][0] = f32::INFINITY;
    assert!(build_majorant(&medium).is_err());
}

#[test]
fn majorant_handles_negative_indices_and_empty_cells() {
    let mut bytes = root_tile("density", 2.0, ([-7; 3], [-1; 3]));
    put64(
        &mut bytes,
        ROOT + 64,
        (0xfffffu64 << 42) | (0xfffffu64 << 21) | 0xfffff,
    );
    let medium = decode_density_grid(&bytes).unwrap();
    assert!(build_majorant(&medium).unwrap().iter().all(|v| *v == 2.0));
    let mut bytes = root_tile("density", 0.0, ([0; 3], [7; 3]));
    put32(&mut bytes, ROOT + 24, 0);
    let mut medium = decode_density_grid(&bytes).unwrap();
    medium.bounds_max = [8192.0; 3];
    assert!(build_majorant(&medium).unwrap().iter().all(|v| *v == 0.0));
}

#[test]
fn majorant_is_x_fastest_and_clips_to_inclusive_bounds() {
    let (bytes, _) = full_tree(false);
    let medium = decode_density_grid(&bytes).unwrap();
    let majorant = build_majorant(&medium).unwrap();

    assert_eq!(majorant[9].to_bits(), (137.0_f32 / 512.0).to_bits());
    assert_eq!(majorant[9 * 64].to_bits(), (81.0_f32 / 512.0).to_bits());
    assert_eq!(majorant[9 * 4096].to_bits(), (74.0_f32 / 512.0).to_bits());
}

#[test]
fn majorant_skips_large_empty_regions_between_sparse_roots() {
    let density = SparseScalarGrid {
        roots: [-100_000, 0, 100_000]
            .into_iter()
            .zip([2.0, 3.0, 4.0])
            .map(|(x, value)| SparseGridRoot {
                block: [x, 0, 0],
                entry: SparseGridEntry {
                    child_index: None,
                    value,
                },
            })
            .collect(),
        ..Default::default()
    };
    let medium = NanoVDBMedium {
        density,
        index_bbox_min: [-409_600_000, 0, 0],
        index_bbox_max: [409_604_095, 1, 1],
        inverse_map_f: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_f: [0.0; 3],
        bounds_min: [-409_600_000.0, 0.0, 0.0],
        bounds_max: [409_604_095.0, 1.0, 1.0],
        majorant_resolution: [64; 3],
        majorant: Vec::new(),
    };

    let majorant = build_majorant(&medium).unwrap();
    assert_eq!(majorant[0], 2.0);
    assert_eq!(majorant[32], 3.0);
    assert_eq!(majorant[63], 4.0);
}

#[test]
fn majorant_includes_background_between_sparse_roots() {
    let density = SparseScalarGrid {
        background: 5.0,
        roots: vec![SparseGridRoot {
            block: [0; 3],
            entry: SparseGridEntry {
                child_index: None,
                value: 2.0,
            },
        }],
        ..Default::default()
    };
    let medium = NanoVDBMedium {
        density,
        index_bbox_min: [0; 3],
        index_bbox_max: [8191, 1, 1],
        inverse_map_f: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_f: [0.0; 3],
        bounds_min: [0.0; 3],
        bounds_max: [8192.0, 2.0, 2.0],
        majorant_resolution: [64; 3],
        majorant: Vec::new(),
    };

    let majorant = build_majorant(&medium).unwrap();
    assert_eq!(majorant[0], 2.0);
    assert_eq!(majorant[63], 5.0);
}

#[test]
#[ignore = "requires local pbrt-v4-scenes assets"]
fn converts_real_cloud_asset() {
    let mut params = ParameterDictionary::new();
    let asset = std::env::var("NANOVDB_ASSET").expect("NANOVDB_ASSET");
    params.add_string("filename", &asset);
    let medium = create_nanovdb(&params).unwrap();
    assert_eq!(medium.majorant.len(), 64usize.pow(3));
    medium.density.validate().unwrap();
    if let Ok(path) = std::env::var("NANOVDB_MAJORANT_OUTPUT") {
        let bytes: Vec<u8> = medium
            .majorant
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path, bytes).unwrap();
    }
    let file = nanovdb_rs::NvdbFile::open(&asset).unwrap();
    let source_grid = file.grids().iter().find(|g| g.name() == "density").unwrap();
    let mut source = source_grid.float_read_accessor().unwrap();
    for x in 0..=8 {
        for y in 0..=8 {
            for z in 0..=8 {
                let coord = [x, y, z].map(|i| i as i64);
                let coord = std::array::from_fn(|axis| {
                    let lo = i64::from(medium.index_bbox_min[axis]);
                    let hi = i64::from(medium.index_bbox_max[axis]);
                    (lo - 1 + coord[axis] * (hi - lo + 2) / 8) as i32
                });
                assert_eq!(
                    medium.density.get_value(coord).unwrap().to_bits(),
                    source.get_value(coord).to_bits()
                );
            }
        }
    }
}

#[test]
#[ignore = "exports fixtures for the local v4 reader check"]
fn export_v4_fixtures() {
    let directory = std::env::var("NANOVDB_FIXTURE_OUTPUT").unwrap();
    for negative in [false, true] {
        for zip in [false, true] {
            let (bytes, _) = full_tree(negative);
            std::fs::write(
                format!("{directory}/fixture-{negative}-{zip}.nvdb"),
                file_bytes(&[bytes], zip),
            )
            .unwrap();
        }
    }
}
