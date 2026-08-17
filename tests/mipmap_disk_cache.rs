use pbrt_r4::prelude::*;
use pbrt_r4::textures::{
    load_float_mipmap_cache, load_spectrum_mipmap_cache, remove_mipmap_cache,
    save_float_mipmap_cache, save_spectrum_mipmap_cache, MIPMapLevelStorage, MIPMapStorageKind,
};

use std::fs::{self, OpenOptions};
use std::io::Write;

fn texinfo() -> TexInfo {
    TexInfo {
        cache_version: 3,
        filename: "mipmap-disk-cache-test.png".to_string(),
        filter: ImageFilter::Bilinear,
        max_aniso: 8.0,
        swrap_mode: ImageWrap::Clamp,
        twrap_mode: ImageWrap::Clamp,
        scale: 1.0,
        encoding: ColorEncoding::SRgb,
    }
}

#[test]
fn rgba_u8_mipmap_cache_round_trips_storage_and_channels() {
    let mipmap = MIPMap::<Float>::new_with_raw_channels_and_storage(
        &Point2i::new(2, 2),
        &[
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        ],
        4,
        ImageFilter::Bilinear,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
        MIPMapStorageKind::U8 {
            encoding: ColorEncoding::SRgb,
        },
    );
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("cache");
    let info = texinfo();
    save_float_mipmap_cache(cache_dir.to_str().unwrap(), &info, "source-hash", &mipmap).unwrap();

    let restored =
        load_float_mipmap_cache(cache_dir.to_str().unwrap(), "source-hash", &info).unwrap();
    assert_eq!(restored.channel_count(), 4);
    assert!(matches!(
        restored.storage.pyramid[0].data,
        MIPMapLevelStorage::U8 {
            encoding: ColorEncoding::SRgb,
            ..
        }
    ));
    if let (
        MIPMapLevelStorage::U8 { data: restored, .. },
        MIPMapLevelStorage::U8 { data: original, .. },
    ) = (
        &restored.storage.pyramid[0].data,
        &mipmap.storage.pyramid[0].data,
    ) {
        assert_eq!(restored, original);
    } else {
        panic!("expected U8 storage after cache round-trip");
    }
}

#[test]
fn mipmap_cache_rejects_a_different_source_hash() {
    let mipmap = MIPMap::<Float>::new(
        &Point2i::new(1, 1),
        &[0.5],
        ImageFilter::Point,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
    );
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("cache");
    let info = texinfo();
    save_float_mipmap_cache(cache_dir.to_str().unwrap(), &info, "source-hash", &mipmap).unwrap();

    assert!(load_float_mipmap_cache(cache_dir.to_str().unwrap(), "changed", &info).is_err());
}

#[test]
fn f32_and_f16_mipmap_cache_round_trip_the_storage_kind() {
    for storage in [MIPMapStorageKind::F32, MIPMapStorageKind::F16] {
        let mipmap = MIPMap::<Float>::new_with_raw_channels_and_storage(
            &Point2i::new(1, 1),
            &[0.25],
            1,
            ImageFilter::Point,
            8.0,
            ImageWrap::Clamp,
            ImageWrap::Clamp,
            storage,
        );
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("cache");
        let info = texinfo();
        save_float_mipmap_cache(cache_dir.to_str().unwrap(), &info, "hash", &mipmap).unwrap();
        let restored = load_float_mipmap_cache(cache_dir.to_str().unwrap(), "hash", &info).unwrap();
        match (
            &storage,
            &mipmap.storage.pyramid[0].data,
            &restored.storage.pyramid[0].data,
        ) {
            (
                MIPMapStorageKind::F32,
                MIPMapLevelStorage::F32(original),
                MIPMapLevelStorage::F32(restored),
            ) => assert_eq!(restored, original),
            (
                MIPMapStorageKind::F16,
                MIPMapLevelStorage::F16(original),
                MIPMapLevelStorage::F16(restored),
            ) => assert_eq!(
                restored
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                original
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            ),
            _ => panic!("MIPMap storage kind changed during round-trip"),
        }
    }
}

#[test]
fn spectrum_mipmap_cache_round_trips_the_rgb_view() {
    let mipmap = MIPMap::<RGBSpectrum>::new_with_raw_channels(
        &Point2i::new(1, 1),
        &[0.1, 0.2, 0.3],
        3,
        ImageFilter::Bilinear,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
    );
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("cache");
    let info = texinfo();
    save_spectrum_mipmap_cache(cache_dir.to_str().unwrap(), &info, "hash", &mipmap).unwrap();

    let restored = load_spectrum_mipmap_cache(cache_dir.to_str().unwrap(), "hash", &info).unwrap();
    assert_eq!(restored.channel_count(), 3);
    let rgb = restored.storage.pyramid[0].data.to_f32_vec();
    assert_eq!(rgb, vec![0.1, 0.2, 0.3]);
}

#[test]
fn mipmap_cache_rejects_checksum_mismatch_and_incomplete_cache() {
    let mipmap = MIPMap::<Float>::new(
        &Point2i::new(1, 1),
        &[0.5],
        ImageFilter::Point,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
    );
    let info = texinfo();

    let checksum_dir = tempfile::tempdir().unwrap();
    let checksum_cache = checksum_dir.path().join("cache");
    save_float_mipmap_cache(checksum_cache.to_str().unwrap(), &info, "hash", &mipmap).unwrap();
    let mut level = OpenOptions::new()
        .append(true)
        .open(checksum_cache.join("level-000.bin"))
        .unwrap();
    level.write_all(&[0]).unwrap();
    assert!(load_float_mipmap_cache(checksum_cache.to_str().unwrap(), "hash", &info).is_err());

    let incomplete_dir = tempfile::tempdir().unwrap();
    let incomplete_cache = incomplete_dir.path().join("cache");
    save_float_mipmap_cache(incomplete_cache.to_str().unwrap(), &info, "hash", &mipmap).unwrap();
    fs::remove_file(incomplete_cache.join("COMPLETE")).unwrap();
    assert!(load_float_mipmap_cache(incomplete_cache.to_str().unwrap(), "hash", &info).is_err());
}

#[test]
fn mipmap_cache_rejects_the_legacy_exr_layout() {
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("cache");
    fs::create_dir_all(&cache_dir).unwrap();
    fs::write(cache_dir.join("cache_info.json"), "{}").unwrap();
    fs::write(cache_dir.join("0.exr"), []).unwrap();

    assert!(load_float_mipmap_cache(cache_dir.to_str().unwrap(), "hash", &texinfo()).is_err());
}

#[test]
fn mipmap_cache_can_remove_a_corrupt_cache_directory() {
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("cache");
    fs::create_dir_all(cache_dir.join("nested")).unwrap();
    fs::write(cache_dir.join("nested/level.bin"), [0_u8]).unwrap();

    assert!(remove_mipmap_cache(cache_dir.to_str().unwrap()).unwrap());
    assert!(!cache_dir.exists());
    assert!(!remove_mipmap_cache(cache_dir.to_str().unwrap()).unwrap());
}

#[test]
fn mipmap_cache_rejects_a_changed_metadata_encoding() {
    let mipmap = MIPMap::<Float>::new_with_raw_channels_and_storage(
        &Point2i::new(1, 1),
        &[0.5],
        1,
        ImageFilter::Point,
        8.0,
        ImageWrap::Clamp,
        ImageWrap::Clamp,
        MIPMapStorageKind::U8 {
            encoding: ColorEncoding::SRgb,
        },
    );
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir.path().join("cache");
    let info = texinfo();
    save_float_mipmap_cache(cache_dir.to_str().unwrap(), &info, "hash", &mipmap).unwrap();

    let metadata_path = cache_dir.join("metadata.json");
    let metadata_text = fs::read_to_string(&metadata_path).unwrap();
    let mut metadata: serde_json::Value = serde_json::from_str(&metadata_text).unwrap();
    metadata["encoding"] = serde_json::Value::String("Linear".to_string());
    fs::write(metadata_path, serde_json::to_vec_pretty(&metadata).unwrap()).unwrap();

    assert!(load_float_mipmap_cache(cache_dir.to_str().unwrap(), "hash", &info).is_err());
}
