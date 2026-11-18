#[path = "../build/spectrum/table_cache.rs"]
mod table_cache;

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use table_cache::{fingerprint, manifest_path, tables_valid, write_manifest, OutputRecord};

const INPUT: &[u8] = b"generator source version 1";
const GAMUTS: [&str; 2] = ["srgb", "aces"];
const NAMES: [&str; 2] = ["rgb_to_spectrum_srgb.bin", "rgb_to_spectrum_aces.bin"];

fn temp_dir() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "pbrt-r4-table-cache-{}-{}-{}",
        std::process::id(),
        nanos,
        id
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn make_fingerprint(input: &[u8]) -> u64 {
    fingerprint(
        &[("generator", input)],
        8,
        &GAMUTS,
        table_cache::OUTPUT_FORMAT_VERSION,
    )
}

fn write_outputs(dir: &PathBuf, values: &[&[u8]]) -> Vec<OutputRecord> {
    NAMES
        .iter()
        .zip(values)
        .map(|(name, bytes)| {
            fs::write(dir.join(name), bytes).unwrap();
            OutputRecord {
                name: (*name).to_string(),
                len: bytes.len() as u64,
                hash: table_cache::content_hash(bytes),
            }
        })
        .collect()
}

#[test]
fn valid_tables_are_reused() {
    let dir = temp_dir();
    let fp = make_fingerprint(INPUT);
    let outputs = write_outputs(&dir, &[b"aaaa", b"bbbb"]);
    write_manifest(&dir, fp, &outputs).unwrap();
    assert!(tables_valid(&dir, fp, &NAMES));
}

#[test]
fn input_change_invalidates() {
    let dir = temp_dir();
    let fp = make_fingerprint(INPUT);
    let outputs = write_outputs(&dir, &[b"aaaa", b"bbbb"]);
    write_manifest(&dir, fp, &outputs).unwrap();
    let changed = make_fingerprint(b"generator source version 2");
    assert_ne!(fp, changed);
    assert!(!tables_valid(&dir, changed, &NAMES));
}

#[test]
fn resolution_gamut_and_format_change_fingerprint() {
    let base = fingerprint(&[("generator", INPUT)], 8, &GAMUTS, 1);
    assert_ne!(base, fingerprint(&[("generator", INPUT)], 9, &GAMUTS, 1));
    assert_ne!(
        base,
        fingerprint(&[("generator", INPUT)], 8, &["srgb", "dci_p3"], 1)
    );
    assert_ne!(base, fingerprint(&[("generator", INPUT)], 8, &GAMUTS, 2));
    assert_eq!(base, fingerprint(&[("generator", INPUT)], 8, &GAMUTS, 1));
}

#[test]
fn same_size_corruption_and_missing_table_invalidate() {
    let dir = temp_dir();
    let fp = make_fingerprint(INPUT);
    let outputs = write_outputs(&dir, &[b"aaaa", b"bbbb"]);
    write_manifest(&dir, fp, &outputs).unwrap();

    let first = dir.join(NAMES[0]);
    let mut bytes = fs::read(&first).unwrap();
    bytes[0] ^= 0xff; // identical size, different contents
    fs::write(&first, bytes).unwrap();
    assert!(!tables_valid(&dir, fp, &NAMES));

    fs::write(&first, b"aaaa").unwrap();
    assert!(tables_valid(&dir, fp, &NAMES));
    fs::remove_file(&first).unwrap();
    assert!(!tables_valid(&dir, fp, &NAMES));
}

#[test]
fn missing_or_interrupted_manifest_invalidates() {
    let dir = temp_dir();
    let fp = make_fingerprint(INPUT);
    let outputs = write_outputs(&dir, &[b"aaaa", b"bbbb"]);
    assert!(!tables_valid(&dir, fp, &NAMES)); // no manifest yet

    // Manifest written before all tables are installed must not be accepted.
    write_manifest(&dir, fp, &outputs).unwrap();
    fs::remove_file(dir.join(NAMES[1])).unwrap();
    assert!(!tables_valid(&dir, fp, &NAMES));
}

#[test]
fn stale_manifest_is_replaced_atomically() {
    let dir = temp_dir();
    let fp = make_fingerprint(INPUT);
    let outputs = write_outputs(&dir, &[b"aaaa", b"bbbb"]);
    let old = write_manifest(&dir, fp.wrapping_add(1), &outputs);
    assert!(old.is_ok());
    assert!(!tables_valid(&dir, fp, &NAMES));

    let outputs = write_outputs(&dir, &[b"cccc", b"dddd"]);
    write_manifest(&dir, fp, &outputs).unwrap();
    assert!(tables_valid(&dir, fp, &NAMES));
    assert!(!manifest_path(&dir).with_extension("manifest.tmp").exists());
}
