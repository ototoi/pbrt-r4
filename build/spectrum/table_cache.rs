//! Validity tracking for the generated RGB→Spectrum tables.
//!
//! The tables are shared through `OUT_DIR`, so size-only checks cannot notice a
//! stale or corrupted table.  A fingerprint of every input plus a content hash
//! per output is written as a manifest after all tables have been installed.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Bump this when the binary layout of the generated tables changes.
pub const OUTPUT_FORMAT_VERSION: u32 = 1;

const MANIFEST_NAME: &str = "rgb_to_spectrum.manifest";
const MANIFEST_VERSION: &str = "v1";

/// FNV-1a: stable across compilers and standard library versions.
fn hash_bytes(bytes: &[u8], hash: u64) -> u64 {
    let mut hash = hash;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn hash_u64(value: u64, hash: u64) -> u64 {
    hash_bytes(&value.to_le_bytes(), hash)
}

fn hash_token(token: &str, hash: u64) -> u64 {
    let hash = hash_bytes(token.as_bytes(), hash);
    hash_bytes(&[0], hash)
}

/// Fingerprint of all inputs that affect the generated tables.
///
/// `inputs` is a list of (label, contents) pairs; callers pass the generator,
/// optimizer, data and meta sources.  Labels and lengths are hashed so that
/// concatenations cannot collide.
pub fn fingerprint(
    inputs: &[(&str, &[u8])],
    resolution: usize,
    gamuts: &[&str],
    format_version: u32,
) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    for (label, contents) in inputs {
        hash = hash_token(label, hash);
        hash = hash_u64(contents.len() as u64, hash);
        hash = hash_bytes(contents, hash);
    }
    hash = hash_u64(resolution as u64, hash);
    for gamut in gamuts {
        hash = hash_token(gamut, hash);
    }
    hash_u64(format_version as u64, hash)
}

pub fn content_hash(bytes: &[u8]) -> u64 {
    hash_bytes(bytes, 0xcbf29ce484222325)
}

pub fn manifest_path(out_dir: &Path) -> PathBuf {
    out_dir.join(MANIFEST_NAME)
}

#[derive(Clone, Debug)]
pub struct OutputRecord {
    pub name: String,
    pub len: u64,
    pub hash: u64,
}

/// Write the manifest last, after every table has been installed.  A crash
/// before this point leaves either the old manifest or no manifest at all,
/// never a manifest that endorses half-written tables.
pub fn write_manifest(
    out_dir: &Path,
    fingerprint: u64,
    outputs: &[OutputRecord],
) -> Result<(), String> {
    let mut text = format!("{}\nfingerprint {:016x}\n", MANIFEST_VERSION, fingerprint);
    for output in outputs {
        text.push_str(&format!(
            "output {} {} {:016x}\n",
            output.name, output.len, output.hash
        ));
    }
    let tmp = out_dir.join(format!("{}.tmp", MANIFEST_NAME));
    fs::write(&tmp, text).map_err(|error| format!("write {:?}: {}", tmp, error))?;
    fs::rename(&tmp, manifest_path(out_dir))
        .map_err(|error| format!("install {:?}: {}", manifest_path(out_dir), error))
}

fn parse_manifest(bytes: &[u8]) -> Option<(u64, BTreeMap<String, (u64, u64)>)> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut lines = text.lines();
    if lines.next()? != MANIFEST_VERSION {
        return None;
    }
    let fingerprint_line = lines.next()?;
    let mut fields = fingerprint_line.split_whitespace();
    if fields.next()? != "fingerprint" {
        return None;
    }
    let fingerprint = u64::from_str_radix(fields.next()?, 16).ok()?;
    if fields.next().is_some() {
        return None;
    }
    let mut outputs = BTreeMap::new();
    for line in lines {
        let mut fields = line.split_whitespace();
        if fields.next()? != "output" {
            return None;
        }
        let name = fields.next()?.to_string();
        let len = fields.next()?.parse::<u64>().ok()?;
        let hash = u64::from_str_radix(fields.next()?, 16).ok()?;
        if fields.next().is_some() || outputs.insert(name, (len, hash)).is_some() {
            return None;
        }
    }
    Some((fingerprint, outputs))
}

/// True only when the manifest fingerprint matches and every expected table
/// is present with exactly the recorded size and content hash.
pub fn tables_valid(out_dir: &Path, fingerprint: u64, names: &[&str]) -> bool {
    let Ok(bytes) = fs::read(manifest_path(out_dir)) else {
        return false;
    };
    let Some((recorded_fingerprint, outputs)) = parse_manifest(&bytes) else {
        return false;
    };
    if recorded_fingerprint != fingerprint || outputs.len() != names.len() {
        return false;
    }
    for name in names {
        let Some(&(len, hash)) = outputs.get(*name) else {
            return false;
        };
        let Ok(bytes) = fs::read(out_dir.join(name)) else {
            return false;
        };
        if bytes.len() as u64 != len || content_hash(&bytes) != hash {
            return false;
        }
    }
    true
}
