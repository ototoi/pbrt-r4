use super::super::utils::*;
use super::rgb2spec_opt::{self, Gamut};
use super::table_cache::{self, OutputRecord};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const RESOLUTION: usize = 64;
const SPECIFICATIONS: [(&str, Gamut); 4] = [
    ("srgb", Gamut::Srgb),
    ("aces", Gamut::Aces2065_1),
    ("dci_p3", Gamut::DciP3),
    ("rec2020", Gamut::Rec2020),
];

const GENERATOR_SOURCES: [&str; 5] = [
    "build/spectrum/build_rgb_to_spectrum.rs",
    "build/spectrum/table_cache.rs",
    "build/spectrum/rgb2spec_opt.rs",
    "build/spectrum/rgb2spec_data.rs",
    "build/spectrum/assets/rgb_to_spectrum_meta.rs",
];

pub fn build() {
    for source in GENERATOR_SOURCES {
        println!("cargo:rerun-if-changed={}", source);
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let meta_src = Path::new("build/spectrum/assets/rgb_to_spectrum_meta.rs");
    let meta_dst = out_dir.join("rgb_to_spectrum_meta.rs");
    let _ = copy_if_modified(meta_src.to_str().unwrap(), meta_dst.to_str().unwrap());

    let inputs: Vec<(&str, Vec<u8>)> = GENERATOR_SOURCES
        .iter()
        .map(|path| {
            let contents = fs::read(path)
                .unwrap_or_else(|error| panic!("rgb2spec_opt: read input {:?}: {}", path, error));
            (*path, contents)
        })
        .collect();
    let input_refs: Vec<(&str, &[u8])> = inputs
        .iter()
        .map(|(path, bytes)| (*path, bytes.as_slice()))
        .collect();
    let gamut_names: Vec<&str> = SPECIFICATIONS.iter().map(|(name, _)| *name).collect();
    let fingerprint = table_cache::fingerprint(
        &input_refs,
        RESOLUTION,
        &gamut_names,
        table_cache::OUTPUT_FORMAT_VERSION,
    );

    let output_names: Vec<String> = SPECIFICATIONS
        .iter()
        .map(|(name, _)| format!("rgb_to_spectrum_{}.bin", name))
        .collect();
    let output_refs: Vec<&str> = output_names.iter().map(String::as_str).collect();
    // A changed table or manifest is itself a reason to re-run the build
    // script: external corruption must not be treated as a fresh artifact.
    for name in &output_names {
        println!("cargo:rerun-if-changed={}", out_dir.join(name).display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        table_cache::manifest_path(&out_dir).display()
    );

    if table_cache::tables_valid(&out_dir, fingerprint, &output_refs) {
        return;
    }

    let expected_bytes =
        (RESOLUTION + 3 * RESOLUTION * RESOLUTION * RESOLUTION * 3) * std::mem::size_of::<f32>();
    let generated = std::thread::scope(|scope| {
        SPECIFICATIONS
            .map(|(name, gamut)| {
                let handle = scope.spawn(move || {
                    let tables = rgb2spec_opt::Tables::new(gamut);
                    (name, rgb2spec_opt::generate_table(&tables, RESOLUTION))
                });
                (name, handle)
            })
            .map(|(name, handle)| {
                (
                    name,
                    handle
                        .join()
                        .unwrap_or_else(|_| panic!("rgb2spec_opt: generation thread panicked"))
                        .1,
                )
            })
    });

    let mut outputs = Vec::new();
    for (name, table) in generated {
        let dst = out_dir.join(format!("rgb_to_spectrum_{}.bin", name));
        let tmp = dst.with_extension("bin.tmp");
        rgb2spec_opt::write_table(&tmp, &table);
        let actual_bytes = fs::metadata(&tmp)
            .unwrap_or_else(|error| panic!("rgb2spec_opt: inspect {:?}: {}", tmp, error))
            .len() as usize;
        assert_eq!(
            actual_bytes, expected_bytes,
            "rgb2spec_opt: unexpected table size for {}",
            name
        );
        fs::rename(&tmp, &dst)
            .unwrap_or_else(|error| panic!("rgb2spec_opt: install {:?}: {}", dst, error));
        let bytes = fs::read(&dst)
            .unwrap_or_else(|error| panic!("rgb2spec_opt: read {:?}: {}", dst, error));
        outputs.push(OutputRecord {
            name: format!("rgb_to_spectrum_{}.bin", name),
            len: bytes.len() as u64,
            hash: table_cache::content_hash(&bytes),
        });
    }

    table_cache::write_manifest(&out_dir, fingerprint, &outputs)
        .unwrap_or_else(|error| panic!("rgb2spec_opt: write manifest: {}", error));
}
