use super::super::utils::*;
use super::rgb2spec_opt::{self, Gamut};
use std::env;
use std::fs;
use std::path::Path;

pub fn build() {
    println!("cargo:rerun-if-changed=build/spectrum/build_rgb_to_spectrum.rs");
    println!("cargo:rerun-if-changed=build/spectrum/rgb2spec_opt.rs");
    println!("cargo:rerun-if-changed=build/spectrum/rgb2spec_data.rs");
    println!("cargo:rerun-if-changed=build/spectrum/assets/rgb_to_spectrum_meta.rs");

    let out_dir = env::var("OUT_DIR").unwrap();

    let meta_src = Path::new("build/spectrum/assets/rgb_to_spectrum_meta.rs");
    let meta_dst = Path::new(&out_dir).join("rgb_to_spectrum_meta.rs");
    let _ = copy_if_modified(meta_src.to_str().unwrap(), meta_dst.to_str().unwrap());

    const RESOLUTION: usize = 64;
    let specifications = [
        ("srgb", Gamut::Srgb),
        ("aces", Gamut::Aces2065_1),
        ("dci_p3", Gamut::DciP3),
        ("rec2020", Gamut::Rec2020),
    ];

    let generated = std::thread::scope(|scope| {
        specifications
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

    for (name, table) in generated {
        let dst = Path::new(&out_dir).join(format!("rgb_to_spectrum_{}.bin", name));
        let tmp = dst.with_extension("bin.tmp");
        rgb2spec_opt::write_table(&tmp, &table);
        let expected_bytes = (RESOLUTION + 3 * RESOLUTION * RESOLUTION * RESOLUTION * 3)
            * std::mem::size_of::<f32>();
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
    }
}
