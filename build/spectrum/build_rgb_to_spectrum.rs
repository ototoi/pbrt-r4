use super::super::utils::*;
use std::env;
use std::path::Path;

pub fn build() {
    println!("cargo:rerun-if-changed=build/spectrum/build_rgb_to_spectrum.rs");
    println!("cargo:rerun-if-changed=build/spectrum/assets/rgb_to_spectrum_meta.rs");
    println!("cargo:rerun-if-changed=build/spectrum/assets/rgb_to_spectrum_srgb.bin");
    println!("cargo:rerun-if-changed=build/spectrum/assets/rgb_to_spectrum_aces.bin");
    println!("cargo:rerun-if-changed=build/spectrum/assets/rgb_to_spectrum_dci_p3.bin");
    println!("cargo:rerun-if-changed=build/spectrum/assets/rgb_to_spectrum_rec2020.bin");

    let out_dir = env::var("OUT_DIR").unwrap();

    let meta_src = Path::new("build/spectrum/assets/rgb_to_spectrum_meta.rs");
    let meta_dst = Path::new(&out_dir).join("rgb_to_spectrum_meta.rs");
    let _ = copy_if_modified(meta_src.to_str().unwrap(), meta_dst.to_str().unwrap());

    // Per-colour-space RGB→spectrum tables. All four are extracted
    // from pbrt-v4's `build/rgbspectrum_<cs>.cpp` (Scale[64] +
    // Data[3][64][64][64][3]) with each colour space's bundled
    // illuminant samples (D65 for sRGB/DCI-P3/Rec2020, D60 for ACES).
    for cs in ["srgb", "aces", "dci_p3", "rec2020"] {
        let src = format!("build/spectrum/assets/rgb_to_spectrum_{}.bin", cs);
        let dst = Path::new(&out_dir).join(format!("rgb_to_spectrum_{}.bin", cs));
        let _ = copy_if_modified(src.as_str(), dst.to_str().unwrap());
    }
}
