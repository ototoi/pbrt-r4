use super::super::utils::*;
use super::cie_data;
use super::to_string::cie_to_string;
use std::env;
use std::fs;
use std::path::Path;

pub fn build_core(path: &str) {
    let mut contents = String::from("");
    //contents += "use super::super::spectrum;\n";
    contents += "use crate::util::base::Float;\n";
    contents += "\n";

    contents += &format!(
        "pub const CIE_SAMPLES: usize = {};\n",
        cie_data::CIE_SAMPLES
    );
    contents += "\n";
    contents += &cie_to_string("CIE_X", &cie_data::CIE_X);
    contents += "\n";
    contents += &cie_to_string("CIE_Y", &cie_data::CIE_Y);
    contents += "\n";
    contents += &cie_to_string("CIE_Z", &cie_data::CIE_Z);

    contents += "\n";
    contents += &cie_to_string("CIE_LAMBDA", &cie_data::CIE_LAMBDA);

    contents += "\n";
    contents += &format!(
        "pub const CIE_Y_INTEGRAL: Float = {};\n",
        cie_data::CIE_Y_INTEGRAL
    );

    fs::write(path, &contents).unwrap();
}

pub fn build() {
    println!("cargo:rerun-if-changed=build/spectrum/cie_data.rs");
    println!("cargo:rerun-if-changed=build/spectrum/build_cie.rs");
    let depends = ["build/spectrum/cie_data.rs", "build/spectrum/build_cie.rs"];
    let target = "spectrum_data_cie.rs";
    let out_dir = env::var("OUT_DIR").unwrap();
    let path = Path::new(&out_dir).join(target);
    let path = String::from(path.to_str().unwrap());
    let depends = make_depends_path(&depends);
    if should_build(&path, &depends) {
        build_core(&path);
    }
}
