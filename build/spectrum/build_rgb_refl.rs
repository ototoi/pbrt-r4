use super::super::utils::*;
use super::config::*;
use super::rgb_data;
use super::to_string::spectrum_to_string;
use super::utils::sample_spectrum;
use std::env;
use std::fs;
use std::path::Path;

pub fn build_core(path: &str) {
    let mut m: Vec<(&str, [Float; N_SPECTRUM_SAMPLES])> = Vec::new();
    m.push((
        "RGBREFL2SPECT_WHITE",
        sample_spectrum(&rgb_data::RGB2SPECT_LAMBDA, &rgb_data::RGBREFL2SPECT_WHITE),
    ));

    m.push((
        "RGBREFL2SPECT_CYAN",
        sample_spectrum(&rgb_data::RGB2SPECT_LAMBDA, &rgb_data::RGBREFL2SPECT_CYAN),
    ));

    m.push((
        "RGBREFL2SPECT_MAGENTA",
        sample_spectrum(
            &rgb_data::RGB2SPECT_LAMBDA,
            &rgb_data::RGBREFL2SPECT_MAGENTA,
        ),
    ));

    m.push((
        "RGBREFL2SPECT_YELLOW",
        sample_spectrum(&rgb_data::RGB2SPECT_LAMBDA, &rgb_data::RGBREFL2SPECT_YELLOW),
    ));

    m.push((
        "RGBREFL2SPECT_RED",
        sample_spectrum(&rgb_data::RGB2SPECT_LAMBDA, &rgb_data::RGBREFL2SPECT_RED),
    ));

    m.push((
        "RGBREFL2SPECT_GREEN",
        sample_spectrum(&rgb_data::RGB2SPECT_LAMBDA, &rgb_data::RGBREFL2SPECT_GREEN),
    ));

    m.push((
        "RGBREFL2SPECT_BLUE",
        sample_spectrum(&rgb_data::RGB2SPECT_LAMBDA, &rgb_data::RGBREFL2SPECT_BLUE),
    ));

    //-----

    m.push((
        "RGBILLUM2SPECT_WHITE",
        sample_spectrum(&rgb_data::RGB2SPECT_LAMBDA, &rgb_data::RGBILLUM2SPECT_WHITE),
    ));

    m.push((
        "RGBILLUM2SPECT_CYAN",
        sample_spectrum(&rgb_data::RGB2SPECT_LAMBDA, &rgb_data::RGBILLUM2SPECT_CYAN),
    ));

    m.push((
        "RGBILLUM2SPECT_MAGENTA",
        sample_spectrum(
            &rgb_data::RGB2SPECT_LAMBDA,
            &rgb_data::RGBILLUM2SPECT_MAGENTA,
        ),
    ));

    m.push((
        "RGBILLUM2SPECT_YELLOW",
        sample_spectrum(
            &rgb_data::RGB2SPECT_LAMBDA,
            &rgb_data::RGBILLUM2SPECT_YELLOW,
        ),
    ));

    m.push((
        "RGBILLUM2SPECT_RED",
        sample_spectrum(&rgb_data::RGB2SPECT_LAMBDA, &rgb_data::RGBILLUM2SPECT_RED),
    ));

    m.push((
        "RGBILLUM2SPECT_GREEN",
        sample_spectrum(&rgb_data::RGB2SPECT_LAMBDA, &rgb_data::RGBILLUM2SPECT_GREEN),
    ));

    m.push((
        "RGBILLUM2SPECT_BLUE",
        sample_spectrum(&rgb_data::RGB2SPECT_LAMBDA, &rgb_data::RGBILLUM2SPECT_BLUE),
    ));

    let mut contents = String::from("");
    contents += "use crate::util::base::Float;\n";
    contents += "\n";

    contents += &format!(
        "const N_SPECTRUM_SAMPLES: usize = {};\n",
        N_SPECTRUM_SAMPLES
    );
    contents += "\n";
    for (key, v) in m.iter() {
        let name = format!("ARRAY_{}", *key);
        contents += &spectrum_to_string(&name, v);
        contents += "\n";
    }

    fs::write(path, &contents).unwrap();
}

pub fn build() {
    for path in [
        "build/spectrum/rgb_data.rs",
        "build/spectrum/build_rgb_refl.rs",
        "build/spectrum/utils.rs",
        "build/spectrum/config.rs",
        "build/spectrum/spectrum_config.rs",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    let depends = [
        "build/spectrum/rgb_data.rs",
        "build/spectrum/build_rgb_refl.rs",
        "build/spectrum/utils.rs",
        "build/spectrum/config.rs",
        "build/spectrum/spectrum_config.rs",
    ];
    let target = "spectrum_data_rgb_refl.rs";
    let out_dir = env::var("OUT_DIR").unwrap();
    let path = Path::new(&out_dir).join(target);
    let path = String::from(path.to_str().unwrap());
    let depends = make_depends_path(&depends);
    if should_build(&path, &depends) {
        build_core(&path);
    }
}
