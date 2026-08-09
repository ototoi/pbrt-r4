use super::super::utils::*;
use std::env;
use std::fs;
use std::path::Path;

/*
for (int i = 0; i < WeightLUTSize; ++i) {
            Float alpha = 2;
            Float r2 = Float(i) / Float(WeightLUTSize - 1);
            weightLut[i] = std::exp(-alpha * r2) - std::exp(-alpha);
        }
*/
const WEIGHT_LUT_SIZE: usize = 128;
fn create_mipmap_weight_lut() -> [f32; WEIGHT_LUT_SIZE] {
    let mut mipmap_weight_lut = [0.0; WEIGHT_LUT_SIZE];
    for i in 0..WEIGHT_LUT_SIZE {
        let alpha = 2.0;
        let r2: f32 = i as f32 / (WEIGHT_LUT_SIZE - 1) as f32;
        mipmap_weight_lut[i] = (-alpha * r2).exp() - (-alpha).exp();
    }
    mipmap_weight_lut
}

pub fn build_core(path: &str) {
    let mipmap_weight_lut = create_mipmap_weight_lut();
    let mut contents = String::from("");

    contents += "use crate::util::base::Float;\n";
    contents += "\n";
    contents += "#[rustfmt::skip]\n";
    contents += &format!(
        "pub const MIPMAP_WEIGHT_LUT: [Float;{}] = [ \n",
        WEIGHT_LUT_SIZE
    );
    for i in 0..WEIGHT_LUT_SIZE {
        if i % 16 == 0 {
            contents += "\t";
        }
        contents += &format!("{:9.8}, ", mipmap_weight_lut[i]);
        if (i + 1) % 16 == 0 {
            contents += "\n";
        }
    }
    contents += "];\n";
    fs::write(path, contents).unwrap();
}

pub fn build() {
    let depends = vec!["build/mipmap/build_mipmap_weight_lut.rs"];
    let target = "mipmap_weight_lut.rs";
    let out_dir = env::var("OUT_DIR").unwrap();
    let path = Path::new(&out_dir).join(target);
    let path = String::from(path.to_str().unwrap());
    let depends = make_depends_path(&depends);
    if should_build(&path, &depends) {
        build_core(&path);
    }
}
