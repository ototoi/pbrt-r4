use super::config::*;

pub fn to_string(name: &str, samples: &[Float], samples_name: &str) -> String {
    let mut contents = String::from("");

    let l = samples.len();
    contents += "#[rustfmt::skip]\n";
    contents += &format!("pub const {}: [Float; {}] = [\n", name, samples_name);
    for i in 0..l {
        let mut indent = String::from("");
        if (i % 10) == 0 {
            indent += "    ";
        } else {
            indent += " ";
        }
        contents += &indent;
        contents += &format!("{:9.8}", samples[i]);
        if i == (l - 1) {
            contents += "\n";
        } else if (i % 10) == 9 {
            contents += ",\n";
        } else {
            contents += ",";
        }
    }
    contents += "];\n";

    return contents;
}

pub fn cie_to_string(name: &str, samples: &[Float]) -> String {
    return to_string(name, samples, "CIE_SAMPLES");
}

pub fn spectrum_to_string(name: &str, samples: &[Float]) -> String {
    return to_string(name, samples, "N_SPECTRUM_SAMPLES");
}
