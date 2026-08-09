use log::*;

use crate::util::base::*;
use crate::util::error::*;

use std::fs::read_to_string;

pub fn read_float_file(path: &str) -> Result<Vec<Float>, PbrtError> {
    let s = read_to_string(path)
        .map_err(|_| PbrtError::error(&format!("Unable to open file \"{}\".", path)))?;
    let mut values = Vec::new();
    for (i, line) in s.lines().enumerate() {
        let line_number = i + 1;
        let line = line.trim();
        if line.find('#').is_none() {
            let mut vv: Vec<Float> = line
                .split_ascii_whitespace()
                .map(|token| -> f32 {
                    if let Ok(f) = token.parse::<f32>() {
                        return f;
                    } else {
                        warn!(
                            "Unexpected text found at line {} of float file \"{}\"",
                            line_number, path
                        );
                    }
                    return 0.0;
                })
                .map(|f| f as Float)
                .collect();
            values.append(&mut vv);
        }
    }
    return Ok(values);
}
