use crate::gpu::flat;
use crate::util::error::PbrtError;

use super::super::abi::{row_major_to_columns, MediumRecord};

pub fn convert_media(
    media: &[flat::Medium],
    spectrum_attribute_count: usize,
) -> Result<Vec<MediumRecord>, PbrtError> {
    media
        .iter()
        .enumerate()
        .map(|(index, medium)| {
            if medium.sigma_a as usize >= spectrum_attribute_count
                || medium.sigma_s as usize >= spectrum_attribute_count
                || medium.le as usize >= spectrum_attribute_count
            {
                return Err(PbrtError::error(&format!(
                    "Flat medium {index} references an invalid spectrum."
                )));
            }
            if medium.kind != "homogeneous" {
                return Err(PbrtError::error(&format!(
                    "Flat medium \"{}\" has unsupported type \"{}\".",
                    medium.name, medium.kind
                )));
            }
            Ok(MediumRecord {
                kind: 0,
                sigma_a: medium.sigma_a,
                sigma_s: medium.sigma_s,
                le: medium.le,
                g: medium.g,
                padding: [0; 3],
                medium_to_world: row_major_to_columns(medium.transform),
            })
        })
        .collect()
}
