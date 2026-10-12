use crate::util::error::PbrtError;
use std::collections::HashSet;

pub const SPARSE_UPPER_SIZE: usize = 32 * 32 * 32;
pub const SPARSE_LOWER_SIZE: usize = 16 * 16 * 16;
pub const SPARSE_LEAF_SIZE: usize = 8 * 8 * 8;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SparseGridEntry {
    pub child_index: Option<u32>,
    pub value: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseGridRoot {
    pub block: [i32; 3],
    pub entry: SparseGridEntry,
}

/// Scalar tree with z varying fastest within each node and leaf.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SparseScalarGrid {
    pub background: f32,
    pub roots: Vec<SparseGridRoot>,
    pub upper_entries: Vec<SparseGridEntry>,
    pub lower_entries: Vec<SparseGridEntry>,
    pub leaf_values: Vec<f32>,
}

impl SparseScalarGrid {
    pub fn validate(&self) -> Result<(), PbrtError> {
        for (length, size) in [
            (self.upper_entries.len(), SPARSE_UPPER_SIZE),
            (self.lower_entries.len(), SPARSE_LOWER_SIZE),
            (self.leaf_values.len(), SPARSE_LEAF_SIZE),
        ] {
            if length % size != 0 || u32::try_from(length).is_err() {
                return Err(PbrtError::error("Invalid sparse grid array length."));
            }
        }
        if u32::try_from(self.roots.len()).is_err() {
            return Err(PbrtError::error("Sparse grid root count exceeds u32."));
        }
        let mut blocks = HashSet::new();
        for root in &self.roots {
            if root.block.iter().any(|v| !(-524288..=524287).contains(v))
                || !blocks.insert(root.block)
            {
                return Err(PbrtError::error(
                    "Invalid or duplicate sparse grid root block.",
                ));
            }
            validate_entry(root.entry, self.upper_entries.len() / SPARSE_UPPER_SIZE)?;
        }
        for entry in &self.upper_entries {
            validate_entry(*entry, self.lower_entries.len() / SPARSE_LOWER_SIZE)?;
        }
        for entry in &self.lower_entries {
            validate_entry(*entry, self.leaf_values.len() / SPARSE_LEAF_SIZE)?;
        }
        for value in std::iter::once(&self.background).chain(&self.leaf_values) {
            validate_value(*value)?;
        }
        Ok(())
    }

    pub fn get_value(&self, coordinate: [i32; 3]) -> Result<f32, PbrtError> {
        let block = coordinate.map(|v| v >> 12);
        let Some(root) = self.roots.iter().find(|root| root.block == block) else {
            return Ok(self.background);
        };
        let Some(upper) = root.entry.child_index else {
            return Ok(root.entry.value);
        };
        let upper = entry_at(
            &self.upper_entries,
            upper,
            SPARSE_UPPER_SIZE,
            local_index(coordinate, 7, 31, 32),
        )?;
        let Some(lower) = upper.child_index else {
            return Ok(upper.value);
        };
        let lower = entry_at(
            &self.lower_entries,
            lower,
            SPARSE_LOWER_SIZE,
            local_index(coordinate, 3, 15, 16),
        )?;
        let Some(leaf) = lower.child_index else {
            return Ok(lower.value);
        };
        let start = (leaf as usize)
            .checked_mul(SPARSE_LEAF_SIZE)
            .ok_or_else(|| PbrtError::error("Sparse leaf offset overflow."))?;
        let end = start
            .checked_add(SPARSE_LEAF_SIZE)
            .ok_or_else(|| PbrtError::error("Sparse leaf range overflow."))?;
        self.leaf_values
            .get(start..end)
            .map(|values| values[local_index(coordinate, 0, 7, 8)])
            .ok_or_else(|| PbrtError::error("Invalid sparse leaf reference."))
    }
}

fn local_index(coordinate: [i32; 3], shift: u32, mask: i32, dimension: usize) -> usize {
    let [x, y, z] = coordinate.map(|v| ((v >> shift) & mask) as usize);
    (x * dimension + y) * dimension + z
}

fn entry_at(
    entries: &[SparseGridEntry],
    node: u32,
    size: usize,
    index: usize,
) -> Result<SparseGridEntry, PbrtError> {
    let start = (node as usize)
        .checked_mul(size)
        .ok_or_else(|| PbrtError::error("Sparse node offset overflow."))?;
    let end = start
        .checked_add(size)
        .ok_or_else(|| PbrtError::error("Sparse node range overflow."))?;
    entries
        .get(start..end)
        .map(|node| node[index])
        .ok_or_else(|| PbrtError::error("Invalid sparse node reference."))
}

fn validate_entry(entry: SparseGridEntry, count: usize) -> Result<(), PbrtError> {
    match entry.child_index {
        Some(index) if index as usize >= count || entry.value != 0.0 => Err(PbrtError::error(
            "Invalid sparse child reference or child value.",
        )),
        Some(_) => Ok(()),
        None => validate_value(entry.value),
    }
}

fn validate_value(value: f32) -> Result<(), PbrtError> {
    if !value.is_finite() || value < 0.0 {
        Err(PbrtError::error(
            "Sparse grid values must be finite and non-negative.",
        ))
    } else {
        Ok(())
    }
}
