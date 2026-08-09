//! Reader for the Eindhoven RGL Tensor file format (`.bsdf`) used by
//! `MeasuredBxDFData`. Mirrors pbrt-v4's anonymous `Tensor` class in
//! `src/pbrt/bxdfs.cpp` -- same binary layout (little-endian) and the
//! same field map structure, just expressed as Rust.
//!
//! Layout (little-endian):
//!
//! ```text
//! 0: 12 bytes  "tensor_file"
//! 12: 2 bytes   version (major, minor) -- expected (1, 0)
//! 14: 4 bytes   u32 n_fields
//! 18: for each field:
//!     u16 name_length
//!     name_length bytes UTF-8 name
//!     u16 ndim
//!     u8  dtype
//!     u64 offset (file-absolute, where this field's raw data starts)
//!     ndim * u64 shape entries
//! data: raw field bytes at each field's `offset`,
//!       sized type_size(dtype) * prod(shape).
//! ```

use crate::util::error::PbrtError;

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Numeric type of a tensor field. Same numbering as pbrt-v4
/// `Tensor::Type` (Invalid=0, then ints, then floats).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    Invalid = 0,
    UInt8 = 1,
    Int8 = 2,
    UInt16 = 3,
    Int16 = 4,
    UInt32 = 5,
    Int32 = 6,
    UInt64 = 7,
    Int64 = 8,
    Float16 = 9,
    Float32 = 10,
    Float64 = 11,
}

impl TensorType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(TensorType::Invalid),
            1 => Some(TensorType::UInt8),
            2 => Some(TensorType::Int8),
            3 => Some(TensorType::UInt16),
            4 => Some(TensorType::Int16),
            5 => Some(TensorType::UInt32),
            6 => Some(TensorType::Int32),
            7 => Some(TensorType::UInt64),
            8 => Some(TensorType::Int64),
            9 => Some(TensorType::Float16),
            10 => Some(TensorType::Float32),
            11 => Some(TensorType::Float64),
            _ => None,
        }
    }

    pub fn size(self) -> usize {
        match self {
            TensorType::Invalid => 0,
            TensorType::UInt8 | TensorType::Int8 => 1,
            TensorType::UInt16 | TensorType::Int16 | TensorType::Float16 => 2,
            TensorType::UInt32 | TensorType::Int32 | TensorType::Float32 => 4,
            TensorType::UInt64 | TensorType::Int64 | TensorType::Float64 => 8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorField {
    pub dtype: TensorType,
    pub offset: u64,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl TensorField {
    /// Reinterpret `self.data` as a slice of `f32`. Returns `None` if
    /// the field is not Float32 or the byte count is not a multiple of 4.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.dtype != TensorType::Float32 {
            return None;
        }
        if self.data.len() % 4 != 0 {
            return None;
        }
        // SAFETY: We just verified that `data.len()` is a multiple of 4
        // and dtype is Float32. The tensor file's Float32 fields are
        // little-endian; this also matches the layout assumed by
        // PiecewiseLinear2D once we wire it up. Misaligned reads would
        // bite us if we did `*(f32*)ptr`, but `align_to::<f32>` keeps
        // the prefix/suffix lengths zero only when alignment matches --
        // copy out to a Vec if you can't guarantee alignment.
        let (head, body, tail) = unsafe { self.data.align_to::<f32>() };
        if head.is_empty() && tail.is_empty() {
            Some(body)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct TensorFile {
    pub filename: String,
    pub size: u64,
    pub fields: HashMap<String, TensorField>,
}

impl TensorFile {
    /// Open and fully parse a `.bsdf` Tensor file. Returns the parsed
    /// fields plus the total file size. Mirrors pbrt-v4
    /// `Tensor::Tensor(const std::string &filename)`.
    pub fn open(filename: &str) -> Result<Self, PbrtError> {
        let mut file = File::open(filename)
            .map_err(|e| PbrtError::error(&format!("{}: open failed: {}", filename, e)))?;
        let size = file
            .seek(SeekFrom::End(0))
            .map_err(|e| PbrtError::error(&format!("{}: seek failed: {}", filename, e)))?;
        file.rewind()
            .map_err(|e| PbrtError::error(&format!("{}: rewind failed: {}", filename, e)))?;

        if size < (12 + 2 + 4) {
            return Err(PbrtError::error(&format!(
                "{}: Invalid tensor file: too small, truncated?",
                filename
            )));
        }

        let mut header = [0u8; 12];
        let mut version = [0u8; 2];
        let mut n_fields_bytes = [0u8; 4];
        read_exact(&mut file, &mut header, filename, "header")?;
        read_exact(&mut file, &mut version, filename, "version")?;
        read_exact(&mut file, &mut n_fields_bytes, filename, "n_fields")?;
        let n_fields = u32::from_le_bytes(n_fields_bytes);

        // pbrt-v4 reads 12 raw bytes and memcmps against "tensor_file";
        // "tensor_file" is 11 bytes plus a trailing NUL the writer emits.
        if &header[..11] != b"tensor_file" {
            return Err(PbrtError::error(&format!(
                "{}: Invalid tensor file: invalid header.",
                filename
            )));
        }
        if version != [1u8, 0u8] {
            return Err(PbrtError::error(&format!(
                "{}: Invalid tensor file: unknown file version ({}.{}).",
                filename, version[0], version[1]
            )));
        }

        let mut fields = HashMap::with_capacity(n_fields as usize);

        for _ in 0..n_fields {
            let name_length = read_u16(&mut file, filename, "name_length")? as usize;
            let mut name_bytes = vec![0u8; name_length];
            read_exact(&mut file, &mut name_bytes, filename, "name")?;
            let name = String::from_utf8(name_bytes).map_err(|_| {
                PbrtError::error(&format!("{}: tensor field name is not UTF-8", filename))
            })?;

            let ndim = read_u16(&mut file, filename, "ndim")? as usize;
            let dtype_byte = read_u8(&mut file, filename, "dtype")?;
            let dtype = TensorType::from_u8(dtype_byte).ok_or_else(|| {
                PbrtError::error(&format!(
                    "{}: Invalid tensor file: unknown type {}.",
                    filename, dtype_byte
                ))
            })?;
            if dtype == TensorType::Invalid {
                return Err(PbrtError::error(&format!(
                    "{}: Invalid tensor file: invalid type for field {:?}.",
                    filename, name
                )));
            }
            let offset = read_u64(&mut file, filename, "offset")?;

            let mut shape = Vec::with_capacity(ndim);
            let mut total_elems: usize = 1;
            for _ in 0..ndim {
                let dim = read_u64(&mut file, filename, "shape entry")? as usize;
                total_elems = total_elems.saturating_mul(dim);
                shape.push(dim);
            }
            let total_bytes = total_elems.saturating_mul(dtype.size());

            // Read raw data at `offset` without disturbing the field-list
            // cursor we're walking through.
            let resume_pos = file
                .stream_position()
                .map_err(|e| PbrtError::error(&format!("{}: stream_position: {}", filename, e)))?;
            file.seek(SeekFrom::Start(offset)).map_err(|e| {
                PbrtError::error(&format!("{}: seek to data {}: {}", filename, offset, e))
            })?;
            let mut data = vec![0u8; total_bytes];
            read_exact(&mut file, &mut data, filename, "field data")?;
            file.seek(SeekFrom::Start(resume_pos))
                .map_err(|e| PbrtError::error(&format!("{}: seek back: {}", filename, e)))?;

            fields.insert(
                name,
                TensorField {
                    dtype,
                    offset,
                    shape,
                    data,
                },
            );
        }

        Ok(Self {
            filename: filename.to_string(),
            size,
            fields,
        })
    }

    pub fn field(&self, name: &str) -> Option<&TensorField> {
        self.fields.get(name)
    }

    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }
}

fn read_exact(
    file: &mut File,
    buf: &mut [u8],
    filename: &str,
    what: &str,
) -> Result<(), PbrtError> {
    file.read_exact(buf)
        .map_err(|e| PbrtError::error(&format!("{}: failed to read {}: {}", filename, what, e)))
}

fn read_u8(file: &mut File, filename: &str, what: &str) -> Result<u8, PbrtError> {
    let mut b = [0u8; 1];
    read_exact(file, &mut b, filename, what)?;
    Ok(b[0])
}

fn read_u16(file: &mut File, filename: &str, what: &str) -> Result<u16, PbrtError> {
    let mut b = [0u8; 2];
    read_exact(file, &mut b, filename, what)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u64(file: &mut File, filename: &str, what: &str) -> Result<u64, PbrtError> {
    let mut b = [0u8; 8];
    read_exact(file, &mut b, filename, what)?;
    Ok(u64::from_le_bytes(b))
}
