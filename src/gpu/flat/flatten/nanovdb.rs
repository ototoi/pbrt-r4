use super::super::medium::NanoVDBMedium;
use super::super::sparse_grid::{
    SparseGridEntry, SparseGridRoot, SparseScalarGrid, SPARSE_LEAF_SIZE, SPARSE_LOWER_SIZE,
    SPARSE_UPPER_SIZE,
};
use crate::paramdict::ParameterDictionary;
use crate::util::error::PbrtError;
use nanovdb_rs::{
    Codec, Grid, GridDataHeader, GridType, NvdbFile, SegmentHeader, ValidatedFloatTree,
};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom};

// NanoVDB v32 Float layout; source offsets are confined to this decoder.
const GRID_SIZE: usize = 672;
const TREE_SIZE: usize = 64;
const ROOT_HEADER_SIZE: usize = 64;
const ROOT_TILE_SIZE: usize = 32;
const UPPER_TABLE: usize = 8256;
const LOWER_TABLE: usize = 1088;
const LEAF_VALUES: usize = 96;
const MAGIC: u64 = 0x304244566f6e614e;

pub fn create_nanovdb(parameters: &ParameterDictionary) -> Result<NanoVDBMedium, PbrtError> {
    let filename = parameters.get_one_filename("filename", "");
    if filename.is_empty() {
        return Err(error("Must supply filename to nanovdb medium."));
    }
    if parameters.get_one_bool("dense", false)
        || parameters
            .get_floats_ref("density")
            .is_some_and(|values| !values.is_empty())
    {
        return Err(error(
            "NanoVDB requires a sparse file grid; dense/inline density aliases are unsupported.",
        ));
    }
    let load = || {
        validate_file_envelope(&filename)?;
        let file = NvdbFile::open(&filename).map_err(|e| error(&e.to_string()))?;
        let name = parameters.get_one_string("gridname", "density");
        let grid = file
            .grids()
            .iter()
            .find(|g| g.name() == name)
            .ok_or_else(|| error(&format!("Missing density grid {name:?}.")))?;
        validate_metadata(grid)?;
        let mut medium = decode_density_grid(grid.raw_bytes())?;
        let temperature = parameters.get_one_string("temperaturename", "temperature");
        if let Some(grid) = file.grids().iter().find(|g| g.name() == temperature) {
            validate_metadata(grid)?;
            let header = validate_header(grid.raw_bytes())?;
            let lescale = parameters.get_one_float("Lescale", 1.0);
            if !lescale.is_finite() || lescale != 0.0 {
                return Err(error(
                    "Temperature emission is not supported; requires Lescale=0.",
                ));
            }
            let (min, max) = header_bounds(&header)?;
            for axis in 0..3 {
                medium.bounds_min[axis] = medium.bounds_min[axis].min(min[axis]);
                medium.bounds_max[axis] = medium.bounds_max[axis].max(max[axis]);
            }
        }
        // Temperature aliases are consumed as in NanoVDBMedium::Create.
        parameters.get_one_float(
            "temperatureoffset",
            parameters.get_one_float("temperaturecutoff", 0.0),
        );
        parameters.get_one_float("temperaturescale", 1.0);
        medium.majorant = build_majorant(&medium)?;
        Ok(medium)
    };
    load().map_err(|e: PbrtError| error(&format!("{filename}: {e}")))
}

/// Decode one raw Float grid. The caller can union temperature bounds before
/// building the majorant. The returned payload owns no source bytes.
pub fn decode_density_grid(bytes: &[u8]) -> Result<NanoVDBMedium, PbrtError> {
    let header = validate_header(bytes)?;
    ValidatedFloatTree::new(bytes).ok_or_else(|| error("Invalid Float tree links."))?;
    let mut decoder = Decoder {
        bytes,
        grid: SparseScalarGrid::default(),
        nodes: std::array::from_fn(|_| HashMap::new()),
        spans: BTreeMap::new(),
    };
    decoder.claim(0, GRID_SIZE + TREE_SIZE)?;
    let root = GRID_SIZE
        .checked_add(
            usize::try_from(read_u64(bytes, GRID_SIZE + 24)?)
                .map_err(|_| error("Root offset exceeds usize."))?,
        )
        .ok_or_else(|| error("Root offset overflow."))?;
    let count = read_u32(bytes, root + 24)? as usize;
    let root_size = count
        .checked_mul(ROOT_TILE_SIZE)
        .and_then(|n| n.checked_add(ROOT_HEADER_SIZE))
        .ok_or_else(|| error("Root table size overflow."))?;
    decoder.claim(root, root_size)?;
    decoder.grid.background = read_f32(bytes, root + 28)?;
    for index in 0..count {
        let tile = root + ROOT_HEADER_SIZE + index * ROOT_TILE_SIZE;
        let key = read_u64(bytes, tile)?;
        // CoordToKey has 20 coordinate bits in each 21-bit slot.
        let allowed = (0xfffffu64 << 42) | (0xfffffu64 << 21) | 0xfffff;
        if key & !allowed != 0 {
            return Err(error("Invalid root coordinate key."));
        }
        let block =
            [42, 21, 0].map(|shift| (((((key >> shift) & 0xfffff) as u32) << 12) as i32) >> 12);
        let relative = read_i64(bytes, tile + 8)?;
        let entry = if relative == 0 {
            SparseGridEntry {
                child_index: None,
                value: read_f32(bytes, tile + 20)?,
            }
        } else {
            SparseGridEntry {
                child_index: Some(decoder.node(child_target(root, relative)?, 2)?),
                value: 0.0,
            }
        };
        decoder.grid.roots.push(SparseGridRoot { block, entry });
    }
    for level in 0..3 {
        if decoder.nodes[level].len() != read_u32(bytes, GRID_SIZE + 32 + level * 4)? as usize {
            return Err(error("Tree node count disagrees with reachable nodes."));
        }
    }
    decoder.grid.validate()?;
    let mut inverse_map_f = [[0.0; 3]; 3];
    let mut translation_f = [0.0; 3];
    for row in 0..3 {
        translation_f[row] = read_f32(bytes, 368 + row * 4)?;
        for col in 0..3 {
            inverse_map_f[row][col] = read_f32(bytes, 332 + (row * 3 + col) * 4)?;
        }
    }
    if translation_f.iter().any(|v| !v.is_finite())
        || (0..3).any(|row| {
            (0..3).any(|col| {
                let v = inverse_map_f[row][col];
                !v.is_finite() || if row == col { v <= 0.0 } else { v != 0.0 }
            })
        })
    {
        return Err(error(
            "Density grid requires a finite positive diagonal inverse map and translation.",
        ));
    }
    let mut index_bbox_min = [0; 3];
    let mut index_bbox_max = [0; 3];
    for axis in 0..3 {
        index_bbox_min[axis] = read_u32(bytes, root + axis * 4)? as i32;
        index_bbox_max[axis] = read_u32(bytes, root + 12 + axis * 4)? as i32;
    }
    if (0..3).any(|axis| index_bbox_min[axis] > index_bbox_max[axis]) {
        return Err(error("Empty density index bounds."));
    }
    let (bounds_min, bounds_max) = header_bounds(&header)?;
    Ok(NanoVDBMedium {
        density: decoder.grid,
        index_bbox_min,
        index_bbox_max,
        inverse_map_f,
        translation_f,
        bounds_min,
        bounds_max,
        majorant_resolution: [64; 3],
        majorant: Vec::new(),
    })
}

struct Decoder<'a> {
    bytes: &'a [u8],
    grid: SparseScalarGrid,
    nodes: [HashMap<usize, u32>; 3],
    spans: BTreeMap<usize, usize>,
}

impl Decoder<'_> {
    fn claim(&mut self, start: usize, size: usize) -> Result<(), PbrtError> {
        let end = start
            .checked_add(size)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| error("Node range exceeds raw grid."))?;
        if self
            .spans
            .range(..=start)
            .next_back()
            .is_some_and(|(_, end)| *end > start)
            || self
                .spans
                .range(start..)
                .next()
                .is_some_and(|(next, _)| *next < end)
        {
            return Err(error(
                "Source node ranges overlap or reference a different level.",
            ));
        }
        self.spans.insert(start, end);
        Ok(())
    }

    fn node(&mut self, source: usize, level: usize) -> Result<u32, PbrtError> {
        if let Some(index) = self.nodes[level].get(&source) {
            return Ok(*index);
        }
        let (size, table, count) = match level {
            2 => (
                UPPER_TABLE + SPARSE_UPPER_SIZE * 8,
                UPPER_TABLE,
                SPARSE_UPPER_SIZE,
            ),
            1 => (
                LOWER_TABLE + SPARSE_LOWER_SIZE * 8,
                LOWER_TABLE,
                SPARSE_LOWER_SIZE,
            ),
            _ => (
                LEAF_VALUES + SPARSE_LEAF_SIZE * 4,
                LEAF_VALUES,
                SPARSE_LEAF_SIZE,
            ),
        };
        self.claim(source, size)?;
        let length = match level {
            2 => self.grid.upper_entries.len(),
            1 => self.grid.lower_entries.len(),
            _ => self.grid.leaf_values.len(),
        };
        let end = length
            .checked_add(count)
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or_else(|| error("Converted sparse grid exceeds u32."))?;
        let index = (length / count) as u32;
        self.nodes[level].insert(source, index);
        if level == 0 {
            self.grid
                .leaf_values
                .try_reserve(count)
                .map_err(|e| error(&e.to_string()))?;
            for i in 0..count {
                self.grid
                    .leaf_values
                    .push(read_f32(self.bytes, source + table + i * 4)?);
            }
        } else {
            let entries = if level == 2 {
                &mut self.grid.upper_entries
            } else {
                &mut self.grid.lower_entries
            };
            entries
                .try_reserve(count)
                .map_err(|e| error(&e.to_string()))?;
            entries.resize(end, SparseGridEntry::default());
            let mask = source + 32 + count / 8;
            for i in 0..count {
                let at = source + table + i * 8;
                let child = (self.bytes[mask + i / 8] >> (i % 8)) & 1 != 0;
                let entry = if child {
                    let relative = read_i64(self.bytes, at)?;
                    if relative == 0 {
                        return Err(error("Internal child offset is zero."));
                    }
                    SparseGridEntry {
                        child_index: Some(self.node(child_target(source, relative)?, level - 1)?),
                        value: 0.0,
                    }
                } else {
                    SparseGridEntry {
                        child_index: None,
                        value: read_f32(self.bytes, at)?,
                    }
                };
                let entries = if level == 2 {
                    &mut self.grid.upper_entries
                } else {
                    &mut self.grid.lower_entries
                };
                entries[length + i] = entry;
            }
        }
        Ok(index)
    }
}

fn child_target(base: usize, relative: i64) -> Result<usize, PbrtError> {
    let target = i128::try_from(base).map_err(|_| error("Source offset exceeds i128."))?
        + i128::from(relative);
    usize::try_from(target)
        .ok()
        .filter(|v| v % 32 == 0)
        .ok_or_else(|| error("Invalid source child offset."))
}

fn validate_header(bytes: &[u8]) -> Result<GridDataHeader, PbrtError> {
    let header = GridDataHeader::parse(bytes).ok_or_else(|| error("Truncated GridData."))?;
    if header.magic != MAGIC
        || header.version.major() != 32
        || header.grid_size != bytes.len() as u64
        || header.grid_type != GridType::Float as u32
        || !matches!(header.grid_class, 0 | 2)
        || header.grid_count == 0
        || header.grid_index >= header.grid_count
    {
        return Err(error(
            "Unsupported or inconsistent Float grid header (requires ABI32, FogVolume/Unknown).",
        ));
    }
    header_bounds(&header)?;
    Ok(header)
}

fn header_bounds(header: &GridDataHeader) -> Result<([f32; 3], [f32; 3]), PbrtError> {
    let min = [
        header.world_bbox_min.x as f32,
        header.world_bbox_min.y as f32,
        header.world_bbox_min.z as f32,
    ];
    let max = [
        header.world_bbox_max.x as f32,
        header.world_bbox_max.y as f32,
        header.world_bbox_max.z as f32,
    ];
    if (0..3).any(|axis| {
        !min[axis].is_finite()
            || !max[axis].is_finite()
            || min[axis] >= max[axis]
            || !(max[axis] - min[axis]).is_finite()
    }) {
        return Err(error("Invalid world bounds."));
    }
    Ok((min, max))
}

fn validate_metadata(grid: &Grid) -> Result<(), PbrtError> {
    if grid.value_type() != GridType::Float {
        return Err(error(&format!("Grid {:?} must be Float.", grid.name())));
    }
    let header = validate_header(grid.raw_bytes())?;
    let meta = grid.grid_metadata();
    if header.version != meta.version
        || header.grid_name != meta.name
        || header.grid_class != meta.grid_class as u32
        || header.grid_size != meta.grid_size
        || header.world_bbox_min != meta.world_bbox_min
        || header.world_bbox_max != meta.world_bbox_max
    {
        return Err(error("Grid metadata disagrees with GridData."));
    }
    ValidatedFloatTree::new(grid.raw_bytes()).ok_or_else(|| error("Invalid Float tree links."))?;
    let bytes = grid.raw_bytes();
    let root = GRID_SIZE
        .checked_add(
            usize::try_from(read_u64(bytes, GRID_SIZE + 24)?)
                .map_err(|_| error("Root offset exceeds usize."))?,
        )
        .ok_or_else(|| error("Root offset overflow."))?;
    for axis in 0..3 {
        if read_u32(bytes, root + axis * 4)? as i32 != meta.index_bbox_min[axis]
            || read_u32(bytes, root + 12 + axis * 4)? as i32 != meta.index_bbox_max[axis]
        {
            return Err(error("Grid metadata disagrees with index bounds."));
        }
    }
    for level in 0..3 {
        if read_u32(bytes, GRID_SIZE + 32 + level * 4)? != meta.node_count[level] {
            return Err(error("Grid metadata disagrees with node counts."));
        }
    }
    Ok(())
}

// The reader's uncompressed path does not check its final mmap range. Check
// segment extents before calling it so truncated user files return an error.
fn validate_file_envelope(filename: &str) -> Result<(), PbrtError> {
    let mut file = std::fs::File::open(filename).map_err(|e| error(&e.to_string()))?;
    let size = file.metadata().map_err(|e| error(&e.to_string()))?.len();
    let mut cursor = 0u64;
    while cursor < size {
        let mut bytes = [0; 16];
        file.read_exact(&mut bytes)
            .map_err(|e| error(&e.to_string()))?;
        let header = SegmentHeader::parse(&bytes, cursor).map_err(|e| error(&e.to_string()))?;
        if header.version.major() != 32 || header.grid_count == 0 {
            return Err(error("Unsupported segment version or empty segment."));
        }
        cursor = cursor
            .checked_add(16)
            .ok_or_else(|| error("File offset overflow."))?;
        let mut sizes = Vec::new();
        for _ in 0..header.grid_count {
            let mut meta = [0; 176];
            file.read_exact(&mut meta)
                .map_err(|e| error(&e.to_string()))?;
            if read_u32(&meta, 172)? != header.version.0
                || u16::from_le_bytes([meta[168], meta[169]]) != header.codec as u16
            {
                return Err(error("Metadata version/codec disagrees with the segment."));
            }
            let name_size = read_u32(&meta, 136)? as u64;
            cursor = cursor
                .checked_add(176)
                .and_then(|v| v.checked_add(name_size))
                .filter(|end| *end <= size)
                .ok_or_else(|| error("Truncated file metadata."))?;
            if name_size == 0 {
                return Err(error("Empty metadata name."));
            }
            file.seek(SeekFrom::Start(cursor))
                .map_err(|e| error(&e.to_string()))?;
            sizes.push(read_u64(&meta, 0)?);
        }
        for grid_size in sizes {
            let stored = match header.codec {
                Codec::None => grid_size,
                Codec::Zip => {
                    let mut bytes = [0; 8];
                    file.read_exact(&mut bytes)
                        .map_err(|e| error(&e.to_string()))?;
                    cursor = cursor
                        .checked_add(8)
                        .ok_or_else(|| error("File offset overflow."))?;
                    u64::from_le_bytes(bytes)
                }
                _ => return Err(error("Unsupported NanoVDB compression codec.")),
            };
            if grid_size < (GRID_SIZE + TREE_SIZE) as u64 {
                return Err(error("Truncated raw grid."));
            }
            cursor = cursor
                .checked_add(stored)
                .filter(|end| *end <= size)
                .ok_or_else(|| error("Truncated grid data."))?;
            file.seek(SeekFrom::Start(cursor))
                .map_err(|e| error(&e.to_string()))?;
        }
    }
    if size == 0 {
        return Err(error("Empty NanoVDB file."));
    }
    Ok(())
}

pub fn world_to_index_f(medium: &NanoVDBMedium, point: [f32; 3]) -> [f32; 3] {
    let d = std::array::from_fn::<_, 3, _>(|axis| point[axis] - medium.translation_f[axis]);
    medium
        .inverse_map_f
        .map(|row| d[0].mul_add(row[0], d[1].mul_add(row[1], d[2] * row[2])))
}

fn majorant_index(medium: &NanoVDBMedium, point: [f32; 3]) -> [f32; 3] {
    let d = std::array::from_fn::<_, 3, _>(|axis| {
        f64::from(point[axis]) - f64::from(medium.translation_f[axis])
    });
    medium.inverse_map_f.map(|row| {
        (d[0] as f32).mul_add(
            row[0],
            (d[1] as f32).mul_add(row[1], (d[2] * f64::from(row[2])) as f32),
        )
    })
}

pub fn build_majorant(medium: &NanoVDBMedium) -> Result<Vec<f32>, PbrtError> {
    medium.density.validate()?;
    if medium.majorant_resolution != [64; 3] {
        return Err(error("NanoVDB majorant resolution must be 64 cubed."));
    }
    if (0..3).any(|axis| {
        !medium.bounds_min[axis].is_finite()
            || !medium.bounds_max[axis].is_finite()
            || medium.bounds_min[axis] >= medium.bounds_max[axis]
            || !(medium.bounds_max[axis] - medium.bounds_min[axis]).is_finite()
            || medium.index_bbox_min[axis] > medium.index_bbox_max[axis]
            || !medium.translation_f[axis].is_finite()
            || (0..3).any(|col| {
                let v = medium.inverse_map_f[axis][col];
                !v.is_finite() || if axis == col { v <= 0.0 } else { v != 0.0 }
            })
    }) {
        return Err(error("Invalid NanoVDB bounds or map."));
    }
    (0..64usize.pow(3))
        .into_par_iter()
        .map(|index| {
            let cell = [index % 64, (index / 64) % 64, index / 4096];
            let corner = |increment: usize| {
                std::array::from_fn(|axis| {
                    let t = (cell[axis] + increment) as f32 / 64.0;
                    (1.0 - t) * medium.bounds_min[axis] + t * medium.bounds_max[axis]
                })
            };
            let p0 = corner(0);
            let p1 = corner(1);
            let i0 = majorant_index(medium, p0);
            let i1 = majorant_index(medium, p1);
            let mut lo = [0; 3];
            let mut hi = [0; 3];
            for axis in 0..3 {
                lo[axis] = checked_coordinate((f64::from(i0[axis]) - 1.0).trunc())?
                    .max(medium.index_bbox_min[axis]);
                hi[axis] = checked_coordinate((f64::from(i1[axis]) + 1.0).trunc())?
                    .min(medium.index_bbox_max[axis]);
            }
            let maximum = range_max(&medium.density, lo, hi)?;
            let s0 = world_to_index_f(medium, p0);
            let s1 = world_to_index_f(medium, p1);
            for axis in 0..3 {
                lo[axis] = checked_coordinate(f64::from(s0[axis]).floor())?;
                hi[axis] = checked_coordinate(f64::from(s1[axis]).floor() + 1.0)?;
            }
            if range_max(&medium.density, lo, hi)? > maximum {
                return Err(error(&format!(
                    "v4 majorant does not bound density in cell {cell:?}."
                )));
            }
            Ok(maximum)
        })
        .collect()
}

fn checked_coordinate(v: f64) -> Result<i32, PbrtError> {
    if !v.is_finite() || v < f64::from(i32::MIN) || v > f64::from(i32::MAX) {
        Err(error("Grid coordinate exceeds i32."))
    } else {
        Ok(v as i32)
    }
}

fn range_max(grid: &SparseScalarGrid, lo: [i32; 3], hi: [i32; 3]) -> Result<f32, PbrtError> {
    if (0..3).any(|axis| lo[axis] > hi[axis]) {
        return Ok(0.0);
    }
    if grid.roots.is_empty() {
        return Ok(grid.background);
    }
    let mut value = 0.0f32;
    for z in lo[2]..=hi[2] {
        for y in lo[1]..=hi[1] {
            for x in lo[0]..=hi[0] {
                value = value.max(grid.get_value([x, y, z])?);
            }
        }
    }
    Ok(value)
}

fn read<const N: usize>(bytes: &[u8], at: usize) -> Result<[u8; N], PbrtError> {
    let end = at
        .checked_add(N)
        .ok_or_else(|| error("Source offset overflow."))?;
    bytes
        .get(at..end)
        .and_then(|v| v.try_into().ok())
        .ok_or_else(|| error("Truncated source grid."))
}
fn read_u32(bytes: &[u8], at: usize) -> Result<u32, PbrtError> {
    Ok(u32::from_le_bytes(read(bytes, at)?))
}
fn read_u64(bytes: &[u8], at: usize) -> Result<u64, PbrtError> {
    Ok(u64::from_le_bytes(read(bytes, at)?))
}
fn read_i64(bytes: &[u8], at: usize) -> Result<i64, PbrtError> {
    Ok(i64::from_le_bytes(read(bytes, at)?))
}
fn read_f32(bytes: &[u8], at: usize) -> Result<f32, PbrtError> {
    Ok(f32::from_le_bytes(read(bytes, at)?))
}
fn error(message: &str) -> PbrtError {
    PbrtError::error(&format!("GPU nanovdb: {message}"))
}
