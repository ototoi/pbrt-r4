use flate2::{write::ZlibEncoder, Compression};
use std::io::Write;

pub const MAGIC: u64 = 0x304244566f6e614e;
pub const VERSION: u32 = (32 << 21) | (3 << 10) | 3;
pub const ROOT: usize = 736;
pub const UPPER: usize = 864;
pub const LOWER: usize = UPPER + 270400;
pub const LEAF: usize = LOWER + 33856;

pub fn put32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
pub fn put64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
pub fn putf(bytes: &mut [u8], offset: usize, value: f32) {
    put32(bytes, offset, value.to_bits());
}

fn header(bytes: &mut [u8], name: &str, min: [i32; 3], max: [i32; 3]) {
    let length = bytes.len();
    put64(bytes, 0, MAGIC);
    put32(bytes, 16, VERSION);
    put32(bytes, 20, 2); // HasBBox
    put32(bytes, 28, 1);
    put64(bytes, 32, length as u64);
    bytes[40..40 + name.len()].copy_from_slice(name.as_bytes());
    for axis in 0..3 {
        putf(bytes, 296 + (axis * 3 + axis) * 4, 1.0);
        putf(bytes, 332 + (axis * 3 + axis) * 4, 1.0);
        put64(bytes, 384 + (axis * 3 + axis) * 8, 1.0f64.to_bits());
        put64(bytes, 456 + (axis * 3 + axis) * 8, 1.0f64.to_bits());
        put64(bytes, 560 + axis * 8, (min[axis] as f64).to_bits());
        put64(bytes, 584 + axis * 8, (max[axis] as f64).to_bits());
        put64(bytes, 608 + axis * 8, 1.0f64.to_bits());
    }
    put32(bytes, 632, 2);
    put32(bytes, 636, 1);
}

pub fn root_tile(name: &str, value: f32, bounds: ([i32; 3], [i32; 3])) -> Vec<u8> {
    let mut bytes = vec![0; ROOT + 96];
    header(&mut bytes, name, bounds.0, bounds.1);
    put64(&mut bytes, 696, 64);
    for axis in 0..3 {
        put32(&mut bytes, ROOT + axis * 4, bounds.0[axis] as u32);
        put32(&mut bytes, ROOT + 12 + axis * 4, bounds.1[axis] as u32);
    }
    put32(&mut bytes, ROOT + 24, 1);
    put32(&mut bytes, ROOT + 64 + 16, 1);
    putf(&mut bytes, ROOT + 64 + 20, value);
    bytes
}

pub fn full_tree(negative_links: bool) -> (Vec<u8>, usize) {
    let (upper, lower, leaf, root) = if negative_links {
        let leaf = UPPER;
        let lower = leaf + 2144;
        let upper = lower + 33856;
        (upper, lower, leaf, upper + 270400)
    } else {
        (UPPER, LOWER, LEAF, ROOT)
    };
    let mut bytes = vec![
        0;
        if negative_links {
            root + 128
        } else {
            leaf + 2144
        }
    ];
    header(&mut bytes, "density", [0; 3], [7; 3]);
    for (level, node) in [leaf, lower, upper, root].into_iter().enumerate() {
        put64(&mut bytes, 672 + level * 8, (node - 672) as u64);
        if level < 3 {
            put32(&mut bytes, 704 + level * 4, 1);
        }
    }
    for axis in 0..3 {
        put32(&mut bytes, root + 12 + axis * 4, 7);
    }
    put32(&mut bytes, root + 24, 2);
    putf(&mut bytes, root + 28, 0.25);
    put64(&mut bytes, root + 72, (upper as i64 - root as i64) as u64);
    // A constant root tile in the negative-x block.
    put64(&mut bytes, root + 96, 0xfffffu64 << 42);
    putf(&mut bytes, root + 116, 3.5);
    put64(&mut bytes, upper + 4128, 1);
    put64(
        &mut bytes,
        upper + 8256,
        (lower as i64 - upper as i64) as u64,
    );
    for i in 1..32768 {
        putf(&mut bytes, upper + 8256 + i * 8, i as f32 / 32768.0);
    }
    put64(&mut bytes, lower + 544, 1);
    put64(
        &mut bytes,
        lower + 1088,
        (leaf as i64 - lower as i64) as u64,
    );
    for i in 1..4096 {
        putf(&mut bytes, lower + 1088 + i * 8, i as f32 / 4096.0);
    }
    // The value mask is deliberately all-off; all values must survive.
    for i in 0..512 {
        putf(&mut bytes, leaf + 96 + i * 4, i as f32 / 512.0);
    }
    (bytes, root)
}

pub fn file_bytes(grids: &[Vec<u8>], zip: bool) -> Vec<u8> {
    let mut output = vec![0; 16];
    put64(&mut output, 0, MAGIC);
    put32(&mut output, 8, VERSION);
    output[12..14].copy_from_slice(&(grids.len() as u16).to_le_bytes());
    output[14..16].copy_from_slice(&(u16::from(zip)).to_le_bytes());
    let mut payloads = Vec::new();
    for grid in grids {
        let mut payload = Vec::new();
        if zip {
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(grid).unwrap();
            let compressed = encoder.finish().unwrap();
            payload.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
            payload.extend_from_slice(&compressed);
        } else {
            payload.extend_from_slice(grid);
        }
        let name_end = grid[40..296].iter().position(|v| *v == 0).unwrap();
        let name = &grid[40..40 + name_end + 1];
        let mut meta = vec![0; 176];
        put64(&mut meta, 0, grid.len() as u64);
        put64(&mut meta, 8, payload.len() as u64);
        let hash = name[..name.len() - 1].iter().fold(0u64, |hash, byte| {
            hash.wrapping_mul(67)
                .wrapping_add(u64::from(*byte) + (hash >> 56))
        });
        put64(&mut meta, 16, hash);
        meta[32..36].copy_from_slice(&grid[636..640]);
        meta[36..40].copy_from_slice(&grid[632..636]);
        meta[40..88].copy_from_slice(&grid[560..608]);
        let root = 672 + u64::from_le_bytes(grid[696..704].try_into().unwrap()) as usize;
        meta[88..112].copy_from_slice(&grid[root..root + 24]);
        meta[112..136].copy_from_slice(&grid[608..632]);
        put32(&mut meta, 136, name.len() as u32);
        meta[140..152].copy_from_slice(&grid[704..716]);
        put32(&mut meta, 152, 1);
        meta[168..170].copy_from_slice(&(u16::from(zip)).to_le_bytes());
        put32(&mut meta, 172, VERSION);
        output.extend_from_slice(&meta);
        output.extend_from_slice(name);
        payloads.push(payload);
    }
    for payload in payloads {
        output.extend_from_slice(&payload);
    }
    output
}
