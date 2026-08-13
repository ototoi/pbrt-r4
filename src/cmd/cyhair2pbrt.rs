// Convert CyHair files to pbrt curve shapes.
//
// The HAIR loader is based on the implementation shipped in
// pbrt-v4/src/pbrt/cmd/cyhair2pbrt.cpp.
//
// Copyright (c) 2016 Light Transport Entertainment, Inc.
// The original loader is available under the MIT License:
// https://opensource.org/licenses/MIT

use std::env;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io::{self, Write};
use std::path::Path;

const FLAG_SEGMENTS: u32 = 1 << 0;
const FLAG_POINTS: u32 = 1 << 1;
const FLAG_THICKNESS: u32 = 1 << 2;
const FLAG_TRANSPARENCY: u32 = 1 << 3;
const FLAG_COLOR: u32 = 1 << 4;

const TO_C2B: [[f32; 4]; 4] = [
    [0.0, 1.0, 0.0, 0.0],
    [-1.0 / 6.0, 1.0, 1.0 / 6.0, 0.0],
    [0.0, 1.0 / 6.0, 1.0, -1.0 / 6.0],
    [0.0, 0.0, 1.0, 0.0],
];
const TO_C2B0: [[f32; 4]; 4] = [
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.5, 2.0 / 3.0, -1.0 / 6.0],
    [0.0, 1.0 / 6.0, 1.0, -1.0 / 6.0],
    [0.0, 0.0, 1.0, 0.0],
];
const TO_C2B1: [[f32; 4]; 4] = [
    [0.0, 1.0, 0.0, 0.0],
    [-1.0 / 6.0, 1.0, 1.0 / 6.0, 0.0],
    [-1.0 / 6.0, 2.0 / 3.0, 0.5, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

#[derive(Clone, Copy, Debug)]
struct Point3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Point3 {
    fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    fn add_scaled(self, other: Self, scale: f32) -> Self {
        Self {
            x: self.x + other.x * scale,
            y: self.y + other.y * scale,
            z: self.z + other.z * scale,
        }
    }

    fn scaled(self, scale: f32) -> Self {
        Self {
            x: self.x * scale,
            y: self.y * scale,
            z: self.z * scale,
        }
    }

    fn add(self, other: Self) -> Self {
        self.add_scaled(other, 1.0)
    }
}

fn mul_matrix(matrix: &[[f32; 4]; 4], points: [Point3; 4]) -> [Point3; 4] {
    std::array::from_fn(|row| {
        let mut result = Point3::zero();
        for (column, point) in points.into_iter().enumerate() {
            result = result.add_scaled(point, matrix[row][column]);
        }
        result
    })
}

fn catmull_rom_to_bezier(points: &[Point3], segment: usize) -> [Point3; 4] {
    if points.len() == 2 {
        let p0 = points[segment];
        let p1 = points[segment + 1];
        return [
            p0,
            p0.scaled(2.0 / 3.0).add(p1.scaled(1.0 / 3.0)),
            p0.scaled(1.0 / 3.0).add(p1.scaled(2.0 / 3.0)),
            p1,
        ];
    }

    let control = if segment == 0 {
        [Point3::zero(), points[0], points[1], points[2]]
    } else if segment == points.len() - 2 {
        [
            points[segment - 1],
            points[segment],
            points[segment + 1],
            Point3::zero(),
        ]
    } else {
        [
            points[segment - 1],
            points[segment],
            points[segment + 1],
            points[segment + 2],
        ]
    };
    let matrix = if segment == 0 {
        &TO_C2B0
    } else if segment == points.len() - 2 {
        &TO_C2B1
    } else {
        &TO_C2B
    };
    mul_matrix(matrix, control)
}

#[derive(Debug)]
struct CyHairError(String);

impl Display for CyHairError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CyHairError {}

type Result<T> = std::result::Result<T, CyHairError>;

fn error(message: impl Into<String>) -> CyHairError {
    CyHairError(message.into())
}

struct CyHair {
    num_strands: usize,
    default_segments: usize,
    default_thickness: f32,
    segments: Vec<u16>,
    points: Vec<Point3>,
    strand_offsets: Vec<usize>,
}

impl CyHair {
    fn load(path: &Path) -> Result<Self> {
        let data = fs::read(path).map_err(|e| error(format!("{}: {e}", path.display())))?;
        if data.len() < 128 {
            return Err(error(format!(
                "{}: file is shorter than the 128-byte header",
                path.display()
            )));
        }
        if &data[..4] != b"HAIR" {
            return Err(error(format!("{}: invalid HAIR magic", path.display())));
        }

        let read_u32 = |offset| {
            data.get(offset..offset + 4)
                .and_then(|bytes| bytes.try_into().ok())
                .map(u32::from_le_bytes)
                .ok_or_else(|| error("invalid HAIR header"))
        };
        let read_f32 = |offset| Ok(f32::from_bits(read_u32(offset)?));
        let num_strands = usize::try_from(read_u32(4)?)
            .map_err(|_| error("HAIR strand count does not fit in usize"))?;
        let total_points = usize::try_from(read_u32(8)?)
            .map_err(|_| error("HAIR point count does not fit in usize"))?;
        let flags = read_u32(12)?;
        let default_segments = usize::try_from(read_u32(16)?)
            .map_err(|_| error("HAIR default segment count does not fit in usize"))?;
        let default_thickness = read_f32(20)?;

        if flags & FLAG_POINTS == 0 {
            return Err(error("No point data in CyHair."));
        }
        if flags & FLAG_SEGMENTS == 0 && default_segments < 1 {
            return Err(error("No valid segment information in CyHair."));
        }

        let mut cursor = 128usize;
        let take = |cursor: &mut usize, count: usize| -> Result<&[u8]> {
            let end = cursor
                .checked_add(count)
                .ok_or_else(|| error("HAIR size overflow"))?;
            let bytes = data
                .get(*cursor..end)
                .ok_or_else(|| error("truncated HAIR array"))?;
            *cursor = end;
            Ok(bytes)
        };
        let segments = if flags & FLAG_SEGMENTS != 0 {
            let bytes = take(
                &mut cursor,
                num_strands
                    .checked_mul(2)
                    .ok_or_else(|| error("HAIR size overflow"))?,
            )?;
            bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        } else {
            Vec::new()
        };
        let point_bytes = total_points
            .checked_mul(12)
            .ok_or_else(|| error("HAIR size overflow"))?;
        let points = take(&mut cursor, point_bytes)?
            .chunks_exact(12)
            .map(|b| Point3 {
                x: f32::from_le_bytes(b[0..4].try_into().unwrap()),
                y: f32::from_le_bytes(b[4..8].try_into().unwrap()),
                z: f32::from_le_bytes(b[8..12].try_into().unwrap()),
            })
            .collect();
        if flags & FLAG_THICKNESS != 0 {
            let _ = take(
                &mut cursor,
                total_points
                    .checked_mul(4)
                    .ok_or_else(|| error("HAIR size overflow"))?,
            )?;
        }
        if flags & FLAG_TRANSPARENCY != 0 {
            let _ = take(
                &mut cursor,
                total_points
                    .checked_mul(4)
                    .ok_or_else(|| error("HAIR size overflow"))?,
            )?;
        }
        if flags & FLAG_COLOR != 0 {
            let _ = take(
                &mut cursor,
                total_points
                    .checked_mul(12)
                    .ok_or_else(|| error("HAIR size overflow"))?,
            )?;
        }

        let mut strand_offsets = Vec::with_capacity(num_strands);
        let mut offset = 0usize;
        for index in 0..num_strands {
            strand_offsets.push(offset);
            let segments = segments
                .get(index)
                .copied()
                .map(usize::from)
                .unwrap_or(default_segments);
            offset = offset
                .checked_add(
                    segments
                        .checked_add(1)
                        .ok_or_else(|| error("HAIR strand offset overflow"))?,
                )
                .ok_or_else(|| error("HAIR strand offset overflow"))?;
        }
        if offset != total_points {
            return Err(error(format!("HAIR point count mismatch: offsets require {offset}, header declares {total_points}")));
        }

        Ok(Self {
            num_strands,
            default_segments,
            default_thickness,
            segments,
            points,
            strand_offsets,
        })
    }

    fn convert(&self, max_strands: i32, thickness: f32) -> Result<(Vec<Point3>, Vec<f32>)> {
        let count = if max_strands > 0 {
            (max_strands as usize).min(self.num_strands)
        } else {
            self.num_strands
        };
        let mut vertices = Vec::new();
        let mut radii = Vec::new();
        for strand in 0..count {
            let num_segments = self
                .segments
                .get(strand)
                .copied()
                .map(usize::from)
                .unwrap_or(self.default_segments);
            if num_segments < 2 {
                continue;
            }
            let start = self.strand_offsets[strand];
            let end = start
                .checked_add(num_segments)
                .ok_or_else(|| error("HAIR point range overflow"))?;
            let segment_points: Vec<_> = self.points[start..end]
                .iter()
                .map(|p| Point3 {
                    x: p.x,
                    y: p.z,
                    z: p.y,
                })
                .collect();
            for s in 1..num_segments - 1 {
                let curve = catmull_rom_to_bezier(&segment_points, s - 1);
                vertices.extend(curve);
                radii.extend(std::iter::repeat_n(
                    if thickness > 0.0 {
                        thickness
                    } else {
                        self.default_thickness
                    },
                    4,
                ));
            }
        }
        Ok((vertices, radii))
    }
}

fn format_output(input: &str, user_thickness: f32, vertices: &[Point3], radii: &[f32]) -> String {
    let mut bounds_min = [1.0e30f64; 3];
    let mut bounds_max = [-1.0e30f64; 3];
    for (point, radius) in vertices
        .iter()
        .zip(radii.iter().cycle())
        .take(vertices.len())
    {
        let values = [point.x as f64, point.y as f64, point.z as f64];
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(values[axis] - *radius as f64);
            bounds_max[axis] = bounds_max[axis].max(values[axis] + *radius as f64);
        }
    }
    let mut output = format!("# Converted from \"{input}\" by cyhair2pbrt\n# The number of strands = {}. user_thickness = {user_thickness:.6}\n# Scene bounds: ({:.6}, {:.6}, {:.6}) - ({:.6}, {:.6}, {:.6})\n\n\n", radii.len() / 4, bounds_min[0], bounds_min[1], bounds_min[2], bounds_max[0], bounds_max[1], bounds_max[2]);
    for (curve_index, curve) in vertices.chunks_exact(4).enumerate() {
        output.push_str("Shape \"curve\" \"string type\" [ \"cylinder\" ] \"point3 P\" [ ");
        for point in curve {
            output.push_str(&format!("{:.6} {:.6} {:.6} ", point.x, point.y, point.z));
        }
        output.push_str(&format!(
            " ] \"float width0\" [ {:.6} ] \"float width1\" [ {:.6} ]\n",
            radii[4 * curve_index],
            radii[4 * curve_index + 3]
        ));
    }
    output
}

fn usage() {
    eprintln!(
        "usage: cyhair2pbrt [CyHair filename] [pbrt output filename] (max strands) (thickness)"
    );
}

fn run(args: &[String]) -> Result<()> {
    debug_assert!(args.len() > 2);
    if args.len() > 5 {
        return Err(error("too many arguments"));
    }
    let max_strands = args
        .get(3)
        .map(|s| s.parse::<i32>().map_err(|_| error("invalid max strands")))
        .transpose()?
        .unwrap_or(-1);
    if max_strands < -1 {
        return Err(error("max strands must be -1 or non-negative"));
    }
    let thickness = args
        .get(4)
        .map(|s| s.parse::<f32>().map_err(|_| error("invalid thickness")))
        .transpose()?
        .unwrap_or(1.0);
    if !thickness.is_finite() {
        return Err(error("thickness must be finite"));
    }
    let hair = CyHair::load(Path::new(&args[1]))?;
    let (vertices, radii) = hair.convert(max_strands, thickness)?;
    let output = format_output(&args[1], thickness, &vertices, &radii);
    if args[2] == "-" {
        io::stdout()
            .write_all(output.as_bytes())
            .map_err(|e| error(e.to_string()))?;
    } else {
        fs::write(&args[2], output).map_err(|e| error(format!("{}: {e}", args[2])))?;
    }
    eprintln!("Converted {} strands.", radii.len() / 4);
    Ok(())
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() <= 2
        || args
            .get(1)
            .is_some_and(|arg| arg == "--help" || arg == "-h")
    {
        usage();
        std::process::exit(1);
    }
    if let Err(err) = run(&args) {
        eprintln!("cyhair2pbrt: {err}");
        std::process::exit(1);
    }
}
