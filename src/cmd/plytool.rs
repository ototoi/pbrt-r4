use std::ffi::OsString;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Write;
use std::path::Path;

use pbrt_r4::util::base::{Float, Normal3f, Point2f, Point3f, Vector3f};
use pbrt_r4::util::geometry::Bounds3f;
use pbrt_r4::util::image::{Image, ImageWrapMode};
use pbrt_r4::util::imageio::{read_raw_image_with_encoding, ColorEncoding};
use pbrt_r4::util::mesh::TriQuadMesh;

#[derive(Debug)]
struct PlyToolError {
    message: String,
    show_help: bool,
}

impl PlyToolError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            show_help: false,
        }
    }

    fn with_help(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            show_help: true,
        }
    }

    fn show_help(&self) -> bool {
        self.show_help
    }
}

impl Display for PlyToolError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PlyToolError {}

/// Text from `pbrt-v4/src/pbrt/cmd/plytool.cpp::help`.
fn help_text() -> &'static str {
    "plytool provides a number of operations on PLY meshes.\n\
\n\
usage: plytool <command> [options]\n\
\n\
where <command> is:\n\
\n\
cat: Print a text representation of the mesh.\n\
\n\
displace: Apply displacement mapping to a mesh.\n\
\n\
info: Print general information about the mesh.\n\
\n\
split: Split the mesh into multiple PLY files.\n\
\n\
\"plytool help <command>\" provides detailed information about <command>.\n"
}

/// Per-command usage corresponding to `plytool.cpp::help(std::vector<std::string>)`.
fn command_help(command: &str) -> Result<&'static str, PlyToolError> {
    match command {
        "cat" => Ok("usage: plytool cat <filename>\n"),
        "info" => Ok("usage: plytool info <filename...>\n"),
        "displace" => Ok("usage: plytool displace [options] <filename>\n\n\
options:\n\
  --scale <s>       Scale to apply to displacement value in image.\n\
                    (Default: 1)\n\
  --uvscale <s>     Scale to apply to uv texture coordinates in image.\n\
                    (Default: 1)\n\
  --edge-length <s> Maximum length of an edge in the undisplaced mesh.\n\
                    (Default: 1)\n\
  --image <name>    Filename for image used to define displacements.\n\
  --outfile <name>  Filename name for emitted PLY file.\n"),
        "split" => Ok("usage: plytool split [options] <filename>\n\n\
options:\n\
  --maxfaces <n>    Maximum number of faces in an output PLY file.\n\
                    (Default: 1000000)\n\
  --outbase <name>  Base name for emitted PLY files.  Consecutive numbers\n\
                    and a \".ply\" suffix will be appended to <name>.\n\
                    (Default: based on <source.ply>.)\n"),
        _ => Err(PlyToolError::with_help(format!(
            "{command}: command unknown"
        ))),
    }
}

/// Corresponds to `pbrt-v4/src/pbrt/cmd/plytool.cpp::info`.
fn info_text(path: &Path) -> Result<String, PlyToolError> {
    let filename = path
        .to_str()
        .ok_or_else(|| PlyToolError::new("PLY path is not valid UTF-8"))?;
    let mesh =
        TriQuadMesh::read_ply(filename).map_err(|error| PlyToolError::new(error.to_string()))?;

    let mut points = mesh.p.iter();
    let first_point = points
        .next()
        .ok_or_else(|| PlyToolError::new("PLY file contains no vertex positions"))?;
    let mut bounds = Bounds3f::from((first_point.x, first_point.y, first_point.z));
    for point in points {
        bounds = bounds.union_p(point);
    }

    let mut output = String::new();
    output.push_str(&format!("{}:\n", path.display()));
    output.push_str(&format!("\tTriangles: {}\n", mesh.tri_indices.len() / 3));
    output.push_str(&format!("\tQuads: {}\n", mesh.quad_indices.len() / 4));
    output.push_str(&format!("\tVertex positions: {}\n", mesh.p.len()));
    output.push_str(&format!("\tVertex normals: {}\n", mesh.n.len()));
    output.push_str(&format!("\tVertex uvs: {}\n", mesh.uv.len()));
    output.push_str(&format!("\tFace indices: {}\n", mesh.face_indices.len()));

    let mut vertex_used = vec![false; mesh.p.len()];
    for &index in mesh.tri_indices.iter().chain(mesh.quad_indices.iter()) {
        let index = index as usize;
        if index >= vertex_used.len() {
            return Err(PlyToolError::new(format!(
                "vertex index {index} is out of bounds for {} vertices",
                vertex_used.len()
            )));
        }
        vertex_used[index] = true;
    }
    for (index, used) in vertex_used.iter().enumerate() {
        if !used {
            output.push_str(&format!("Notice: vertex {index} is not used.\n"));
        }
    }

    output.push_str(&format!("\tBounding box: {}\n", format_bounds(&bounds)));
    Ok(output)
}

fn info(paths: &[OsString]) -> Result<(), PlyToolError> {
    for path in paths {
        let path = Path::new(path);
        print!("{}", info_text(path)?);
    }
    Ok(())
}

fn cat(paths: &[OsString]) -> Result<(), PlyToolError> {
    let path = single_path(paths, "PLY filename")?;
    let mesh = read_mesh(&path)?;

    for triangle in mesh.tri_indices.chunks_exact(3) {
        println!("Triangle: {} {} {}", triangle[0], triangle[1], triangle[2]);
    }
    for quad in mesh.quad_indices.chunks_exact(4) {
        println!("Quad: {} {} {} {}", quad[0], quad[1], quad[2], quad[3]);
    }
    for (index, point) in mesh.p.iter().enumerate() {
        println!("Vertex position {index}: {}", format_point3(point));
    }
    for (index, normal) in mesh.n.iter().enumerate() {
        println!("Vertex normal {index}: {}", format_point3(normal));
    }
    for (index, uv) in mesh.uv.iter().enumerate() {
        println!("Vertex uv {index}: {}", format_point2(uv));
    }
    Ok(())
}

fn displace(args: &[OsString]) -> Result<(), PlyToolError> {
    let mut scale = 1.0;
    let mut uv_scale = 1.0;
    let mut edge_length = 1.0;
    let mut source = None;
    let mut image_path = None;
    let mut output = None;
    let mut index = 0;
    while index < args.len() {
        let argument = args[index].to_string_lossy();
        match argument.as_ref() {
            "--scale" => scale = parse_option(&args, &mut index, "scale")?,
            "--uvscale" => uv_scale = parse_option(&args, &mut index, "uvscale")?,
            "--edge-length" => edge_length = parse_option(&args, &mut index, "edge-length")?,
            "--image" => image_path = Some(option_string(&args, &mut index, "image")?),
            "--outfile" => output = Some(option_string(&args, &mut index, "outfile")?),
            value if value.starts_with('-') => {
                return Err(usage_error(format!("unexpected argument \"{value}\"")));
            }
            _ if source.is_none() => source = Some(path_string(&args[index])?),
            _ => return Err(usage_error(format!("unexpected argument \"{argument}\""))),
        }
        index += 1;
    }

    let source = required_path(source, "source PLY filename")?;
    let image_path = required_path(image_path, "image displacement map")?;
    let output = required_path(output, "output PLY filename")?;
    let mesh = read_mesh(&source)?;
    let raw = read_raw_image_with_encoding(&image_path, ColorEncoding::Linear).map_err(error)?;
    let image = Image::from_channels(raw.resolution, raw.channel_names(), raw.data_f32());
    let displaced = mesh
        .displace(
            |p0, p1| (p0 - p1).length(),
            edge_length,
            |point, normal, uv| {
                let uv = Point2f::new(uv_scale * uv.x, 1.0 - uv_scale * uv.y);
                let value: Float = (0..image.n_channels())
                    .map(|channel| {
                        image.bilerp_channel_with_wrap(
                            &uv,
                            channel,
                            ImageWrapMode::Repeat,
                            ImageWrapMode::Repeat,
                        )
                    })
                    .sum::<Float>()
                    / image.n_channels() as Float;
                point + Vector3f::new(normal.x, normal.y, normal.z) * (scale * value)
            },
        )
        .map_err(error)?;
    write_triangle_ply(
        &output,
        &displaced.tri_indices,
        &displaced.p,
        &displaced.n,
        &displaced.uv,
    )
}

fn split(args: &[OsString]) -> Result<(), PlyToolError> {
    let mut max_faces = 1_000_000usize;
    let mut source = None;
    let mut outbase = None;
    let mut index = 0;
    while index < args.len() {
        let argument = args[index].to_string_lossy();
        match argument.as_ref() {
            "--maxfaces" => {
                max_faces = parse_option_usize(&args, &mut index, "maxfaces")?;
                if max_faces == 0 {
                    return Err(usage_error("--maxfaces must be greater than zero"));
                }
            }
            "--outbase" => outbase = Some(option_string(&args, &mut index, "outbase")?),
            value if value.starts_with('-') => {
                return Err(usage_error(format!("unexpected argument \"{value}\"")));
            }
            _ if source.is_none() => source = Some(path_string(&args[index])?),
            _ => return Err(usage_error(format!("unexpected argument \"{argument}\""))),
        }
        index += 1;
    }

    let source = required_path(source, "source PLY filename")?;
    let mesh = read_mesh(&source)?;
    if !mesh.quad_indices.is_empty() {
        return Err(error_message(format!(
            "{source}: sorry, mesh has quad faces. plytool currently only supports triangle meshes."
        )));
    }
    if !mesh.face_indices.is_empty() {
        return Err(error_message(format!(
            "{source}: sorry, mesh has faceIndices, which are not currently supported by plytool."
        )));
    }

    let face_count = mesh.tri_indices.len() / 3;
    if face_count <= max_faces {
        eprintln!("{source}: mesh has {face_count} faces and so has not been split up.");
        return Ok(());
    }

    let output_base = outbase.unwrap_or_else(|| remove_extension(&source));
    let mesh_count = face_count.div_ceil(max_faces);
    eprintln!("{source}: mesh has {face_count} faces and will be split into {mesh_count} meshes.");
    let faces_per_mesh = face_count / mesh_count;
    for mesh_index in 0..mesh_count {
        let first_face = mesh_index * faces_per_mesh;
        let last_face = if mesh_index == mesh_count - 1 {
            face_count
        } else {
            (mesh_index + 1) * faces_per_mesh
        };
        let (indices, points, normals, uvs) = remap_triangle_range(&mesh, first_face, last_face);
        let filename = format!("{output_base}-{mesh_index:03}.ply");
        write_triangle_ply(&filename, &indices, &points, &normals, &uvs)?;
    }
    Ok(())
}

fn remap_triangle_range(
    mesh: &TriQuadMesh,
    first_face: usize,
    last_face: usize,
) -> (Vec<u32>, Vec<Point3f>, Vec<Normal3f>, Vec<Point2f>) {
    let mut remap = std::collections::HashMap::new();
    let mut indices = Vec::new();
    let mut points = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for &old_index in &mesh.tri_indices[3 * first_face..3 * last_face] {
        let new_index = if let Some(&new_index) = remap.get(&old_index) {
            new_index
        } else {
            let new_index = points.len() as u32;
            remap.insert(old_index, new_index);
            points.push(mesh.p[old_index as usize]);
            if !mesh.n.is_empty() {
                normals.push(mesh.n[old_index as usize]);
            }
            if !mesh.uv.is_empty() {
                uvs.push(mesh.uv[old_index as usize]);
            }
            new_index
        };
        indices.push(new_index);
    }
    (indices, points, normals, uvs)
}

fn write_triangle_ply(
    filename: &str,
    indices: &[u32],
    points: &[Point3f],
    normals: &[Normal3f],
    uvs: &[Point2f],
) -> Result<(), PlyToolError> {
    // Work around ply-rs 0.1.3 writing the element count instead of the
    // actual list length for binary list properties. pbrt-v4 emits binary
    // little-endian PLY files, so keep that format and write the payload
    // directly until the dependency can provide a compatible fix.
    if indices.len() % 3 != 0 {
        return Err(error_message(
            "triangle index count must be a multiple of three",
        ));
    }
    if (!normals.is_empty() && normals.len() != points.len())
        || (!uvs.is_empty() && uvs.len() != points.len())
    {
        return Err(error_message(
            "vertex attribute count does not match position count",
        ));
    }

    let mut file = File::create(filename).map_err(error)?;
    writeln!(file, "ply").map_err(error)?;
    writeln!(file, "format binary_little_endian 1.0").map_err(error)?;
    writeln!(file, "element vertex {}", points.len()).map_err(error)?;
    for property in ["x", "y", "z"] {
        writeln!(file, "property float {property}").map_err(error)?;
    }
    if !normals.is_empty() {
        for property in ["nx", "ny", "nz"] {
            writeln!(file, "property float {property}").map_err(error)?;
        }
    }
    if !uvs.is_empty() {
        for property in ["u", "v"] {
            writeln!(file, "property float {property}").map_err(error)?;
        }
    }
    writeln!(file, "element face {}", indices.len() / 3).map_err(error)?;
    writeln!(file, "property list uchar int vertex_indices").map_err(error)?;
    writeln!(file, "end_header").map_err(error)?;

    for index in 0..points.len() {
        for value in [points[index].x, points[index].y, points[index].z] {
            file.write_all(&(value as f32).to_le_bytes())
                .map_err(error)?;
        }
        if !normals.is_empty() {
            for value in [normals[index].x, normals[index].y, normals[index].z] {
                file.write_all(&(value as f32).to_le_bytes())
                    .map_err(error)?;
            }
        }
        if !uvs.is_empty() {
            for value in [uvs[index].x, uvs[index].y] {
                file.write_all(&(value as f32).to_le_bytes())
                    .map_err(error)?;
            }
        }
    }
    for triangle in indices.chunks_exact(3) {
        file.write_all(&[3]).map_err(error)?;
        for &index in triangle {
            file.write_all(&(index as i32).to_le_bytes())
                .map_err(error)?;
        }
    }
    file.flush().map_err(error)
}

fn read_mesh(path: &str) -> Result<TriQuadMesh, PlyToolError> {
    TriQuadMesh::read_ply(path).map_err(error)
}

fn single_path(args: &[OsString], label: &str) -> Result<String, PlyToolError> {
    if args.len() != 1 {
        return Err(usage_error(format!("must specify exactly one {label}")));
    }
    path_string(&args[0])
}

fn required_path(value: Option<String>, label: &str) -> Result<String, PlyToolError> {
    value.ok_or_else(|| usage_error(format!("must specify {label}")))
}

fn path_string(value: &OsString) -> Result<String, PlyToolError> {
    value
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| error_message("path is not valid UTF-8"))
}

fn option_string(args: &[OsString], index: &mut usize, name: &str) -> Result<String, PlyToolError> {
    *index += 1;
    let value = args
        .get(*index)
        .ok_or_else(|| usage_error(format!("missing value for --{name}")))?;
    path_string(value)
}

fn parse_option(args: &[OsString], index: &mut usize, name: &str) -> Result<Float, PlyToolError> {
    let value = option_string(args, index, name)?;
    value
        .parse::<Float>()
        .map_err(|_| usage_error(format!("invalid value for --{name}: {value}")))
}

fn parse_option_usize(
    args: &[OsString],
    index: &mut usize,
    name: &str,
) -> Result<usize, PlyToolError> {
    let value = option_string(args, index, name)?;
    value
        .parse::<usize>()
        .map_err(|_| usage_error(format!("invalid value for --{name}: {value}")))
}

fn remove_extension(path: &str) -> String {
    Path::new(path)
        .with_extension("")
        .to_string_lossy()
        .into_owned()
}

// Matches pbrt-v4's Vector2/Vector3 and Bounds3 ToString() formatting.
fn format_point2(point: &Point2f) -> String {
    format!("[ {:.6}, {:.6} ]", point.x, point.y)
}

fn format_point3(point: &Point3f) -> String {
    format!("[ {:.6}, {:.6}, {:.6} ]", point.x, point.y, point.z)
}

fn format_bounds(bounds: &Bounds3f) -> String {
    format!(
        "[ {} - {} ]",
        format_point3(&bounds.min),
        format_point3(&bounds.max)
    )
}

fn error(value: impl Display) -> PlyToolError {
    error_message(value.to_string())
}

fn error_message(message: impl Into<String>) -> PlyToolError {
    PlyToolError::new(message)
}

fn usage_error(message: impl Into<String>) -> PlyToolError {
    PlyToolError::with_help(message)
}

fn run<I, T>(args: I) -> Result<(), PlyToolError>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let mut args = args.into_iter().map(Into::into);
    let _program = args.next();
    let Some(command) = args.next() else {
        print!("{}", help_text());
        return Ok(());
    };
    let command = command
        .into_string()
        .map_err(|_| PlyToolError::new("command is not valid UTF-8"))?;
    let arguments: Vec<OsString> = args.collect();

    match command.as_str() {
        "help" => {
            if arguments.is_empty() {
                print!("{}", help_text());
            } else {
                for command in arguments {
                    let command = command
                        .into_string()
                        .map_err(|_| PlyToolError::new("command is not valid UTF-8"))?;
                    print!("{}", command_help(&command)?);
                }
            }
            Ok(())
        }
        "info" => {
            info(&arguments)?;
            Ok(())
        }
        "cat" => cat(&arguments),
        "displace" => displace(&arguments),
        "split" => split(&arguments),
        _ => Err(PlyToolError::with_help(format!(
            "{command}: command unknown"
        ))),
    }
}

fn main() {
    match run(std::env::args_os()) {
        Ok(()) => std::process::exit(0),
        Err(error) => {
            eprintln!("plytool: {error}");
            if error.show_help() {
                print!("{}", help_text());
            }
            std::process::exit(1);
        }
    }
}
