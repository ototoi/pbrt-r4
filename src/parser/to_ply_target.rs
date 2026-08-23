use super::parse_target::*;
use super::parsed_parameter::{into_parameter_dictionary, parsed_parameters_from_dictionary};
use super::{ParsedParameterValues, ParsedParameterVector};
use crate::paramdict::*;
use crate::util::base::*;
use crate::util::error::*;

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use log::*;

use ply_rs::ply::*;
use ply_rs::writer::Writer;

pub struct ToPlyTarget {
    dir: String,
    context: Arc<RefCell<dyn ParseTarget>>,
    count: usize,
}

impl ToPlyTarget {
    pub fn new(dir: &str, context: Arc<RefCell<dyn ParseTarget>>) -> Self {
        ToPlyTarget {
            dir: dir.to_string(),
            context,
            count: 1,
        }
    }

    pub fn get_ply_filename(&self) -> String {
        let ply_prefix = std::env::var("PLY_PREFIX").unwrap_or("mesh".to_string());
        let count = self.count;
        format!("{}_{:0>5}.ply", ply_prefix, count)
    }
}

#[derive(Debug, Default)]
struct Mesh {
    pub p: Vec<Point3f>,
    pub n: Vec<Vector3f>,
    pub uv: Vec<Point2f>,
    pub vertex_indices: Vec<u32>,
    pub face_indices: Vec<u32>,
}

fn convert_to_mesh(params: &ParameterDictionary) -> Option<Mesh> {
    let mut vertex_indices = Vec::new();
    let mut p: Vec<Vector3f> = Vec::new();
    let mut n: Vec<Vector3f> = Vec::new();
    let mut uv: Vec<Vector2f> = Vec::new();
    let mut face_indices = Vec::new();

    if let Some(vi) = params.get_ints_ref("indices") {
        vertex_indices.resize(vi.len(), 0);
        for i in 0..vi.len() {
            vertex_indices[i] = vi[i] as u32;
        }
    }

    if let Some(ps) = params.get_points_ref("P") {
        let sz = ps.len() / 3;
        p.resize(sz, Point3f::zero());
        for i in 0..sz {
            p[i] = Point3f::new(ps[3 * i + 0], ps[3 * i + 1], ps[3 * i + 2]);
        }
    }

    let uv_points = params.get_point2f_array("uv");
    if !uv_points.is_empty() {
        uv = uv_points;
    } else {
        let st_points = params.get_point2f_array("st");
        if !st_points.is_empty() {
            uv = st_points;
        }
    }

    if let Some(ps) = params.get_points_ref("N") {
        let sz = ps.len() / 3;
        n.resize(sz, Normal3f::zero());
        for i in 0..sz {
            n[i] = Normal3f::new(ps[3 * i + 0], ps[3 * i + 1], ps[3 * i + 2]);
        }
    }

    if let Some(vi) = params.get_ints_ref("faceIndices") {
        face_indices.resize(vi.len(), 0);
        for i in 0..vi.len() {
            face_indices[i] = vi[i] as u32;
        }
    }

    if !p.is_empty() && !vertex_indices.is_empty() {
        let mesh = Mesh {
            p,
            n,
            uv,
            vertex_indices,
            face_indices,
        };
        return Some(mesh);
    } else {
        return None;
    }
}

fn write_mesh_to_ply(mesh: &Mesh, file_name: &Path) -> Result<(), PbrtError> {
    let p = &mesh.p;
    let n = &mesh.n;
    let uv = &mesh.uv;

    let vertex_indices = &mesh.vertex_indices;
    let face_indices = &mesh.face_indices;

    let vertex_count = p.len();
    let face_count = vertex_indices.len() / 3;
    let mut header = Header::new();
    header.encoding = Encoding::BinaryLittleEndian;
    {
        let mut element = ElementDef::new("vertex".to_string());
        element.count = vertex_count;
        element.properties.insert(
            "x".to_string(),
            PropertyDef::new("x".to_string(), PropertyType::Scalar(ScalarType::Float)),
        );
        element.properties.insert(
            "y".to_string(),
            PropertyDef::new("y".to_string(), PropertyType::Scalar(ScalarType::Float)),
        );
        element.properties.insert(
            "z".to_string(),
            PropertyDef::new("z".to_string(), PropertyType::Scalar(ScalarType::Float)),
        );
        if n.len() > 0 {
            element.properties.insert(
                "nx".to_string(),
                PropertyDef::new("nx".to_string(), PropertyType::Scalar(ScalarType::Float)),
            );
            element.properties.insert(
                "ny".to_string(),
                PropertyDef::new("ny".to_string(), PropertyType::Scalar(ScalarType::Float)),
            );
            element.properties.insert(
                "nz".to_string(),
                PropertyDef::new("nz".to_string(), PropertyType::Scalar(ScalarType::Float)),
            );
        }
        if uv.len() > 0 {
            element.properties.insert(
                "u".to_string(),
                PropertyDef::new("u".to_string(), PropertyType::Scalar(ScalarType::Float)),
            );
            element.properties.insert(
                "v".to_string(),
                PropertyDef::new("v".to_string(), PropertyType::Scalar(ScalarType::Float)),
            );
        }
        header.elements.insert("vertex".to_string(), element);
    }
    {
        let mut element = ElementDef::new("face".to_string());
        element.count = face_count;
        element.properties.insert(
            "vertex_indices".to_string(),
            PropertyDef::new(
                "vertex_indices".to_string(),
                PropertyType::List(ScalarType::UChar, ScalarType::Int),
            ),
        );
        if face_indices.len() > 0 {
            element.properties.insert(
                "face_indices".to_string(),
                PropertyDef::new(
                    "face_indices".to_string(),
                    PropertyType::Scalar(ScalarType::Int),
                ),
            );
        }
        header.elements.insert("face".to_string(), element);
    }

    let mut ply = Ply::<DefaultElement>::new();
    ply.header = header;
    {
        let mut vertices = Vec::new();
        for i in 0..vertex_count {
            let mut vertex = DefaultElement::new();
            vertex.set_property("x".to_string(), Property::Float(p[i].x as f32));
            vertex.set_property("y".to_string(), Property::Float(p[i].y as f32));
            vertex.set_property("z".to_string(), Property::Float(p[i].z as f32));
            if n.len() > 0 {
                vertex.set_property("nx".to_string(), Property::Float(n[i].x as f32));
                vertex.set_property("ny".to_string(), Property::Float(n[i].y as f32));
                vertex.set_property("nz".to_string(), Property::Float(n[i].z as f32));
            }
            if uv.len() > 0 {
                vertex.set_property("u".to_string(), Property::Float(uv[i].x as f32));
                vertex.set_property("v".to_string(), Property::Float(uv[i].y as f32));
            }
            vertices.push(vertex);
        }
        ply.payload.insert("vertex".to_string(), vertices);
    }
    {
        let mut faces = Vec::new();
        for i in 0..face_count {
            let mut face = DefaultElement::new();
            face.set_property(
                "vertex_indices".to_string(),
                Property::ListInt(vec![
                    vertex_indices[3 * i + 0] as i32,
                    vertex_indices[3 * i + 1] as i32,
                    vertex_indices[3 * i + 2] as i32,
                ]),
            );
            if face_indices.len() > 0 {
                face.set_property(
                    "face_indices".to_string(),
                    Property::Int(face_indices[i] as i32),
                );
            }
            faces.push(face);
        }
        ply.payload.insert("face".to_string(), faces);
    }

    let mut buf = std::fs::File::create(file_name)?;
    let writer = Writer::new();
    writer.write_ply(&mut buf, &mut ply)?;
    Ok(())
}

fn create_plymesh_params(params: &ParameterDictionary) -> ParameterDictionary {
    let mut p = ParameterDictionary::new();
    let keys = params.get_keys();
    for key in keys {
        let keyname = params.get_key_name(&key);
        match keyname.as_str() {
            "indices" | "P" | "uv" | "st" | "S" | "N" | "faceIndices" => {
                continue;
            }
            _ => {
                let key = key.as_str();
                if let Some(v) = params.get_bools_ref(key) {
                    p.add_bools(key, &v);
                } else if let Some(v) = params.get_ints_ref(key) {
                    p.add_ints(key, &v);
                } else if let Some(v) = params.get_floats_ref(key) {
                    p.add_floats(key, &v);
                } else if let Some(v) = params.get_points_ref(key) {
                    p.add_floats(key, &v);
                } else if let Some(v) = params.get_strings_ref(key) {
                    let v = v.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
                    p.add_strings(key, &v);
                } else if let Some(v) = params.get_textures_ref(key) {
                    let v = v.iter().map(|x| x.as_str()).collect::<Vec<&str>>();
                    p.add_strings(key, &v);
                } else {
                    warn!("Unsupported type for key: {}", key);
                }
            }
        }
    }
    return p;
}

impl ParseTarget for ToPlyTarget {
    fn cleanup(&mut self) {
        self.context.borrow_mut().cleanup();
    }

    fn identity(&mut self) {
        self.context.borrow_mut().identity();
    }

    fn translate(&mut self, dx: Float, dy: Float, dz: Float) {
        self.context.borrow_mut().translate(dx, dy, dz);
    }

    fn rotate(&mut self, angle: Float, ax: Float, ay: Float, az: Float) {
        self.context.borrow_mut().rotate(angle, ax, ay, az);
    }

    fn scale(&mut self, sx: Float, sy: Float, sz: Float) {
        self.context.borrow_mut().scale(sx, sy, sz);
    }

    fn look_at(
        &mut self,
        ex: Float,
        ey: Float,
        ez: Float,
        lx: Float,
        ly: Float,
        lz: Float,
        ux: Float,
        uy: Float,
        uz: Float,
    ) {
        self.context
            .borrow_mut()
            .look_at(ex, ey, ez, lx, ly, lz, ux, uy, uz);
    }

    fn concat_transform(&mut self, transform: &[Float]) {
        self.context.borrow_mut().concat_transform(transform);
    }

    fn transform(&mut self, transform: &[Float]) {
        self.context.borrow_mut().transform(transform);
    }

    fn color_space(&mut self, name: &str) {
        self.context.borrow_mut().color_space(name);
    }

    fn option(&mut self, name: &str, value: &str) {
        self.context.borrow_mut().option(name, value);
    }

    fn attribute(&mut self, target: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().attribute(target, params);
    }

    fn coordinate_system(&mut self, name: &str) {
        self.context.borrow_mut().coordinate_system(name);
    }

    fn coord_sys_transform(&mut self, name: &str) {
        self.context.borrow_mut().coord_sys_transform(name);
    }

    fn active_transform_all(&mut self) {
        self.context.borrow_mut().active_transform_all();
    }

    fn active_transform_end_time(&mut self) {
        self.context.borrow_mut().active_transform_end_time();
    }

    fn active_transform_start_time(&mut self) {
        self.context.borrow_mut().active_transform_start_time();
    }

    fn transform_times(&mut self, start: Float, end: Float) {
        self.context.borrow_mut().transform_times(start, end);
    }

    fn pixel_filter(&mut self, name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().pixel_filter(name, params);
    }

    fn film(&mut self, name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().film(name, params);
    }

    fn sampler(&mut self, name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().sampler(name, params);
    }

    fn accelerator(&mut self, name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().accelerator(name, params);
    }

    fn integrator(&mut self, name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().integrator(name, params);
    }
    fn camera(&mut self, name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().camera(name, params);
    }

    fn make_named_medium(&mut self, name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().make_named_medium(name, params);
    }

    fn medium_interface(&mut self, inside_name: &str, outside_name: &str) {
        self.context
            .borrow_mut()
            .medium_interface(inside_name, outside_name);
    }

    fn world_begin(&mut self) {
        self.context.borrow_mut().world_begin();
    }

    fn attribute_begin(&mut self) {
        self.context.borrow_mut().attribute_begin();
    }

    fn attribute_end(&mut self) {
        self.context.borrow_mut().attribute_end();
    }

    fn transform_begin(&mut self) {
        self.context.borrow_mut().transform_begin();
    }

    fn transform_end(&mut self) {
        self.context.borrow_mut().transform_end();
    }

    fn texture(&mut self, name: &str, t: &str, tex_name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().texture(name, t, tex_name, params);
    }

    fn material(&mut self, name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().material(name, params);
    }

    fn make_named_material(&mut self, name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().make_named_material(name, params);
    }

    fn named_material(&mut self, name: &str) {
        self.context.borrow_mut().named_material(name);
    }

    fn light_source(&mut self, name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().light_source(name, params);
    }

    fn area_light_source(&mut self, name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().area_light_source(name, params);
    }

    fn shape(&mut self, name: &str, params: ParsedParameterVector) {
        if name != "trianglemesh" {
            self.context.borrow_mut().shape(name, params);
            return;
        }

        let has_small_index_buffer = params.iter().any(|parameter| {
            parameter.name == "indices"
                && matches!(&parameter.values, ParsedParameterValues::Ints(values) if values.len() < 500)
        });
        if has_small_index_buffer {
            self.context.borrow_mut().shape(name, params);
            return;
        }

        let params = into_parameter_dictionary(params);
        if let Some(mesh) = convert_to_mesh(&params) {
            let dir = Path::new(&self.dir).join("geometry");
            if let Err(e) = std::fs::create_dir_all(&dir) {
                error!("Error: {}", e);
                return;
            }
            let filename = self.get_ply_filename();
            let filepath = dir.join(&filename);
            {
                // Check if the mesh has tangent vectors
                if params.get_points_ref("S").is_some() {
                    warn!(
                        "{}: PLY mesh will be missing tangent vectors \"S\".",
                        filename
                    );
                }
            }
            match write_mesh_to_ply(&mesh, &filepath) {
                Ok(_) => {
                    let mut params = create_plymesh_params(&params);
                    let filepath = filepath
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "mesh.ply".to_string());
                    let filepath = Path::new("geometry").join(filepath);
                    let filepath = filepath.to_string_lossy().into_owned();
                    params.add_string("string filename", &filepath);
                    self.context
                        .borrow_mut()
                        .shape("plymesh", parsed_parameters_from_dictionary(&params));
                }
                Err(e) => {
                    error!("Error: {}", e);
                }
            }
        }
        self.count += 1;
    }

    fn reverse_orientation(&mut self) {
        self.context.borrow_mut().reverse_orientation();
    }

    fn object_begin(&mut self, name: &str) {
        self.context.borrow_mut().object_begin(name);
    }

    fn object_end(&mut self) {
        self.context.borrow_mut().object_end();
    }
    fn object_instance(&mut self, name: &str) {
        self.context.borrow_mut().object_instance(name);
    }

    fn world_end(&mut self) {
        self.context.borrow_mut().world_end();
    }

    fn parse_file(&mut self, file_name: &str) {
        self.context.borrow_mut().parse_file(file_name);
    }

    fn parse_string(&mut self, s: &str) {
        self.context.borrow_mut().parse_string(s);
    }

    fn work_dir_begin(&mut self, path: &str) {
        self.context.borrow_mut().work_dir_begin(path);
    }

    fn work_dir_end(&mut self) {
        self.context.borrow_mut().work_dir_end();
    }

    fn include(&mut self, file_name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().include(file_name, params);
    }

    fn import(&mut self, file_name: &str, params: ParsedParameterVector) {
        self.context.borrow_mut().import(file_name, params);
    }
}
