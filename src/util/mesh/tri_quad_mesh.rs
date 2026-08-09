use crate::util::base::*;
use crate::util::error::*;

use ply_rs::parser;
use ply_rs::ply;
use std::collections::HashMap;
use std::io::BufRead;

const VERTEX_P: u32 = 1;
const VERTEX_N: u32 = 2;
const VERTEX_UV: u32 = 8;

#[derive(Debug, Clone)]
pub struct TriQuadMesh {
    pub p: Vec<Point3f>,
    pub n: Vec<Normal3f>,
    pub uv: Vec<Point2f>,
    pub face_indices: Vec<i32>,
    pub tri_indices: Vec<u32>,
    /// Quad indices are stored in pbrt-v4's bilinear-patch order:
    /// `[p00, p10, p01, p11]`, i.e. PLY perimeter `[v0, v1, v2, v3]`
    /// becomes `[v0, v1, v3, v2]`.
    pub quad_indices: Vec<u32>,
}

#[derive(Debug)]
struct Vertex {
    x: f32,
    y: f32,
    z: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    u: f32,
    v: f32,
    flags: u32,
}

#[derive(Debug)]
struct Face {
    vertex_index: Vec<i32>,
    face_index: Option<i32>,
}

impl ply::PropertyAccess for Vertex {
    fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 0.0,
            u: 0.0,
            v: 0.0,
            flags: 0,
        }
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        if let ply::Property::Float(v) = property {
            match key.as_str() {
                "x" => {
                    self.x = v;
                    self.flags |= VERTEX_P;
                }
                "y" => {
                    self.y = v;
                    self.flags |= VERTEX_P;
                }
                "z" => {
                    self.z = v;
                    self.flags |= VERTEX_P;
                }
                "nx" => {
                    self.nx = v;
                    self.flags |= VERTEX_N;
                }
                "ny" => {
                    self.ny = v;
                    self.flags |= VERTEX_N;
                }
                "nz" => {
                    self.nz = v;
                    self.flags |= VERTEX_N;
                }
                "u" | "s" | "texture_u" | "texture_s" => {
                    self.u = v;
                    self.flags |= VERTEX_UV;
                }
                "v" | "t" | "texture_v" | "texture_t" => {
                    self.v = v;
                    self.flags |= VERTEX_UV;
                }
                _ => {}
            }
        }
    }
}

impl ply::PropertyAccess for Face {
    fn new() -> Self {
        Self {
            vertex_index: Vec::new(),
            face_index: None,
        }
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        if key == "vertex_indices" || key == "vertex_index" {
            match property {
                ply::Property::ListInt(vec) => {
                    self.vertex_index.extend(vec);
                }
                ply::Property::ListUInt(vec) => {
                    self.vertex_index.extend(vec.into_iter().map(|v| v as i32));
                }
                _ => {}
            }
        } else if key == "face_indices" {
            match property {
                ply::Property::Int(v) => self.face_index = Some(v),
                ply::Property::UInt(v) => self.face_index = Some(v as i32),
                _ => {}
            }
        }
    }
}

fn create_reader(filename: &str) -> Result<Box<dyn BufRead>, PbrtError> {
    let path = std::path::PathBuf::from(filename);
    let extent = path
        .extension()
        .ok_or(PbrtError::error("No extension found"))?
        .to_string_lossy()
        .into_owned();

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    if extent == "gz" {
        let reader = flate2::read::GzDecoder::new(reader);
        Ok(Box::new(std::io::BufReader::new(reader)))
    } else {
        Ok(Box::new(reader))
    }
}

impl TriQuadMesh {
    pub fn read_ply(filename: &str) -> Result<Self, PbrtError> {
        let mut reader = create_reader(filename)?;
        let vertex_parser = parser::Parser::<Vertex>::new();
        let face_parser = parser::Parser::<Face>::new();
        let header = vertex_parser.read_header(&mut reader).map_err(|e| {
            PbrtError::error(&format!(
                "plymesh: failed to read header for \"{}\": {}",
                filename, e
            ))
        })?;

        let mut mesh = Self {
            p: Vec::new(),
            n: Vec::new(),
            uv: Vec::new(),
            face_indices: Vec::new(),
            tri_indices: Vec::new(),
            quad_indices: Vec::new(),
        };

        for (_name, element) in header.elements.iter() {
            match element.name.as_ref() {
                "vertex" => {
                    let vertex_list = vertex_parser
                        .read_payload_for_element(&mut reader, element, &header)
                        .map_err(PbrtError::from)?;
                    if vertex_list.is_empty() {
                        continue;
                    }

                    let flags = vertex_list[0].flags;
                    if (flags & VERTEX_P) != 0 {
                        mesh.p.reserve(vertex_list.len());
                        for v in vertex_list.iter() {
                            mesh.p
                                .push(Point3f::new(v.x as Float, v.y as Float, v.z as Float));
                        }
                    }
                    if (flags & VERTEX_N) != 0 {
                        mesh.n.reserve(vertex_list.len());
                        for v in vertex_list.iter() {
                            mesh.n
                                .push(Normal3f::new(v.nx as Float, v.ny as Float, v.nz as Float));
                        }
                    }
                    if (flags & VERTEX_UV) != 0 {
                        mesh.uv.reserve(vertex_list.len());
                        for v in vertex_list.iter() {
                            mesh.uv.push(Point2f::new(v.u as Float, v.v as Float));
                        }
                    }
                }
                "face" => {
                    let face_list = face_parser
                        .read_payload_for_element(&mut reader, element, &header)
                        .map_err(PbrtError::from)?;
                    mesh.tri_indices.reserve(face_list.len() * 3);
                    mesh.quad_indices.reserve(face_list.len() * 4);
                    for face in face_list.into_iter() {
                        if let Some(face_index) = face.face_index {
                            mesh.face_indices.push(face_index);
                        }
                        match face.vertex_index.len() {
                            3 => {
                                mesh.tri_indices
                                    .extend(face.vertex_index.into_iter().map(|idx| idx as u32));
                            }
                            4 => {
                                mesh.quad_indices.push(face.vertex_index[0] as u32);
                                mesh.quad_indices.push(face.vertex_index[1] as u32);
                                mesh.quad_indices.push(face.vertex_index[3] as u32);
                                mesh.quad_indices.push(face.vertex_index[2] as u32);
                            }
                            n_vert => {
                                log::warn!(
                                    "plymesh: Ignoring face with {} vertices (only triangles and quads are supported!)",
                                    n_vert
                                );
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if mesh.p.is_empty() || (mesh.tri_indices.is_empty() && mesh.quad_indices.is_empty()) {
            return Err(PbrtError::error(&format!(
                "{}: PLY file is invalid! No face/vertex elements found!",
                filename
            )));
        }

        mesh.validate_indices()?;
        Ok(mesh)
    }

    pub fn convert_to_only_triangles(&mut self) {
        if self.quad_indices.is_empty() {
            return;
        }

        self.tri_indices
            .reserve(self.tri_indices.len() + 3 * self.quad_indices.len() / 2);
        for quad in self.quad_indices.chunks_exact(4) {
            self.tri_indices.extend([quad[0], quad[1], quad[3]]);
            self.tri_indices.extend([quad[0], quad[3], quad[2]]);
        }
        self.quad_indices.clear();
    }

    pub fn compute_normals(&mut self) {
        self.n = vec![Normal3f::zero(); self.p.len()];
        for tri in self.tri_indices.chunks_exact(3) {
            let v0 = tri[0] as usize;
            let v1 = tri[1] as usize;
            let v2 = tri[2] as usize;
            let v10 = self.p[v1] - self.p[v0];
            let v21 = self.p[v2] - self.p[v1];
            let mut vn = Vector3f::cross(&v10, &v21);
            if vn.length_squared() > 0.0 {
                vn = vn.normalize();
                self.n[v0] += vn;
                self.n[v1] += vn;
                self.n[v2] += vn;
            }
        }

        for n in self.n.iter_mut() {
            if n.length_squared() > 0.0 {
                *n = n.normalize();
            }
        }
    }

    pub fn displace<Dist, Disp>(
        mut self,
        distance: Dist,
        max_dist: Float,
        mut displacement: Disp,
    ) -> Result<Self, PbrtError>
    where
        Dist: Fn(Point3f, Point3f) -> Float,
        Disp: FnMut(Point3f, Normal3f, Point2f) -> Point3f,
    {
        if self.uv.is_empty() {
            return Err(PbrtError::error(
                "Vertex uvs are currently required by Displace(). Sorry.",
            ));
        }
        if max_dist <= 0.0 {
            return Err(PbrtError::error(
                "plymesh: displacement edgelength must be greater than zero",
            ));
        }

        self.convert_to_only_triangles();
        if self.n.is_empty() {
            self.compute_normals();
        }

        let old_tri_indices = std::mem::take(&mut self.tri_indices);
        let mut edge_split = HashMap::new();
        for tri in old_tri_indices.chunks_exact(3) {
            self.refine(&distance, max_dist, tri[0], tri[1], tri[2], &mut edge_split);
        }

        for i in 0..self.p.len() {
            self.p[i] = displacement(self.p[i], self.n[i], self.uv[i]);
        }

        self.compute_normals();
        Ok(self)
    }

    fn validate_indices(&self) -> Result<(), PbrtError> {
        let vertex_count = self.p.len() as u32;
        for idx in self.tri_indices.iter().chain(self.quad_indices.iter()) {
            if *idx >= vertex_count {
                return Err(PbrtError::error(&format!(
                    "plymesh: Vertex index {} is out of bounds! Valid range is [0..{})",
                    idx, vertex_count
                )));
            }
        }
        Ok(())
    }

    fn refine<Dist>(
        &mut self,
        distance: &Dist,
        max_dist: Float,
        v0: u32,
        v1: u32,
        v2: u32,
        edge_split: &mut HashMap<(u32, u32), u32>,
    ) where
        Dist: Fn(Point3f, Point3f) -> Float,
    {
        let p0 = self.p[v0 as usize];
        let p1 = self.p[v1 as usize];
        let p2 = self.p[v2 as usize];
        let d01 = distance(p0, p1);
        let d12 = distance(p1, p2);
        let d20 = distance(p2, p0);

        if d01 < max_dist && d12 < max_dist && d20 < max_dist {
            self.tri_indices.extend([v0, v1, v2]);
            return;
        }

        let v = if d01 > d12 {
            if d01 > d20 {
                [v0, v1, v2]
            } else {
                [v2, v0, v1]
            }
        } else if d12 > d20 {
            [v1, v2, v0]
        } else {
            [v2, v0, v1]
        };

        let edge = if v[0] < v[1] {
            (v[0], v[1])
        } else {
            (v[1], v[0])
        };

        let vmid = if let Some(vmid) = edge_split.get(&edge) {
            *vmid
        } else {
            let vmid = self.p.len() as u32;
            edge_split.insert(edge, vmid);
            self.p
                .push((self.p[v[0] as usize] + self.p[v[1] as usize]) * 0.5);
            if !self.n.is_empty() {
                let mut nn = self.n[v[0] as usize] + self.n[v[1] as usize];
                if nn.length_squared() > 0.0 {
                    nn = nn.normalize();
                }
                self.n.push(nn);
            }
            if !self.uv.is_empty() {
                self.uv
                    .push((self.uv[v[0] as usize] + self.uv[v[1] as usize]) * 0.5);
            }
            vmid
        };

        self.refine(distance, max_dist, v[0], vmid, v[2], edge_split);
        self.refine(distance, max_dist, vmid, v[1], v[2], edge_split);
    }
}
