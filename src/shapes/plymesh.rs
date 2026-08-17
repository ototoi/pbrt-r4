use super::alphamask::AlphaMaskShape;
use super::bilinearmesh::create_bilinear_patch_mesh;
use super::triangle::*;
use crate::base::shape::Shape;
use crate::interaction::SurfaceInteraction;
use crate::options::PbrtOptions;
use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::mesh::TriQuadMesh;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

use log::warn;
use std::collections::HashMap;
use std::sync::Arc;

type FloatTextureMap = HashMap<String, Arc<FloatTexture>>;

fn memory_probe_ply_stage(stage: &str, filename: &str) {
    if std::env::var_os("PBRT_MEMORY_PROFILE").is_none() {
        return;
    }
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let rss_kb = status
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:")
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|value| value.parse::<u64>().ok())
        })
        .unwrap_or(0);
    eprintln!(
        "[MEM-INVESTIGATION] ply-stage={} rss_kb={} file={}",
        stage, rss_kb, filename
    );
}

pub struct PlyMesh;

pub fn create_ply_mesh(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    params: &ParameterDictionary,
    float_textures: &FloatTextureMap,
) -> Result<Vec<Shape>, PbrtError> {
    let filename = params.get_one_string("filename", "");
    let mut tri_quad_mesh = TriQuadMesh::read_ply(&filename)?;
    memory_probe_ply_stage("read", &filename);

    let edge_length =
        params.get_one_float("edgelength", 1.0) * PbrtOptions::get().displacement_edge_scale;

    if let Some(displacement_name) = params
        .get_textures_ref("displacement")
        .and_then(|textures| textures.first().cloned())
    {
        let displacement = float_textures.get(&displacement_name).ok_or_else(|| {
            PbrtError::error(&format!("{}: no such texture defined.", displacement_name))
        })?;

        tri_quad_mesh = tri_quad_mesh.displace(
            |p0, p1| {
                let p0 = o2w.transform_point(&p0);
                let p1 = o2w.transform_point(&p1);
                Vector3f::distance(&p0, &p1)
            },
            edge_length,
            |p, n, uv| {
                let mut si = SurfaceInteraction::default();
                si.p = p;
                si.n = n;
                si.uv = uv;
                let ctx = TextureEvalContext::from(&si);
                let d = displacement.evaluate(&ctx);
                p + d * n
            },
        )?;
    }

    let mut mesh: Vec<Shape> = Vec::new();
    if !tri_quad_mesh.tri_indices.is_empty() {
        memory_probe_ply_stage("before-triangle-clone", &filename);
        let tris = create_triangle_mesh(
            o2w,
            w2o,
            reverse_orientation,
            tri_quad_mesh.tri_indices.clone(),
            tri_quad_mesh.p.clone(),
            Vec::new(),
            tri_quad_mesh.n.clone(),
            tri_quad_mesh.uv.clone(),
            params,
        )?;
        memory_probe_ply_stage("after-triangle-create", &filename);
        mesh.extend(tris.into_iter().map(Shape::Triangle));
    }
    if !tri_quad_mesh.quad_indices.is_empty() {
        let patches = create_bilinear_patch_mesh(
            o2w,
            w2o,
            reverse_orientation,
            tri_quad_mesh.quad_indices,
            tri_quad_mesh.p,
            tri_quad_mesh.n,
            tri_quad_mesh.uv,
            tri_quad_mesh.face_indices,
        )?;
        mesh.extend(patches.into_iter().map(Shape::BilinearPatch));
    }
    if mesh.is_empty() {
        warn!(
            "plymesh: no non-degenerate triangles or quads were created for \"{}\"; skipping shape",
            filename
        );
        return Ok(mesh);
    }
    let alpha_mask_info = get_alpha_texture(params, float_textures)?;
    let shadow_alpha_mask_info = get_shadow_alpha_texture(params, float_textures)?;
    if alpha_mask_info.is_some() || shadow_alpha_mask_info.is_some() {
        return Ok(mesh
            .into_iter()
            .map(|shape| {
                let shape = Arc::new(shape);
                Shape::AlphaMask(Box::new(AlphaMaskShape::new(
                    &shape,
                    &alpha_mask_info,
                    &shadow_alpha_mask_info,
                )))
            })
            .collect());
    }

    Ok(mesh)
}

impl PlyMesh {
    pub fn create(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        params: &ParameterDictionary,
        float_textures: &FloatTextureMap,
    ) -> Result<Vec<Shape>, PbrtError> {
        create_ply_mesh(o2w, w2o, reverse_orientation, params, float_textures)
    }
}
