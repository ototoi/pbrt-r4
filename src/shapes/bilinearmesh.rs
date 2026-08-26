use super::alphamask::AlphaMaskShape;
use super::triangle::{get_alpha_texture, get_shadow_alpha_texture};
use crate::base::shape::{Shape, ShapeSampleContext};
use crate::interaction::*;
use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::lowdiscrepancy::*;
use crate::util::sampling;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

use std::collections::HashMap;
use std::sync::Arc;

type FloatTextureMap = HashMap<String, Arc<FloatTexture>>;

pub struct BilinearPatchMesh;

struct BilinearPatchMeshData {
    world_to_object: Transform,
    reverse_orientation: bool,
    swaps_handedness: bool,
    p: Vec<Point3f>,
    n: Vec<Normal3f>,
    uv: Vec<Point2f>,
    vertex_indices: Vec<u32>,
    face_indices: Vec<i32>,
}

#[derive(Clone)]
pub struct BilinearPatch {
    mesh: Arc<BilinearPatchMeshData>,
    patch_index: usize,
    area: Float,
}

impl BilinearPatchMesh {
    pub fn create(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        params: &ParameterDictionary,
        float_textures: &FloatTextureMap,
    ) -> Result<Vec<Shape>, PbrtError> {
        create_bilinear_mesh_shape(o2w, w2o, reverse_orientation, params, float_textures)
    }
}

fn lerp_point2(t: Float, a: Point2f, b: Point2f) -> Point2f {
    a * (1.0 - t) + b * t
}

fn lerp_point3(t: Float, a: Point3f, b: Point3f) -> Point3f {
    a * (1.0 - t) + b * t
}

fn lerp_normal3(t: Float, a: Normal3f, b: Normal3f) -> Normal3f {
    a * (1.0 - t) + b * t
}

fn determinant3(c0: Vector3f, c1: Vector3f, c2: Vector3f) -> Float {
    Vector3f::dot(&c0, &Vector3f::cross(&c1, &c2))
}

fn rotate_from_to_vector(from: Vector3f, to: Vector3f, x: Vector3f) -> Vector3f {
    let from_len2 = from.length_squared();
    let to_len2 = to.length_squared();
    if from_len2 == 0.0 || to_len2 == 0.0 {
        return x;
    }

    let from = from.normalize();
    let to = to.normalize();
    if (from - to).length_squared() < 1e-12 {
        return x;
    }

    let refl = if Float::abs(from.x) < 0.72 && Float::abs(to.x) < 0.72 {
        Vector3f::new(1.0, 0.0, 0.0)
    } else if Float::abs(from.y) < 0.72 && Float::abs(to.y) < 0.72 {
        Vector3f::new(0.0, 1.0, 0.0)
    } else {
        Vector3f::new(0.0, 0.0, 1.0)
    };

    let u = refl - from;
    let v = refl - to;
    let uu = Vector3f::dot(&u, &u);
    let vv = Vector3f::dot(&v, &v);
    if uu == 0.0 || vv == 0.0 {
        return x;
    }
    let uv = Vector3f::dot(&u, &v);

    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] =
                if i == j { 1.0 } else { 0.0 } - 2.0 / uu * u[i] * u[j] - 2.0 / vv * v[i] * v[j]
                    + 4.0 * uv / (uu * vv) * v[i] * u[j];
        }
    }

    Vector3f::new(
        r[0][0] * x.x + r[0][1] * x.y + r[0][2] * x.z,
        r[1][0] * x.x + r[1][1] * x.y + r[1][2] * x.z,
        r[2][0] * x.x + r[2][1] * x.y + r[2][2] * x.z,
    )
}

fn sample_linear(u: Float, a: Float, b: Float) -> Float {
    if u == 0.0 && a == 0.0 {
        return 0.0;
    }
    let denom = a + Float::sqrt(lerp(u, a * a, b * b));
    if denom == 0.0 {
        return 0.0;
    }
    Float::min(u * (a + b) / denom, ONE_MINUS_EPSILON)
}

fn bilinear_pdf(p: Point2f, w: [Float; 4]) -> Float {
    if p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 {
        return 0.0;
    }
    let w_sum = w.iter().sum::<Float>();
    if w_sum == 0.0 {
        return 1.0;
    }
    4.0 * (((1.0 - p.x) * (1.0 - p.y) * w[0])
        + (p.x * (1.0 - p.y) * w[1])
        + ((1.0 - p.x) * p.y * w[2])
        + (p.x * p.y * w[3]))
        / w_sum
}

fn sample_bilinear(u: Point2f, w: [Float; 4]) -> Point2f {
    let y = sample_linear(u.y, w[0] + w[1], w[2] + w[3]);
    let x = sample_linear(u.x, lerp(y, w[0], w[2]), lerp(y, w[1], w[3]));
    Point2f::new(x, y)
}

fn solve_bilinear_u(a: Float, b: Float, c: Float) -> Option<(Float, Float)> {
    if Float::abs(a) < 1e-12 {
        if Float::abs(b) < 1e-12 {
            return None;
        }
        let u = -c / b;
        return Some((u, u));
    }
    let (mut u0, mut u1) = quadratic(a, b, c)?;
    if u0 > u1 {
        std::mem::swap(&mut u0, &mut u1);
    }
    Some((u0, u1))
}

fn interpolate_uv(uv: Point2f, p00: Point2f, p10: Point2f, p01: Point2f, p11: Point2f) -> Point2f {
    lerp_point2(
        uv.x,
        lerp_point2(uv.y, p00, p01),
        lerp_point2(uv.y, p10, p11),
    )
}

fn interpolate_p(uv: Point2f, p00: Point3f, p10: Point3f, p01: Point3f, p11: Point3f) -> Point3f {
    lerp_point3(
        uv.x,
        lerp_point3(uv.y, p00, p01),
        lerp_point3(uv.y, p10, p11),
    )
}

fn interpolate_n(
    uv: Point2f,
    n00: Normal3f,
    n10: Normal3f,
    n01: Normal3f,
    n11: Normal3f,
) -> Normal3f {
    lerp_normal3(
        uv.x,
        lerp_normal3(uv.y, n00, n01),
        lerp_normal3(uv.y, n10, n11),
    )
}

fn patch_dpdu(uv: Point2f, p00: Point3f, p10: Point3f, p01: Point3f, p11: Point3f) -> Vector3f {
    lerp_point3(uv.y, p10, p11) - lerp_point3(uv.y, p00, p01)
}

fn patch_dpdv(uv: Point2f, p00: Point3f, p10: Point3f, p01: Point3f, p11: Point3f) -> Vector3f {
    lerp_point3(uv.x, p01, p11) - lerp_point3(uv.x, p00, p10)
}

fn validate_bilinear_mesh_params(
    vertex_indices: &[u32],
    p: &[Point3f],
    n: &[Normal3f],
    uv: &[Point2f],
    face_indices: &[i32],
) -> Result<(), PbrtError> {
    if vertex_indices.len() % 4 != 0 {
        return Err(PbrtError::from(
            "Invalid bilinearmesh: indices length must be a multiple of 4",
        ));
    }
    if p.is_empty() {
        return Err(PbrtError::from("Invalid bilinearmesh: missing P"));
    }
    let vertex_count = p.len() as u32;
    if vertex_indices.iter().any(|&i| i >= vertex_count) {
        return Err(PbrtError::from(
            "Invalid bilinearmesh: index out of range for provided P",
        ));
    }
    if !n.is_empty() && n.len() != p.len() {
        return Err(PbrtError::from(
            "Invalid bilinearmesh: N must be empty or have the same length as P",
        ));
    }
    if !uv.is_empty() && uv.len() != p.len() {
        return Err(PbrtError::from(
            "Invalid bilinearmesh: uv must be empty or have the same length as P",
        ));
    }
    if !face_indices.is_empty() && face_indices.len() != vertex_indices.len() / 4 {
        return Err(PbrtError::from(
            "Invalid bilinearmesh: faceIndices length must match patch count",
        ));
    }
    Ok(())
}

pub fn create_bilinear_patch_mesh(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    vertex_indices: Vec<u32>,
    p: Vec<Point3f>,
    n: Vec<Normal3f>,
    uv: Vec<Point2f>,
    face_indices: Vec<i32>,
) -> Result<Vec<BilinearPatch>, PbrtError> {
    validate_bilinear_mesh_params(&vertex_indices, &p, &n, &uv, &face_indices)?;

    let p = p
        .iter()
        .map(|p| o2w.transform_point(p))
        .collect::<Vec<Point3f>>();
    let n = n
        .iter()
        .map(|n| {
            let mut nn = o2w.transform_normal(n);
            if reverse_orientation {
                nn = -nn;
            }
            nn
        })
        .collect::<Vec<Normal3f>>();
    let mesh = Arc::new(BilinearPatchMeshData {
        world_to_object: *w2o,
        reverse_orientation,
        swaps_handedness: o2w.swaps_handedness(),
        p,
        n,
        uv,
        vertex_indices,
        face_indices,
    });

    let mut patches = Vec::with_capacity(mesh.vertex_indices.len() / 4);
    for patch_index in 0..(mesh.vertex_indices.len() / 4) {
        patches.push(BilinearPatch::new(&mesh, patch_index));
    }
    Ok(patches)
}

impl BilinearPatch {
    const MIN_SPHERICAL_SAMPLE_AREA: Float = 1e-4;

    fn new(mesh: &Arc<BilinearPatchMeshData>, patch_index: usize) -> Self {
        // pbrt-v4 `BilinearPatch::CreatePatches` (shapes.cpp) keeps every
        // patch in the mesh and lets `IntersectBilinearPatch` cope with
        // near-degenerate quads. Filtering on `area <= 1e-16` here
        // creates holes in plymesh-derived floors (e.g. sportscar's
        // `Plane_003_0000_m000.ply` 4-vertex faces) — rays passing
        // through the hole fall back to env light and the GBuffer
        // accumulator records `vs.set = false`, leaving stripes of
        // `uv = (0, 0)` (blue in the UV PNG) and `albedo = 0`.
        let area = Self::approximate_area(mesh, patch_index);
        BilinearPatch {
            mesh: Arc::clone(mesh),
            patch_index,
            area,
        }
    }

    fn vertex_indices(&self) -> [u32; 4] {
        let i = 4 * self.patch_index;
        [
            self.mesh.vertex_indices[i],
            self.mesh.vertex_indices[i + 1],
            self.mesh.vertex_indices[i + 2],
            self.mesh.vertex_indices[i + 3],
        ]
    }

    fn control_points(&self) -> (Point3f, Point3f, Point3f, Point3f) {
        let v = self.vertex_indices();
        (
            self.mesh.p[v[0] as usize],
            self.mesh.p[v[1] as usize],
            self.mesh.p[v[2] as usize],
            self.mesh.p[v[3] as usize],
        )
    }

    fn face_index(&self) -> u32 {
        self.mesh
            .face_indices
            .get(self.patch_index)
            .copied()
            .unwrap_or(0) as u32
    }

    fn is_rectangle_points(p00: Point3f, p10: Point3f, p01: Point3f, p11: Point3f) -> bool {
        if p00 == p01 || p01 == p11 || p11 == p10 || p10 == p00 {
            return false;
        }

        let n = Vector3f::cross(&(p10 - p00), &(p01 - p00)).normalize();
        if Float::abs(Vector3f::dot(&(p11 - p00).normalize(), &n)) > 1e-5 {
            return false;
        }

        let center = (p00 + p01 + p10 + p11) / 4.0;
        let d0 = Point3f::distance_squared(&p00, &center);
        let d1 = Point3f::distance_squared(&p01, &center);
        let d2 = Point3f::distance_squared(&p10, &center);
        let d3 = Point3f::distance_squared(&p11, &center);
        if d0 == 0.0 {
            return false;
        }
        for di in [d1, d2, d3] {
            if Float::abs(di - d0) / d0 > 1e-4 {
                return false;
            }
        }
        true
    }

    fn is_rectangle(&self) -> bool {
        let (p00, p10, p01, p11) = self.control_points();
        Self::is_rectangle_points(p00, p10, p01, p11)
    }

    fn approximate_area(mesh: &BilinearPatchMeshData, patch_index: usize) -> Float {
        let i = 4 * patch_index;
        let v = [
            mesh.vertex_indices[i] as usize,
            mesh.vertex_indices[i + 1] as usize,
            mesh.vertex_indices[i + 2] as usize,
            mesh.vertex_indices[i + 3] as usize,
        ];
        let p00 = mesh.p[v[0]];
        let p10 = mesh.p[v[1]];
        let p01 = mesh.p[v[2]];
        let p11 = mesh.p[v[3]];

        if Self::is_rectangle_points(p00, p10, p01, p11) {
            return Point3f::distance(&p00, &p01) * Point3f::distance(&p00, &p10);
        }

        let mut p = [[Point3f::zero(); 4]; 4];
        for (i, row) in p.iter_mut().enumerate() {
            let u = i as Float / 3.0;
            for (j, pij) in row.iter_mut().enumerate() {
                let v = j as Float / 3.0;
                *pij = interpolate_p(Point2f::new(u, v), p00, p10, p01, p11);
            }
        }

        let mut area = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                area += 0.5
                    * Vector3f::cross(&(p[i + 1][j + 1] - p[i][j]), &(p[i + 1][j] - p[i][j + 1]))
                        .length();
            }
        }
        area
    }

    fn intersect_patch(
        ray: &Ray,
        t_max: Float,
        p00: Point3f,
        p10: Point3f,
        p01: Point3f,
        p11: Point3f,
    ) -> Option<(Point2f, Float)> {
        let a = Vector3f::dot(&Vector3f::cross(&(p10 - p00), &(p01 - p11)), &ray.d);
        let c = Vector3f::dot(&Vector3f::cross(&(p00 - ray.o), &ray.d), &(p01 - p00));
        let b = Vector3f::dot(&Vector3f::cross(&(p10 - ray.o), &ray.d), &(p11 - p10)) - (a + c);

        let (u0, u1) = solve_bilinear_u(a, b, c)?;
        let eps = gamma(30.0)
            * (max_component(&ray.o.abs())
                + max_component(&ray.d.abs())
                + max_component(&p00.abs())
                + max_component(&p10.abs())
                + max_component(&p01.abs())
                + max_component(&p11.abs()));

        let mut best: Option<(Point2f, Float)> = None;
        for (root_index, u) in [u0, u1].into_iter().enumerate() {
            if root_index == 1 && u == u0 {
                continue;
            }
            if !(0.0..=1.0).contains(&u) {
                continue;
            }

            let uo = lerp_point3(u, p00, p10);
            let ud = lerp_point3(u, p01, p11) - uo;
            let deltao = uo - ray.o;
            let perp = Vector3f::cross(&ray.d, &ud);
            let p2 = perp.length_squared();
            if p2 == 0.0 {
                continue;
            }

            let v_num = determinant3(deltao, ray.d, perp);
            let t_num = determinant3(deltao, ud, perp);
            if t_num <= p2 * eps || v_num < 0.0 || v_num > p2 {
                continue;
            }

            let t = t_num / p2;
            if t <= eps || t >= t_max {
                continue;
            }
            let v = v_num / p2;
            if best.as_ref().map(|(_, best_t)| t < *best_t).unwrap_or(true) {
                best = Some((Point2f::new(u, v), t));
            }
        }
        best
    }

    fn interaction_from_uv(&self, uv: Point2f, time: Float, wo: Vector3f) -> SurfaceInteraction {
        let (p00, p10, p01, p11) = self.control_points();
        let p = interpolate_p(uv, p00, p10, p01, p11);
        let mut dpdu = patch_dpdu(uv, p00, p10, p01, p11);
        let mut dpdv = patch_dpdv(uv, p00, p10, p01, p11);
        let mut st = uv;
        let mut duds = 1.0;
        let mut dudt = 0.0;
        let mut dvds = 0.0;
        let mut dvdt = 1.0;

        let vtx = self.vertex_indices();
        if !self.mesh.uv.is_empty() {
            let uv00 = self.mesh.uv[vtx[0] as usize];
            let uv10 = self.mesh.uv[vtx[1] as usize];
            let uv01 = self.mesh.uv[vtx[2] as usize];
            let uv11 = self.mesh.uv[vtx[3] as usize];
            st = interpolate_uv(uv, uv00, uv10, uv01, uv11);

            let dstdu = lerp_point2(uv.y, uv10, uv11) - lerp_point2(uv.y, uv00, uv01);
            let dstdv = lerp_point2(uv.x, uv01, uv11) - lerp_point2(uv.x, uv00, uv10);
            duds = if Float::abs(dstdu.x) < 1e-8 {
                0.0
            } else {
                1.0 / dstdu.x
            };
            dvds = if Float::abs(dstdv.x) < 1e-8 {
                0.0
            } else {
                1.0 / dstdv.x
            };
            dudt = if Float::abs(dstdu.y) < 1e-8 {
                0.0
            } else {
                1.0 / dstdu.y
            };
            dvdt = if Float::abs(dstdv.y) < 1e-8 {
                0.0
            } else {
                1.0 / dstdv.y
            };

            let dpds = dpdu * duds + dpdv * dvds;
            let mut dpdt = dpdu * dudt + dpdv * dvdt;
            if Vector3f::cross(&dpds, &dpdt).length_squared() > 0.0 {
                if Vector3f::dot(
                    &Vector3f::cross(&dpdu, &dpdv),
                    &Vector3f::cross(&dpds, &dpdt),
                ) < 0.0
                {
                    dpdt = -dpdt;
                }
                dpdu = dpds;
                dpdv = dpdt;
            }
        }

        let d2pduu = Vector3f::zero();
        let d2pdvv = Vector3f::zero();
        let d2pduv = (p00 - p01) + (p11 - p10);
        let e = Vector3f::dot(&dpdu, &dpdu);
        let f = Vector3f::dot(&dpdu, &dpdv);
        let g = Vector3f::dot(&dpdv, &dpdv);
        let n_geom = Vector3f::cross(&dpdu, &dpdv).normalize();
        let ee = Vector3f::dot(&n_geom, &d2pduu);
        let ff = Vector3f::dot(&n_geom, &d2pduv);
        let gg = Vector3f::dot(&n_geom, &d2pdvv);
        let egf2 = e * g - f * f;
        let inv_egf2 = if egf2 == 0.0 { 0.0 } else { 1.0 / egf2 };
        let dndu_geom =
            ((ff * f - ee * g) * inv_egf2) * dpdu + ((ee * f - ff * e) * inv_egf2) * dpdv;
        let dndv_geom =
            ((gg * f - ff * g) * inv_egf2) * dpdu + ((ff * f - gg * e) * inv_egf2) * dpdv;
        let dnds_geom = dndu_geom * duds + dndv_geom * dvds;
        let dndt_geom = dndu_geom * dudt + dndv_geom * dvdt;

        let mut n = n_geom;
        if self.mesh.reverse_orientation ^ self.mesh.swaps_handedness {
            n = -n;
        }
        let p_error = gamma(6.0) * (p00.abs() + p01.abs() + p10.abs() + p11.abs());
        let mut isect = SurfaceInteraction::new(
            &p,
            &p_error,
            &st,
            &wo,
            &n,
            &dpdu,
            &dpdv,
            &dnds_geom,
            &dndt_geom,
            time,
            self.face_index(),
        );
        isect.shading.n = n;

        if !self.mesh.n.is_empty() {
            let n00 = self.mesh.n[vtx[0] as usize];
            let n10 = self.mesh.n[vtx[1] as usize];
            let n01 = self.mesh.n[vtx[2] as usize];
            let n11 = self.mesh.n[vtx[3] as usize];
            let mut ns = interpolate_n(uv, n00, n10, n01, n11);
            if ns.length_squared() > 0.0 {
                ns = ns.normalize();
                let dndu_shading = lerp_normal3(uv.y, n10, n11) - lerp_normal3(uv.y, n00, n01);
                let dndv_shading = lerp_normal3(uv.x, n01, n11) - lerp_normal3(uv.x, n00, n10);
                let dnds = dndu_shading * duds + dndv_shading * dvds;
                let dndt = dndu_shading * dudt + dndv_shading * dvdt;

                let r_dpdu = rotate_from_to_vector(isect.n.normalize(), ns, dpdu);
                let r_dpdv = rotate_from_to_vector(isect.n.normalize(), ns, dpdv);
                isect.set_shading_geometry(&ns, &r_dpdu, &r_dpdv, &dnds, &dndt, true);
            }
        }

        isect
    }

    pub fn object_bound(&self) -> Bounds3f {
        let (p00, p10, p01, p11) = self.control_points();
        let p00 = self.mesh.world_to_object.transform_point(&p00);
        let p10 = self.mesh.world_to_object.transform_point(&p10);
        let p01 = self.mesh.world_to_object.transform_point(&p01);
        let p11 = self.mesh.world_to_object.transform_point(&p11);
        Bounds3f::new(&p00, &p10).union(&Bounds3f::new(&p01, &p11))
    }

    pub fn world_bound(&self) -> Bounds3f {
        let (p00, p10, p01, p11) = self.control_points();
        Bounds3f::new(&p00, &p10).union(&Bounds3f::new(&p01, &p11))
    }

    /// pbrt-v4 `BilinearPatch::NormalBounds` (shapes.cpp:1080).
    /// Triangle-like degenerate patches return a single-direction cone;
    /// non-degenerate patches return the average of the four corner
    /// normals with a half-angle wide enough to enclose all of them.
    pub fn normal_bounds(&self) -> DirectionCone {
        let (p00, p10, p01, p11) = self.control_points();
        let mesh = &self.mesh;
        let v = self.vertex_indices();

        let face_forward = |n: Vector3f, ns: &Normal3f| -> Vector3f {
            if Vector3f::dot(&n, &Vector3f::from(*ns)) < 0.0 {
                -n
            } else {
                n
            }
        };
        let flip_for_orientation = |n: Vector3f| -> Vector3f {
            if mesh.reverse_orientation ^ mesh.swaps_handedness {
                -n
            } else {
                n
            }
        };

        // Triangle-degenerate patch (one pair of corners coincide).
        if p00 == p10 || p10 == p11 || p11 == p01 || p01 == p00 {
            let dpdu = lerp_point3(0.5, p10, p11) - lerp_point3(0.5, p00, p01);
            let dpdv = lerp_point3(0.5, p01, p11) - lerp_point3(0.5, p00, p10);
            let mut n = Vector3f::cross(&dpdu, &dpdv).normalize();
            if !mesh.n.is_empty() {
                let ns = (mesh.n[v[0] as usize]
                    + mesh.n[v[1] as usize]
                    + mesh.n[v[2] as usize]
                    + mesh.n[v[3] as usize])
                    * 0.25;
                n = face_forward(n, &ns);
            } else {
                n = flip_for_orientation(n);
            }
            return DirectionCone::from_direction(n);
        }

        // Per-corner geometric normals.
        let mut n00 = Vector3f::cross(&(p10 - p00), &(p01 - p00)).normalize();
        let mut n10 = Vector3f::cross(&(p11 - p10), &(p00 - p10)).normalize();
        let mut n01 = Vector3f::cross(&(p00 - p01), &(p11 - p01)).normalize();
        let mut n11 = Vector3f::cross(&(p01 - p11), &(p10 - p11)).normalize();
        if !mesh.n.is_empty() {
            n00 = face_forward(n00, &mesh.n[v[0] as usize]);
            n10 = face_forward(n10, &mesh.n[v[1] as usize]);
            n01 = face_forward(n01, &mesh.n[v[2] as usize]);
            n11 = face_forward(n11, &mesh.n[v[3] as usize]);
        } else {
            n00 = flip_for_orientation(n00);
            n10 = flip_for_orientation(n10);
            n01 = flip_for_orientation(n01);
            n11 = flip_for_orientation(n11);
        }

        let n = (n00 + n10 + n01 + n11).normalize();
        let cos_theta = Vector3f::dot(&n, &n00)
            .min(Vector3f::dot(&n, &n01))
            .min(Vector3f::dot(&n, &n10))
            .min(Vector3f::dot(&n, &n11))
            .clamp(-1.0, 1.0);
        DirectionCone::new(n, cos_theta)
    }

    pub fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let (p00, p10, p01, p11) = self.control_points();
        let (uv, t_hit) = Self::intersect_patch(ray, t_max, p00, p10, p01, p11)?;
        let isect = self.interaction_from_uv(uv, ray.time, -ray.d);
        Some(ShapeIntersection::new(isect, t_hit))
    }

    pub fn intersect_p(&self, ray: &Ray, t_max: Float) -> bool {
        let (p00, p10, p01, p11) = self.control_points();
        Self::intersect_patch(ray, t_max, p00, p10, p01, p11).is_some()
    }

    pub fn area(&self) -> Float {
        self.area
    }

    pub fn sample(&self, u: &Point2f) -> Option<(Interaction, Float)> {
        let (p00, p10, p01, p11) = self.control_points();
        let weights = [
            Vector3f::cross(&(p10 - p00), &(p01 - p00)).length(),
            Vector3f::cross(&(p10 - p00), &(p11 - p10)).length(),
            Vector3f::cross(&(p01 - p00), &(p11 - p01)).length(),
            Vector3f::cross(&(p11 - p10), &(p11 - p01)).length(),
        ];
        let uv = if self.is_rectangle() {
            *u
        } else {
            sample_bilinear(*u, weights)
        };

        let p = interpolate_p(uv, p00, p10, p01, p11);
        let dpdu = patch_dpdu(uv, p00, p10, p01, p11);
        let dpdv = patch_dpdv(uv, p00, p10, p01, p11);
        let dndp = Vector3f::cross(&dpdu, &dpdv);
        if dndp.length_squared() == 0.0 {
            return None;
        }

        let mut n = dndp.normalize();
        let vtx = self.vertex_indices();
        if !self.mesh.n.is_empty() {
            let ns = interpolate_n(
                uv,
                self.mesh.n[vtx[0] as usize],
                self.mesh.n[vtx[1] as usize],
                self.mesh.n[vtx[2] as usize],
                self.mesh.n[vtx[3] as usize],
            );
            n = face_forward(&n, &ns);
        } else if self.mesh.reverse_orientation ^ self.mesh.swaps_handedness {
            n = -n;
        }

        let p_error = gamma(6.0) * (p00.abs() + p01.abs() + p10.abs() + p11.abs());
        let pdf = if self.is_rectangle() {
            1.0 / dndp.length()
        } else {
            bilinear_pdf(uv, weights) / dndp.length()
        };
        Some((Interaction::from_surface_sample(&p, &p_error, &n), pdf))
    }

    pub fn pdf(&self, _inter: &Interaction) -> Float {
        Float::recip(self.area())
    }

    pub fn sample_from(&self, inter: &Interaction, u: &Point2f) -> Option<(Interaction, Float)> {
        let (p00, p10, p01, p11) = self.control_points();
        let p_ref = inter.get_p();
        let v00 = (p00 - p_ref).normalize();
        let v10 = (p10 - p_ref).normalize();
        let v01 = (p01 - p_ref).normalize();
        let v11 = (p11 - p_ref).normalize();
        let solid_angle = sampling::spherical_quad_area(&v00, &v10, &v11, &v01);

        if !self.is_rectangle() || solid_angle <= Self::MIN_SPHERICAL_SAMPLE_AREA {
            let (mut intr, pdf) = self.sample(u)?;
            intr.set_time(inter.get_time());
            let wi = intr.get_p() - p_ref;
            if wi.length_squared() == 0.0 {
                return None;
            }
            let wi = wi.normalize();
            let pdf =
                pdf * Point3f::distance_squared(&p_ref, &intr.get_p()) / intr.get_n().abs_dot(&-wi);
            if pdf.is_infinite() {
                return None;
            }
            return Some((intr, pdf));
        }

        let context = ShapeSampleContext::from(inter);
        let mut u = *u;
        let mut pdf = 1.0;
        if context.ns != Normal3f::zero() {
            let weights = [
                Float::max(0.01, Vector3f::abs_dot(&v00, &context.ns)),
                Float::max(0.01, Vector3f::abs_dot(&v10, &context.ns)),
                Float::max(0.01, Vector3f::abs_dot(&v01, &context.ns)),
                Float::max(0.01, Vector3f::abs_dot(&v11, &context.ns)),
            ];
            u = sampling::sample_bilinear(u, &weights);
            pdf *= sampling::bilinear_pdf(u, &weights);
        }

        let eu = p10 - p00;
        let ev = p01 - p00;
        let (p, quad_pdf) = sampling::sample_spherical_rectangle(p_ref, p00, eu, ev, u);
        pdf *= quad_pdf;

        let uv = Point2f::new(
            Vector3f::dot(&(p - p00), &eu) / Point3f::distance_squared(&p10, &p00),
            Vector3f::dot(&(p - p00), &ev) / Point3f::distance_squared(&p01, &p00),
        );
        let mut n = Vector3f::cross(&eu, &ev).normalize();
        let vertices = self.vertex_indices();
        if !self.mesh.n.is_empty() {
            let ns = interpolate_n(
                uv,
                self.mesh.n[vertices[0] as usize],
                self.mesh.n[vertices[1] as usize],
                self.mesh.n[vertices[2] as usize],
                self.mesh.n[vertices[3] as usize],
            );
            n = face_forward(&n, &ns);
        } else if self.mesh.reverse_orientation ^ self.mesh.swaps_handedness {
            n = -n;
        }

        let st = if self.mesh.uv.is_empty() {
            uv
        } else {
            interpolate_uv(
                uv,
                self.mesh.uv[vertices[0] as usize],
                self.mesh.uv[vertices[1] as usize],
                self.mesh.uv[vertices[2] as usize],
                self.mesh.uv[vertices[3] as usize],
            )
        };
        let intr = Interaction::Base(BaseInteraction {
            p,
            n,
            uv: st,
            time: context.time,
            ..Default::default()
        });
        Some((intr, pdf))
    }

    pub fn pdf_from(&self, inter: &Interaction, wi: &Vector3f) -> Float {
        let ray = inter.spawn_ray(wi);
        let Some(si) = self.intersect(&ray, Float::INFINITY) else {
            return 0.0;
        };
        let isect_light = si.intr;

        let (p00, p10, p01, p11) = self.control_points();
        let p_ref = inter.get_p();
        let v00 = (p00 - p_ref).normalize();
        let v10 = (p10 - p_ref).normalize();
        let v01 = (p01 - p_ref).normalize();
        let v11 = (p11 - p_ref).normalize();
        let solid_angle = sampling::spherical_quad_area(&v00, &v10, &v11, &v01);

        if !self.is_rectangle() || solid_angle <= Self::MIN_SPHERICAL_SAMPLE_AREA {
            let pdf = self.pdf(&Interaction::Surface(isect_light.clone()))
                * Point3f::distance_squared(&p_ref, &isect_light.p)
                / Vector3f::abs_dot(&isect_light.n, &(-*wi));
            if pdf.is_infinite() {
                return 0.0;
            }
            return pdf;
        }

        let mut pdf = 1.0 / solid_angle;
        let context = ShapeSampleContext::from(inter);
        if context.ns != Normal3f::zero() {
            let weights = [
                Float::max(0.01, Vector3f::abs_dot(&v00, &context.ns)),
                Float::max(0.01, Vector3f::abs_dot(&v10, &context.ns)),
                Float::max(0.01, Vector3f::abs_dot(&v01, &context.ns)),
                Float::max(0.01, Vector3f::abs_dot(&v11, &context.ns)),
            ];
            let u = sampling::invert_spherical_rectangle_sample(
                p_ref,
                p00,
                p10 - p00,
                p01 - p00,
                isect_light.p,
            );
            pdf *= sampling::bilinear_pdf(u, &weights);
        }
        pdf
    }

    pub fn solid_angle(&self, p: &Point3f, n_samples: i32) -> Float {
        let mut it = BaseInteraction::default();
        it.p = *p;
        it.wo = Vector3f::new(0.0, 0.0, 1.0);
        let inter = Interaction::from(it);
        let mut solid_angle = 0.0;
        for i in 0..n_samples {
            let u = Point2f::new(radical_inverse(0, i as u64), radical_inverse(1, i as u64));
            if let Some((p_shape, pdf)) = self.sample_from(&inter, &u) {
                let r = Ray::new(p, &(p_shape.get_p() - *p), 0.999, 0.0);
                if !self.intersect_p(&r, 0.999) {
                    solid_angle += 1.0 / pdf;
                }
            }
        }
        solid_angle / n_samples as Float
    }
}

pub fn create_bilinear_mesh_shape(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    params: &ParameterDictionary,
    float_textures: &FloatTextureMap,
) -> Result<Vec<Shape>, PbrtError> {
    let mut p: Vec<Point3f> = Vec::new();
    if let Some(points) = params.get_points_ref("P") {
        let point_count = points.len() / 3;
        p.reserve(point_count);
        for i in 0..point_count {
            p.push(Point3f::new(
                points[3 * i],
                points[3 * i + 1],
                points[3 * i + 2],
            ));
        }
    }
    if p.is_empty() {
        return Err(PbrtError::from("Invalid bilinearmesh: missing P"));
    }

    let mut quad_indices: Vec<u32> = Vec::new();
    if let Some(indices) = params.get_ints_ref("indices") {
        if indices.len() % 4 != 0 {
            return Err(PbrtError::from(
                "Invalid bilinearmesh: indices length must be a multiple of 4",
            ));
        }
        quad_indices.reserve(indices.len());
        for &i in indices.iter() {
            quad_indices.push(i as u32);
        }
    } else {
        if p.len() % 4 != 0 {
            return Err(PbrtError::from(
                "Invalid bilinearmesh: point count must be a multiple of 4 when indices are omitted",
            ));
        }
        quad_indices.extend((0..p.len()).map(|i| i as u32));
    }

    let mut n: Vec<Normal3f> = Vec::new();
    if let Some(normals) = params.get_points_ref("N") {
        let normal_count = normals.len() / 3;
        n.reserve(normal_count);
        for i in 0..normal_count {
            n.push(Normal3f::new(
                normals[3 * i],
                normals[3 * i + 1],
                normals[3 * i + 2],
            ));
        }
    }

    let mut uv: Vec<Point2f> = params.get_point2f_array("uv");
    if uv.is_empty() {
        uv = params.get_point2f_array("st");
    }

    let mut face_indices = Vec::new();
    if let Some(indices) = params.get_ints_ref("faceIndices") {
        face_indices.extend(indices.iter().copied());
    }

    let shapes: Vec<Shape> = create_bilinear_patch_mesh(
        o2w,
        w2o,
        reverse_orientation,
        quad_indices,
        p,
        n,
        uv,
        face_indices,
    )?
    .into_iter()
    .map(Shape::BilinearPatch)
    .collect();

    let alpha_mask_info = get_alpha_texture(params, float_textures)?;
    let shadow_alpha_mask_info = get_shadow_alpha_texture(params, float_textures)?;
    if alpha_mask_info.is_some() || shadow_alpha_mask_info.is_some() {
        return Ok(shapes
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

    Ok(shapes)
}
