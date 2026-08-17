use super::alphamask::*;
use crate::base::shape::Shape;
use crate::interaction::*;
use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::profile::*;
use crate::util::sampling::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::stats::*;

use std::collections::HashMap;
use std::sync::Arc;

thread_local!(static TESTS: StatPercent = StatPercent::new("Intersections/Ray-triangle intersection tests"));
thread_local!(static TRI_MESH_BYTES: StatMemoryCounter = StatMemoryCounter::new("Memory/Triangle meshes"));

pub struct TriangleMesh {
    pub object_to_world: Transform,
    pub world_to_object: Transform,
    pub reverse_orientation: bool,
    pub swaps_handedness: bool,
    pub two_sided: bool,
    /// 3 indices per triangle, flat-packed. `Triangle::tri_index`
    /// indexes into this; the three vertex indices for triangle `t`
    /// are `indices[3*t .. 3*t + 3]`. Storing the indices on the mesh
    /// (instead of duplicating them inside every `Triangle`) shrinks
    /// `Triangle` from 32 to 16 bytes — same total bytes per triangle,
    /// but the smaller `Triangle` brings the `Shape` enum size down.
    pub vertex_indices: Vec<u32>,
    pub p: Vec<Point3f>,
    pub s: Vec<Vector3f>,
    pub n: Vec<Vector3f>,
    pub uv: Vec<Point2f>,
}
const MACHINE_EPSILON: Float = Float::EPSILON * 0.5;
const GAMMA2: Float = (2.0 * MACHINE_EPSILON) / (1.0 - (2.0 * MACHINE_EPSILON));
const GAMMA3: Float = (3.0 * MACHINE_EPSILON) / (1.0 - (3.0 * MACHINE_EPSILON));
const GAMMA5: Float = (5.0 * MACHINE_EPSILON) / (1.0 - (5.0 * MACHINE_EPSILON));
const GAMMA6: Float = (6.0 * MACHINE_EPSILON) / (1.0 - (5.0 * MACHINE_EPSILON));
const GAMMA7: Float = (7.0 * MACHINE_EPSILON) / (1.0 - (7.0 * MACHINE_EPSILON));
const TRI: [usize; 4] = [0, 1, 2, 0];

impl TriangleMesh {
    pub fn new(
        object_to_world: &Transform,
        reverse_orientation: bool,
        two_sided: bool,
        vertex_indices: Vec<u32>,
        p: Vec<Point3f>,
        s: Vec<Vector3f>,
        n: Vec<Normal3f>,
        uv: Vec<Point2f>,
    ) -> Self {
        let p: Vec<Point3f> = p
            .iter()
            .map(|p| -> Point3f {
                return object_to_world.transform_point(p);
            })
            .collect();
        let s: Vec<Vector3f> = s
            .iter()
            .map(|s| -> Vector3f {
                return object_to_world.transform_vector(s);
            })
            .collect();
        // pbrt-v4 `TriangleMesh` ctor (util/mesh.cpp:49-55) flips per-vertex
        // normals when `reverseOrientation` is set. Without this flip the
        // shading normal disagrees with the geometric normal that
        // `Triangle::Intersect` produced (which IS flipped by RO via the
        // `cross(dp02, dp12)` sign), so `SetShadingGeometry(..., auth=true)`
        // pulls the surface normal back to the un-flipped direction and the
        // `MediumInterface` inside/outside lookup ends up inverted (dambreak1
        // bias).
        let n: Vec<Normal3f> = n
            .iter()
            .map(|n| -> Normal3f {
                let mut nn = object_to_world.transform_normal(n);
                if reverse_orientation {
                    nn = -nn;
                }
                nn
            })
            .collect();
        let swaps_handedness = object_to_world.swaps_handedness();
        TriangleMesh {
            object_to_world: *object_to_world,
            world_to_object: object_to_world.inverse(),
            reverse_orientation,
            swaps_handedness,
            two_sided,
            vertex_indices,
            p,
            s,
            n,
            uv,
        }
    }

    pub fn calc_normal(&self, dpdu: &Vector3f, dpdv: &Vector3f) -> Normal3f {
        let mut n = Vector3f::cross(dpdu, dpdv).normalize();
        if self.reverse_orientation ^ self.swaps_handedness {
            n *= -1.0;
        }
        return n;
    }

    pub fn create(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        params: &ParameterDictionary,
        float_textures: &HashMap<String, Arc<FloatTexture>>,
    ) -> Result<Vec<Shape>, PbrtError> {
        create_triangle_mesh_shape(o2w, w2o, reverse_orientation, params, float_textures)
    }
}

/// pbrt-v4 `TriangleIntersection` (shapes.h:819-824). Holds the
/// barycentric coordinates and ray parameter of a ray/triangle hit so
/// `Triangle::intersect` can be split into a hot edge-function test
/// (`intersect_triangle`) followed by `Triangle::interaction_from_intersection`.
#[derive(Clone, Copy, Debug)]
pub struct TriangleIntersection {
    pub b0: Float,
    pub b1: Float,
    pub b2: Float,
    pub t: Float,
}

/// pbrt-v4 `IntersectTriangle` (shapes.cpp:168-269), verbatim. Returns
/// the barycentric coordinates and ray parameter when the ray hits the
/// triangle `(p0, p1, p2)` within `[0, t_max]`; `None` otherwise.
pub fn intersect_triangle(
    ray: &Ray,
    t_max: Float,
    p0: Point3f,
    p1: Point3f,
    p2: Point3f,
) -> Option<TriangleIntersection> {
    // Return no intersection if triangle is degenerate
    if Vector3f::cross(&(p2 - p0), &(p1 - p0)).length_squared() == 0.0 {
        return None;
    }

    // Transform triangle vertices to ray coordinate space
    let mut p0t = p0 - ray.o;
    let mut p1t = p1 - ray.o;
    let mut p2t = p2 - ray.o;

    // Permute components of triangle vertices and ray direction
    let kz = max_dimension(&ray.d.abs()) as usize;
    let kx = TRI[kz + 1];
    let ky = TRI[kx + 1];
    let d = permute(&ray.d, kx, ky, kz);
    p0t = permute(&p0t, kx, ky, kz);
    p1t = permute(&p1t, kx, ky, kz);
    p2t = permute(&p2t, kx, ky, kz);

    // Apply shear transformation to translated vertex positions
    let sx = -d.x / d.z;
    let sy = -d.y / d.z;
    let sz = 1.0 / d.z;
    p0t.x += sx * p0t.z;
    p0t.y += sy * p0t.z;
    p1t.x += sx * p1t.z;
    p1t.y += sy * p1t.z;
    p2t.x += sx * p2t.z;
    p2t.y += sy * p2t.z;

    // Compute edge function coefficients via `difference_of_products`
    let mut e0 = difference_of_products(p1t.x, p2t.y, p1t.y, p2t.x);
    let mut e1 = difference_of_products(p2t.x, p0t.y, p2t.y, p0t.x);
    let mut e2 = difference_of_products(p0t.x, p1t.y, p0t.y, p1t.x);

    // Fall back to double-precision at triangle edges
    if e0 == 0.0 || e1 == 0.0 || e2 == 0.0 {
        let p2txp1ty = p2t.x as f64 * p1t.y as f64;
        let p2typ1tx = p2t.y as f64 * p1t.x as f64;
        e0 = (p2typ1tx - p2txp1ty) as Float;
        let p0txp2ty = p0t.x as f64 * p2t.y as f64;
        let p0typ2tx = p0t.y as f64 * p2t.x as f64;
        e1 = (p0typ2tx - p0txp2ty) as Float;
        let p1txp0ty = p1t.x as f64 * p0t.y as f64;
        let p1typ0tx = p1t.y as f64 * p0t.x as f64;
        e2 = (p1typ0tx - p1txp0ty) as Float;
    }

    // Perform triangle edge and determinant tests
    if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
        return None;
    }
    let det = e0 + e1 + e2;
    if det == 0.0 {
        return None;
    }

    // Compute scaled hit distance to triangle and test against ray t range
    p0t.z *= sz;
    p1t.z *= sz;
    p2t.z *= sz;
    let t_scaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if det < 0.0 && (t_scaled >= 0.0 || t_scaled < t_max * det) {
        return None;
    } else if det > 0.0 && (t_scaled <= 0.0 || t_scaled > t_max * det) {
        return None;
    }

    // Compute barycentric coordinates and t value
    let inv_det = 1.0 / det;
    let b0 = e0 * inv_det;
    let b1 = e1 * inv_det;
    let b2 = e2 * inv_det;
    let t = t_scaled * inv_det;

    // Ensure that computed triangle t is conservatively greater than zero
    let max_zt = max_component(&Vector3f::new(p0t.z, p1t.z, p2t.z).abs());
    let delta_z = GAMMA3 * max_zt;
    let max_xt = max_component(&Vector3f::new(p0t.x, p1t.x, p2t.x).abs());
    let max_yt = max_component(&Vector3f::new(p0t.y, p1t.y, p2t.y).abs());
    let delta_x = GAMMA5 * (max_xt + max_zt);
    let delta_y = GAMMA5 * (max_yt + max_zt);
    let delta_e = 2.0 * (GAMMA2 * max_xt * max_yt + delta_y * max_xt + delta_x * max_yt);
    let max_e = max_component(&Vector3f::new(e0, e1, e2).abs());
    let delta_t =
        3.0 * (GAMMA3 * max_e * max_zt + delta_e * max_zt + delta_z * max_e) * Float::abs(inv_det);
    if t <= delta_t {
        return None;
    }

    Some(TriangleIntersection { b0, b1, b2, t })
}

pub struct Triangle {
    pub mesh: Arc<TriangleMesh>,
    /// Index of this triangle's first vertex slot in `mesh.vertex_indices`.
    /// The three vertex indices are `mesh.vertex_indices[3*tri_index ..]`.
    /// Stored as `u32` (was `usize`) so `Triangle` fits in 16 bytes
    /// with the `Arc<TriangleMesh>` pointer alone — shrinks the
    /// enclosing `Shape` enum from 40 to 32 bytes per slot.
    pub tri_index: u32,
}

impl Triangle {
    pub fn new(mesh: &Arc<TriangleMesh>, tri_index: u32) -> Self {
        TRI_MESH_BYTES.with(|s| {
            s.add(std::mem::size_of::<Triangle>());
        });

        Triangle {
            mesh: Arc::clone(mesh),
            tri_index,
        }
    }

    /// Vertex indices for this triangle (`mesh.vertex_indices[3*tri_index..]`).
    #[inline]
    pub fn vertex_indices(&self) -> [u32; 3] {
        let base = 3 * self.tri_index as usize;
        [
            self.mesh.vertex_indices[base],
            self.mesh.vertex_indices[base + 1],
            self.mesh.vertex_indices[base + 2],
        ]
    }

    pub fn calc_normal(&self, dpdu: &Vector3f, dpdv: &Vector3f) -> Normal3f {
        return self.mesh.as_ref().calc_normal(dpdu, dpdv);
    }

    /// pbrt-v4 `Triangle::InteractionFromIntersection` (shapes.h:884-1010),
    /// verbatim. Builds the `SurfaceInteraction` for a confirmed
    /// ray/triangle hit from the barycentric coordinates returned by
    /// `intersect_triangle`. Generates per-face default UVs
    /// `(0,0)/(1,0)/(1,1)` when `mesh.uv` is empty, and falls back to a
    /// double-precision cross product when single-precision dpdu/dpdv
    /// collapse to zero.
    pub fn interaction_from_intersection(
        &self,
        ti: TriangleIntersection,
        time: Float,
        wo: Vector3f,
    ) -> SurfaceInteraction {
        let mesh = self.mesh.as_ref();
        let base = 3 * self.tri_index as usize;
        let v = [
            self.mesh.vertex_indices[base] as usize,
            self.mesh.vertex_indices[base + 1] as usize,
            self.mesh.vertex_indices[base + 2] as usize,
        ];
        let p0 = mesh.p[v[0]];
        let p1 = mesh.p[v[1]];
        let p2 = mesh.p[v[2]];

        // Per-face default UVs when mesh has no per-vertex uv (shapes.h:893-897).
        let uv = if !mesh.uv.is_empty() {
            [mesh.uv[v[0]], mesh.uv[v[1]], mesh.uv[v[2]]]
        } else {
            [
                Point2f::new(0.0, 0.0),
                Point2f::new(1.0, 0.0),
                Point2f::new(1.0, 1.0),
            ]
        };

        let duv02 = uv[0] - uv[2];
        let duv12 = uv[1] - uv[2];
        let dp02 = p0 - p2;
        let dp12 = p1 - p2;
        let determinant = difference_of_products(duv02[0], duv12[1], duv02[1], duv12[0]);

        let mut dpdu = Vector3f::zero();
        let mut dpdv = Vector3f::zero();
        let degenerate_uv = Float::abs(determinant) < 1e-9;
        if !degenerate_uv {
            let invdet = 1.0 / determinant;
            // dpdu = (duv12[1] * dp02 - duv02[1] * dp12) * invdet (via DOP per component)
            dpdu = Vector3f::new(
                difference_of_products(duv12[1], dp02.x, duv02[1], dp12.x),
                difference_of_products(duv12[1], dp02.y, duv02[1], dp12.y),
                difference_of_products(duv12[1], dp02.z, duv02[1], dp12.z),
            ) * invdet;
            dpdv = Vector3f::new(
                difference_of_products(duv02[0], dp12.x, duv12[0], dp02.x),
                difference_of_products(duv02[0], dp12.y, duv12[0], dp02.y),
                difference_of_products(duv02[0], dp12.z, duv12[0], dp02.z),
            ) * invdet;
        }
        // Degenerate (u,v) parameterization or partial derivatives: fall back
        // to a coordinate system built from the geometric normal, retrying with
        // double precision if the single-precision cross collapses.
        if degenerate_uv || Vector3f::cross(&dpdu, &dpdv).length_squared() == 0.0 {
            let mut ng = Vector3f::cross(&(p2 - p0), &(p1 - p0));
            if ng.length_squared() == 0.0 {
                let d02 = [
                    (p2.x - p0.x) as f64,
                    (p2.y - p0.y) as f64,
                    (p2.z - p0.z) as f64,
                ];
                let d01 = [
                    (p1.x - p0.x) as f64,
                    (p1.y - p0.y) as f64,
                    (p1.z - p0.z) as f64,
                ];
                let cx = d02[1] * d01[2] - d02[2] * d01[1];
                let cy = d02[2] * d01[0] - d02[0] * d01[2];
                let cz = d02[0] * d01[1] - d02[1] * d01[0];
                ng = Vector3f::new(cx as Float, cy as Float, cz as Float);
            }
            let (du, dv) = coordinate_system(&ng.normalize());
            dpdu = du;
            dpdv = dv;
        }

        // Interpolate (u,v) parametric coordinates and hit point
        let p_hit = ti.b0 * p0 + ti.b1 * p1 + ti.b2 * p2;
        let uv_hit = ti.b0 * uv[0] + ti.b1 * uv[1] + ti.b2 * uv[2];

        // Compute error bounds for triangle intersection
        let p_abs_sum = Vector3f::abs(&(ti.b0 * p0))
            + Vector3f::abs(&(ti.b1 * p1))
            + Vector3f::abs(&(ti.b2 * p2));
        let p_error = GAMMA7 * Vector3f::new(p_abs_sum.x, p_abs_sum.y, p_abs_sum.z);

        // Geometric normal from cross(dp02, dp12), then flip if requested.
        let mut n = Vector3f::cross(&dp02, &dp12).normalize();
        if mesh.reverse_orientation ^ mesh.swaps_handedness {
            n *= -1.0;
        }

        let mut isect = SurfaceInteraction::new(
            &p_hit,
            &p_error,
            &uv_hit,
            &wo,
            &n,
            &dpdu,
            &dpdv,
            &Normal3f::zero(),
            &Normal3f::zero(),
            time,
            self.tri_index as u32,
        );
        isect.n = n;
        isect.shading.n = n;

        if !mesh.n.is_empty() || !mesh.s.is_empty() {
            // Shading normal
            let mut ns = if !mesh.n.is_empty() {
                let nn = ti.b0 * mesh.n[v[0]] + ti.b1 * mesh.n[v[1]] + ti.b2 * mesh.n[v[2]];
                if nn.length_squared() > 0.0 {
                    nn.normalize()
                } else {
                    isect.n
                }
            } else {
                isect.n
            };

            // Shading tangent
            let mut ss = if !mesh.s.is_empty() {
                let ss0 = ti.b0 * mesh.s[v[0]] + ti.b1 * mesh.s[v[1]] + ti.b2 * mesh.s[v[2]];
                if ss0.length_squared() == 0.0 {
                    isect.dpdu
                } else {
                    ss0
                }
            } else {
                isect.dpdu
            };

            // Shading bitangent and re-orthogonalised tangent
            let mut ts = Vector3f::cross(&ns, &ss);
            if ts.length_squared() > 0.0 {
                ss = Vector3f::cross(&ts, &ns);
            } else {
                let (ss1, ts1) = coordinate_system(&ns);
                ss = ss1;
                ts = ts1;
            }

            // Shading dndu/dndv
            let mut dndu = Vector3f::zero();
            let mut dndv = Vector3f::zero();
            if !mesh.n.is_empty() {
                let duv02 = uv[0] - uv[2];
                let duv12 = uv[1] - uv[2];
                let dn1 = mesh.n[v[0]] - mesh.n[v[2]];
                let dn2 = mesh.n[v[1]] - mesh.n[v[2]];
                let determinant = difference_of_products(duv02[0], duv12[1], duv02[1], duv12[0]);
                let degenerate_uv = Float::abs(determinant) < 1e-9;
                if degenerate_uv {
                    let dn = Vector3f::cross(
                        &(mesh.n[v[2]] - mesh.n[v[0]]),
                        &(mesh.n[v[1]] - mesh.n[v[0]]),
                    );
                    if dn.length_squared() > 0.0 {
                        let (dnu, dnv) = coordinate_system(&dn);
                        dndu = dnu;
                        dndv = dnv;
                    }
                } else {
                    let inv_det = 1.0 / determinant;
                    dndu = Vector3f::new(
                        difference_of_products(duv12[1], dn1.x, duv02[1], dn2.x),
                        difference_of_products(duv12[1], dn1.y, duv02[1], dn2.y),
                        difference_of_products(duv12[1], dn1.z, duv02[1], dn2.z),
                    ) * inv_det;
                    dndv = Vector3f::new(
                        difference_of_products(duv02[0], dn2.x, duv12[0], dn1.x),
                        difference_of_products(duv02[0], dn2.y, duv12[0], dn1.y),
                        difference_of_products(duv02[0], dn2.z, duv12[0], dn1.z),
                    ) * inv_det;
                }
            }

            isect.set_shading_geometry(&ns, &ss, &ts, &dndu, &dndv, true);
            // Suppress unused warning when shading.n is set by SetShadingGeometry.
            let _ = &mut ns;
        }

        isect
    }
}

fn union3(p0: Point3f, p1: Point3f, p2: Point3f) -> Bounds3f {
    let a = [[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]];
    let mut min: [Float; 3] = [p0.x, p0.y, p0.z];
    let mut max: [Float; 3] = [p0.x, p0.y, p0.z];
    for j in 0..2 {
        for i in 0..3 {
            min[i] = Float::min(min[i], a[j][i]);
            max[i] = Float::max(max[i], a[j][i]);
        }
    }
    return Bounds3f::from(((min[0], min[1], min[2]), (max[0], max[1], max[2])));
}

impl Triangle {
    pub fn object_bound(&self) -> Bounds3f {
        let mesh = self.mesh.as_ref();
        let world_to_object = mesh.world_to_object;
        let i0 = self.mesh.vertex_indices[3 * self.tri_index as usize + 0] as usize;
        let i1 = self.mesh.vertex_indices[3 * self.tri_index as usize + 1] as usize;
        let i2 = self.mesh.vertex_indices[3 * self.tri_index as usize + 2] as usize;
        let p0 = world_to_object.transform_point(&mesh.p[i0]);
        let p1 = world_to_object.transform_point(&mesh.p[i1]);
        let p2 = world_to_object.transform_point(&mesh.p[i2]);
        return union3(p0, p1, p2);
    }

    pub fn world_bound(&self) -> Bounds3f {
        let mesh = self.mesh.as_ref();
        let i0 = self.mesh.vertex_indices[3 * self.tri_index as usize + 0] as usize;
        let i1 = self.mesh.vertex_indices[3 * self.tri_index as usize + 1] as usize;
        let i2 = self.mesh.vertex_indices[3 * self.tri_index as usize + 2] as usize;
        let p0 = mesh.p[i0];
        let p1 = mesh.p[i1];
        let p2 = mesh.p[i2];
        return union3(p0, p1, p2);
    }

    /// pbrt-v4 `Triangle::NormalBounds` (shapes.cpp:299). Returns a
    /// tight cone around the face normal of this triangle. When the
    /// mesh has per-vertex shading normals, the face normal is
    /// face-forwarded to align with their average so the cone matches
    /// the side the shading frame is on.
    pub fn normal_bounds(&self) -> DirectionCone {
        use crate::util::base::{Normal3f, Vector3f};
        use crate::util::geometry::DirectionCone;
        let mesh = self.mesh.as_ref();
        let base = 3 * self.tri_index as usize;
        let i0 = mesh.vertex_indices[base] as usize;
        let i1 = mesh.vertex_indices[base + 1] as usize;
        let i2 = mesh.vertex_indices[base + 2] as usize;
        let p0 = mesh.p[i0];
        let p1 = mesh.p[i1];
        let p2 = mesh.p[i2];
        let mut n: Normal3f = Normal3f::from(Vector3f::cross(&(p1 - p0), &(p2 - p0))).normalize();
        if !mesh.n.is_empty() {
            // Face-forward the geometric normal to the side of the
            // shading frame (v4 lines 307-309).
            let ns_sum = mesh.n[i0] + mesh.n[i1] + mesh.n[i2];
            if Vector3f::dot(&Vector3f::from(n), &ns_sum) < 0.0 {
                n = -n;
            }
        } else if mesh.reverse_orientation ^ mesh.swaps_handedness {
            n = -n;
        }
        DirectionCone::from_direction(Vector3f::from(n))
    }

    /// pbrt-v4 `Triangle::Intersect` (shapes.cpp). Calls the shared
    /// `intersect_triangle` edge-function test, then constructs the
    /// `SurfaceInteraction` via `interaction_from_intersection`. The
    /// two-step structure mirrors v4 verbatim so any shading-geometry
    /// regression is contained to `interaction_from_intersection`.
    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let _p = ProfilePhase::new(Prof::TriIntersect);
        TESTS.with(|stat| {
            stat.add_denom(1);
        });

        let mesh = self.mesh.as_ref();
        let base = 3 * self.tri_index as usize;
        let p0 = mesh.p[self.mesh.vertex_indices[base] as usize];
        let p1 = mesh.p[self.mesh.vertex_indices[base + 1] as usize];
        let p2 = mesh.p[self.mesh.vertex_indices[base + 2] as usize];

        let ti = intersect_triangle(r, t_max, p0, p1, p2)?;
        let isect = self.interaction_from_intersection(ti, r.time, -r.d);

        TESTS.with(|stat| {
            stat.add_num(1);
        });
        Some(ShapeIntersection::new(isect, ti.t))
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        let _p = ProfilePhase::new(Prof::TriIntersectP);
        TESTS.with(|stat| {
            stat.add_denom(1);
        });

        let mesh = self.mesh.as_ref();
        let base = 3 * self.tri_index as usize;
        let p0 = mesh.p[self.mesh.vertex_indices[base] as usize];
        let p1 = mesh.p[self.mesh.vertex_indices[base + 1] as usize];
        let p2 = mesh.p[self.mesh.vertex_indices[base + 2] as usize];

        let hit = intersect_triangle(r, t_max, p0, p1, p2).is_some();
        if hit {
            TESTS.with(|stat| {
                stat.add_num(1);
            });
        }
        hit
    }

    pub fn area(&self) -> Float {
        let mesh = self.mesh.as_ref();
        let i0 = self.mesh.vertex_indices[3 * self.tri_index as usize + 0] as usize;
        let i1 = self.mesh.vertex_indices[3 * self.tri_index as usize + 1] as usize;
        let i2 = self.mesh.vertex_indices[3 * self.tri_index as usize + 2] as usize;
        let p0 = mesh.p[i0];
        let p1 = mesh.p[i1];
        let p2 = mesh.p[i2];
        return 0.5 * Vector3f::cross(&(p1 - p0), &(p2 - p0)).length();
    }

    pub fn sample(&self, u: &Point2f) -> Option<(Interaction, Float)> {
        let b = uniform_sample_triangle(u);
        // Get triangle vertices in _p0_, _p1_, and _p2_
        let mesh = self.mesh.as_ref();
        let i0 = self.mesh.vertex_indices[3 * self.tri_index as usize + 0] as usize;
        let i1 = self.mesh.vertex_indices[3 * self.tri_index as usize + 1] as usize;
        let i2 = self.mesh.vertex_indices[3 * self.tri_index as usize + 2] as usize;
        let p0 = mesh.p[i0];
        let p1 = mesh.p[i1];
        let p2 = mesh.p[i2];
        let p = b[0] * p0 + b[1] * p1 + (1.0 - b[0] - b[1]) * p2;
        // Compute surface normal for sampled point on triangle
        let mut n = Vector3f::cross(&(p1 - p0), &(p2 - p0)).normalize();
        // Ensure correct orientation of the geometric normal; follow the same
        // approach as was used in Triangle::Intersect().
        if !mesh.n.is_empty() {
            let ns = b[0] * mesh.n[i0] + b[1] * mesh.n[i1] + (1.0 - b[0] - b[1]) * mesh.n[i2];
            n = face_forward(&n, &ns);
        } else if mesh.reverse_orientation ^ mesh.swaps_handedness {
            n *= -1.0;
        }
        // Compute error bounds for sampled point on triangle
        let p_abs_sum = Vector3f::abs(&(b[0] * p0))
            + Vector3f::abs(&(b[1] * p1))
            + Vector3f::abs(&((1.0 - b[0] - b[1]) * p2));
        let p_error = GAMMA6 * Vector3f::new(p_abs_sum.x, p_abs_sum.y, p_abs_sum.z);
        let pdf = 1.0 / self.area();
        let it = Interaction::from_surface_sample(&p, &p_error, &n);
        return Some((it, pdf));
    }

    /// pbrt-v4 `Triangle::Sample(ShapeSampleContext &ctx, Point2f u)`.
    /// Uses spherical-triangle sampling with a bilinear cosine warp
    /// product when the solid angle subtended by the triangle is in
    /// `[MIN_SPHERICAL_SAMPLE_AREA, MAX_SPHERICAL_SAMPLE_AREA]`; falls
    /// back to uniform-area sampling for very small or very large
    /// triangles. Returns `(interaction, solid-angle PDF)`.
    pub fn sample_from(&self, inter: &Interaction, u: &Point2f) -> Option<(Interaction, Float)> {
        let mesh = self.mesh.as_ref();
        let i0 = self.mesh.vertex_indices[3 * self.tri_index as usize + 0] as usize;
        let i1 = self.mesh.vertex_indices[3 * self.tri_index as usize + 1] as usize;
        let i2 = self.mesh.vertex_indices[3 * self.tri_index as usize + 2] as usize;
        let p0 = mesh.p[i0];
        let p1 = mesh.p[i1];
        let p2 = mesh.p[i2];

        let p_ref = inter.get_p();
        let solid_angle = {
            let d0 = p0 - p_ref;
            let d1 = p1 - p_ref;
            let d2 = p2 - p_ref;
            if d0.length_squared() <= 0.0
                || d1.length_squared() <= 0.0
                || d2.length_squared() <= 0.0
            {
                return None;
            }
            spherical_triangle_area(&d0.normalize(), &d1.normalize(), &d2.normalize())
        };

        if solid_angle < MIN_SPHERICAL_SAMPLE_AREA || solid_angle > MAX_SPHERICAL_SAMPLE_AREA {
            // Use v4's uniform-area branch outside the spherical-triangle range.
            let (intr, pdf) = self.sample(u)?;
            assert!(intr.is_surface_interaction());
            let wi = intr.get_p() - p_ref;
            if wi.length_squared() <= 0.0 {
                return None;
            }
            let wi = wi.normalize();
            if !mesh.two_sided && Vector3::dot(&intr.get_n(), &-wi) <= 0.0 {
                return None;
            }
            let pdf = pdf * Vector3f::distance_squared(&p_ref, &intr.get_p())
                / Vector3f::abs_dot(&intr.get_n(), &-wi);
            if pdf <= 0.0 || pdf.is_infinite() {
                return None;
            }
            return Some((intr, pdf));
        }

        // Spherical-triangle sampling, optionally pre-warped by a
        // cosine-bilinear distribution that biases samples toward
        // directions with high `|ns · wi|`.
        let ns_opt = inter.as_surface_interaction().map(|si| si.shading.n);
        let mut u_warp = *u;
        let mut warp_pdf: Float = 1.0;
        if let Some(ns) = ns_opt {
            if ns.length_squared() > 0.0 {
                let rp = p_ref;
                let wi_corners = [
                    (p0 - rp).normalize(),
                    (p1 - rp).normalize(),
                    (p2 - rp).normalize(),
                ];
                // Layout matches pbrt-v4 shapes.h:
                //   w = { max(0.01, |ns·wi[1]|), max(0.01, |ns·wi[1]|),
                //         max(0.01, |ns·wi[0]|), max(0.01, |ns·wi[2]|) }
                // (Two corners share `wi[1]` deliberately; this is the
                //  cosine warp v4 ships with.)
                let w_table: [Float; 4] = [
                    Float::max(0.01, Float::abs(ns.dot(&wi_corners[1]))),
                    Float::max(0.01, Float::abs(ns.dot(&wi_corners[1]))),
                    Float::max(0.01, Float::abs(ns.dot(&wi_corners[0]))),
                    Float::max(0.01, Float::abs(ns.dot(&wi_corners[2]))),
                ];
                u_warp = sample_bilinear(*u, &w_table);
                warp_pdf = bilinear_pdf(u_warp, &w_table);
            }
        }

        let (bary_opt, tri_pdf) = sample_spherical_triangle(&[p0, p1, p2], p_ref, u_warp);
        if tri_pdf == 0.0 {
            return None;
        }
        let b = bary_opt?;
        let pdf = warp_pdf * tri_pdf;
        if pdf <= 0.0 || pdf.is_infinite() {
            return None;
        }

        let p = b[0] * p0 + b[1] * p1 + b[2] * p2;
        let p_abs_sum =
            Vector3f::abs(&(b[0] * p0)) + Vector3f::abs(&(b[1] * p1)) + Vector3f::abs(&(b[2] * p2));
        let p_error = GAMMA6 * Vector3f::new(p_abs_sum.x, p_abs_sum.y, p_abs_sum.z);

        let mut n = Vector3f::cross(&(p1 - p0), &(p2 - p0)).normalize();
        if !mesh.n.is_empty() {
            let ns_sample = b[0] * mesh.n[i0] + b[1] * mesh.n[i1] + b[2] * mesh.n[i2];
            n = face_forward(&n, &ns_sample);
        } else if mesh.reverse_orientation ^ mesh.swaps_handedness {
            n *= -1.0;
        }

        let wi = p - p_ref;
        if wi.length_squared() <= 0.0 {
            return None;
        }
        let wi = wi.normalize();
        if !mesh.two_sided && Vector3::dot(&n, &-wi) <= 0.0 {
            return None;
        }

        let intr = Interaction::from_surface_sample(&p, &p_error, &n);
        Some((intr, pdf))
    }

    pub fn pdf(&self, _inter: &Interaction) -> Float {
        Float::recip(self.area())
    }

    pub fn pdf_from(&self, inter: &Interaction, wi: &Vector3f) -> Float {
        // pbrt-v4 shapes.h:1132 Triangle::PDF(ShapeSampleContext, Vector3f).
        // The spherical-triangle branch computes pdf directly from the
        // solid angle WITHOUT calling Intersect; only the uniform-area
        // This branch needs to ray-cast against the triangle.
        let mesh = self.mesh.as_ref();
        let i0 = self.mesh.vertex_indices[3 * self.tri_index as usize + 0] as usize;
        let i1 = self.mesh.vertex_indices[3 * self.tri_index as usize + 1] as usize;
        let i2 = self.mesh.vertex_indices[3 * self.tri_index as usize + 2] as usize;
        let p0 = mesh.p[i0];
        let p1 = mesh.p[i1];
        let p2 = mesh.p[i2];

        let p_ref = inter.get_p();
        let d0 = p0 - p_ref;
        let d1 = p1 - p_ref;
        let d2 = p2 - p_ref;
        if d0.length_squared() <= 0.0 || d1.length_squared() <= 0.0 || d2.length_squared() <= 0.0 {
            return 0.0;
        }
        let solid_angle =
            spherical_triangle_area(&d0.normalize(), &d1.normalize(), &d2.normalize());

        if solid_angle < MIN_SPHERICAL_SAMPLE_AREA || solid_angle > MAX_SPHERICAL_SAMPLE_AREA {
            // Uniform-area branch: intersect to recover the area-sample point
            // (v4 shapes.h:1138-1149).
            let ray = inter.spawn_ray(wi);
            let si = match self.intersect(&ray, Float::INFINITY) {
                Some(s) => s,
                None => return 0.0,
            };
            let isect_light = si.intr;
            let pdf = Vector3f::distance_squared(&p_ref, &isect_light.p)
                / (Vector3f::abs_dot(&isect_light.n, &(-*wi)) * self.area());
            if pdf.is_infinite() {
                return 0.0;
            }
            return pdf;
        }

        let mut pdf = 1.0 / solid_angle;
        // v4 shapes.h:1154 — adjust PDF for warp product sampling of triangle cos-theta factor.
        if let Some(ns) = inter.as_surface_interaction().map(|si| si.shading.n) {
            if ns.length_squared() > 0.0 {
                let wi_corners = [
                    (p0 - p_ref).normalize(),
                    (p1 - p_ref).normalize(),
                    (p2 - p_ref).normalize(),
                ];
                let w_table: [Float; 4] = [
                    Float::max(0.01, Float::abs(ns.dot(&wi_corners[1]))),
                    Float::max(0.01, Float::abs(ns.dot(&wi_corners[1]))),
                    Float::max(0.01, Float::abs(ns.dot(&wi_corners[0]))),
                    Float::max(0.01, Float::abs(ns.dot(&wi_corners[2]))),
                ];
                let u = invert_spherical_triangle_sample(&[p0, p1, p2], p_ref, *wi);
                pdf *= bilinear_pdf(u, &w_table);
            }
        }
        pdf
    }

    /// pbrt-v4 `Triangle::SolidAngle` — closed-form solid angle
    /// subtended by the triangle at `p`. Replaces the Monte-Carlo
    /// `n_samples` is retained as the v4-compatible parameter name.
    pub fn solid_angle(&self, p: &Point3f, _n_samples: i32) -> Float {
        let mesh = self.mesh.as_ref();
        let p0 = mesh.p[self.mesh.vertex_indices[3 * self.tri_index as usize + 0] as usize];
        let p1 = mesh.p[self.mesh.vertex_indices[3 * self.tri_index as usize + 1] as usize];
        let p2 = mesh.p[self.mesh.vertex_indices[3 * self.tri_index as usize + 2] as usize];
        let d0 = p0 - *p;
        let d1 = p1 - *p;
        let d2 = p2 - *p;
        if d0.length_squared() <= 0.0 || d1.length_squared() <= 0.0 || d2.length_squared() <= 0.0 {
            return 0.0;
        }
        spherical_triangle_area(&d0.normalize(), &d1.normalize(), &d2.normalize())
    }
}

pub fn get_alpha_texture(
    params: &ParameterDictionary,
    float_textures: &FloatTextureMap,
) -> Result<Option<AlphaMaskInfo>, PbrtError> {
    if let Some(textures) = params.get_textures_ref("alpha") {
        if textures.len() >= 1 {
            let alpha_tex_name = textures[0].clone();
            if let Some(tex) = float_textures.get(&alpha_tex_name) {
                return Ok(Some(AlphaMaskInfo::Texture {
                    texture: Arc::clone(tex),
                }));
            }
            return Err(PbrtError::error(&format!(
                "Couldn't find float texture for \"alpha\" parameter: {}",
                alpha_tex_name
            )));
        }
    } else if let Some(alpha) = params.get_floats_ref("alpha") {
        if alpha.len() > 0 {
            return Ok(Some(AlphaMaskInfo::Value { value: alpha[0] }));
        }
    }
    return Ok(None);
}

pub fn get_shadow_alpha_texture(
    params: &ParameterDictionary,
    float_textures: &FloatTextureMap,
) -> Result<Option<AlphaMaskInfo>, PbrtError> {
    if let Some(textures) = params.get_textures_ref("shadowalpha") {
        if textures.len() >= 1 {
            let alpha_tex_name = textures[0].clone();
            if let Some(tex) = float_textures.get(&alpha_tex_name) {
                return Ok(Some(AlphaMaskInfo::Texture {
                    texture: Arc::clone(tex),
                }));
            }
            return Err(PbrtError::error(&format!(
                "Couldn't find float texture for \"shadowalpha\" parameter: {}",
                alpha_tex_name
            )));
        }
    } else if let Some(alpha) = params.get_floats_ref("shadowalpha") {
        if alpha.len() > 0 {
            return Ok(Some(AlphaMaskInfo::Value { value: alpha[0] }));
        }
    }
    return Ok(None);
}

fn validate_triangle_mesh_params(
    vertex_indices: &[u32],
    p: &[Point3f],
    s: &[Vector3f],
    n: &[Vector3f],
    uv: &[Point2f],
) -> Result<(), PbrtError> {
    if vertex_indices.len() % 3 != 0 {
        return Err(PbrtError::from(
            "Invalid trianglemesh: indices length must be a multiple of 3",
        ));
    }
    if p.is_empty() {
        return Err(PbrtError::from(
            "Invalid trianglemesh: missing vertex positions",
        ));
    }
    let vertex_count = p.len() as u32;
    if vertex_indices.iter().any(|&i| i >= vertex_count) {
        return Err(PbrtError::from(
            "Invalid trianglemesh: index out of range for provided P",
        ));
    }
    if !s.is_empty() && s.len() != p.len() {
        return Err(PbrtError::from(
            "Invalid trianglemesh: S must be empty or have the same length as P",
        ));
    }
    if !n.is_empty() && n.len() != p.len() {
        return Err(PbrtError::from(
            "Invalid trianglemesh: N must be empty or have the same length as P",
        ));
    }
    if !uv.is_empty() && uv.len() != p.len() {
        return Err(PbrtError::from(
            "Invalid trianglemesh: uv must be empty or have the same length as P",
        ));
    }
    Ok(())
}

pub fn create_triangle_mesh(
    o2w: &Transform,
    _: &Transform,
    reverse_orientation: bool,
    vertex_indices: Vec<u32>,
    p: Vec<Point3f>,
    s: Vec<Vector3f>,
    n: Vec<Vector3f>,
    uv: Vec<Point2f>,
    params: &ParameterDictionary,
) -> Result<Vec<Triangle>, PbrtError> {
    validate_triangle_mesh_params(&vertex_indices, &p, &s, &n, &uv)?;

    let two_sided = params.get_one_bool("twosided", true);
    let n_triangles = vertex_indices.len() / 3;
    let mesh = Arc::new(TriangleMesh::new(
        o2w,
        reverse_orientation,
        two_sided,
        vertex_indices,
        p,
        s,
        n,
        uv,
    ));
    let mut tris: Vec<Triangle> = Vec::with_capacity(n_triangles);
    for i in 0..n_triangles {
        let tri = Triangle::new(&mesh, i as u32);
        if tri.area() > 1e-16 {
            tris.push(tri);
        }
    }
    return Ok(tris);
}

type FloatTextureMap = HashMap<String, Arc<FloatTexture>>;

pub fn create_triangle_mesh_shape(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    params: &ParameterDictionary,
    float_textures: &FloatTextureMap,
) -> Result<Vec<Shape>, PbrtError> {
    let mut vertex_indices = Vec::new();
    let mut p: Vec<Point3f> = Vec::new();
    let mut s: Vec<Vector3f> = Vec::new();
    let mut n: Vec<Normal3f> = Vec::new();
    let mut uv: Vec<Vector2f> = Vec::new();

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
    // pbrt-v4 verbatim: leave `uv` empty when no "uv"/"st" parameter is
    // supplied. Per-face defaults (0,0)/(1,0)/(1,1) are produced at
    // intersect time in `get_uvs`, matching `InteractionFromIntersection`
    // in shapes.h. Pre-filling a shared `uv` array is
    // incorrect when vertices are shared across triangles because the
    // earliest triangle locks the vertex UV and later triangles inherit
    // an inconsistent value, producing constant-UV (black) patches.

    if let Some(ps) = params.get_points_ref("S") {
        let sz = ps.len() / 3;
        s.resize(sz, Vector3::zero());
        for i in 0..sz {
            s[i] = Vector3f::new(ps[3 * i + 0], ps[3 * i + 1], ps[3 * i + 2]);
        }
    }

    if let Some(ps) = params.get_points_ref("N") {
        let sz = ps.len() / 3;
        n.resize(sz, Normal3f::zero());
        for i in 0..sz {
            n[i] = Normal3f::new(ps[3 * i + 0], ps[3 * i + 1], ps[3 * i + 2]);
        }
    }

    if !vertex_indices.is_empty() && !p.is_empty() {
        let mesh = create_triangle_mesh(
            o2w,
            w2o,
            reverse_orientation,
            vertex_indices,
            p,
            s,
            n,
            uv,
            params,
        )?;
        let mesh: Vec<Shape> = mesh.into_iter().map(Shape::Triangle).collect();

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
        return Ok(mesh);
    } else {
        return Err(PbrtError::from("Invalid mesh"));
    }
}
