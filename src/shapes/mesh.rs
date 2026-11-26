use super::alphamask::AlphaMaskTest;
use super::triangle::{
    intersect_triangle, triangle_interaction, triangle_vertices, TriangleIntersection, TriangleMesh,
};
use crate::cpu::aggregates::bvh::accel::BVHTree;
use crate::cpu::aggregates::bvh::build::SplitMethod;
use crate::interaction::*;
use crate::util::base::*;
use crate::util::geometry::*;
use std::ops::Range;
use std::sync::Arc;

pub struct MeshShape {
    mesh: Arc<TriangleMesh>,
    faces: Vec<u32>,
    tree: BVHTree,
    alpha: Option<AlphaMaskTest>,
}

impl MeshShape {
    pub fn new(
        mesh: Arc<TriangleMesh>,
        faces: Vec<u32>,
        alpha: Option<AlphaMaskTest>,
        split_method: SplitMethod,
        max_prims_in_node: usize,
    ) -> Option<Self> {
        if faces.is_empty() {
            return None;
        }
        let bounds: Vec<Bounds3f> = faces
            .iter()
            .map(|face| {
                let (p0, p1, p2) = triangle_vertices(&mesh, *face);
                Bounds3f::new(&p0, &p1).union_p(&p2)
            })
            .collect();
        let (tree, order) = BVHTree::new(&bounds, max_prims_in_node, split_method);
        let faces = order.iter().map(|index| faces[*index]).collect();
        Some(Self {
            mesh,
            faces,
            tree,
            alpha,
        })
    }

    pub fn triangle_count(&self) -> usize {
        self.faces.len()
    }
    pub fn has_constant_zero_alpha(&self) -> bool {
        self.alpha
            .as_ref()
            .is_some_and(AlphaMaskTest::has_constant_zero_alpha)
    }
    pub fn world_bound(&self) -> Bounds3f {
        self.tree.bounds()
    }
    pub fn object_bound(&self) -> Bounds3f {
        self.mesh
            .world_to_object
            .transform_bounds(&self.world_bound())
    }

    pub fn area(&self) -> Float {
        self.faces.iter().fold(0.0, |area, face| {
            let (p0, p1, p2) = triangle_vertices(&self.mesh, *face);
            area + 0.5 * Vector3f::cross(&(p1 - p0), &(p2 - p0)).length()
        })
    }

    pub fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        if self
            .alpha
            .as_ref()
            .is_some_and(|test| !test.tests_intersection())
        {
            return None;
        }
        let mut best: Option<(Float, u32, TriangleIntersection, Option<SurfaceInteraction>)> = None;
        let has_alpha = self.alpha.is_some();
        self.tree
            .intersect(ray, t_max, |range: Range<usize>, mut limit| {
                let mut leaf_best = None;
                for slot in range {
                    let face = self.faces[slot];
                    let (p0, p1, p2) = triangle_vertices(&self.mesh, face);
                    let Some(ti) = intersect_triangle(ray, limit, p0, p1, p2) else {
                        continue;
                    };
                    let si = if has_alpha {
                        let si = triangle_interaction(&self.mesh, face, ti, ray.time, -ray.d);
                        if !self.alpha.as_ref().unwrap().accept(&si, ray) {
                            continue;
                        }
                        Some(si)
                    } else {
                        None
                    };
                    limit = ti.t;
                    leaf_best = Some((ti.t, face, ti, si));
                }
                leaf_best.map(|hit| (hit.0, hit))
            })
            .map(|hit| {
                best = Some(hit);
            });
        let (t, face, ti, intr) = best?;
        let intr =
            intr.unwrap_or_else(|| triangle_interaction(&self.mesh, face, ti, ray.time, -ray.d));
        Some(ShapeIntersection::new(intr, t))
    }

    pub fn intersect_p(&self, ray: &Ray, t_max: Float) -> bool {
        if self
            .alpha
            .as_ref()
            .is_some_and(|test| !test.tests_intersection_p())
        {
            return false;
        }
        self.tree.intersect_p(ray, t_max, |range, limit| {
            range.into_iter().any(|slot| {
                let face = self.faces[slot];
                let (p0, p1, p2) = triangle_vertices(&self.mesh, face);
                let Some(ti) = intersect_triangle(ray, limit, p0, p1, p2) else {
                    return false;
                };
                let Some(alpha) = &self.alpha else {
                    return true;
                };
                let si = triangle_interaction(&self.mesh, face, ti, ray.time, -ray.d);
                alpha.accept(&si, ray) && alpha.accept_shadow(&si, ray)
            })
        })
    }
}
