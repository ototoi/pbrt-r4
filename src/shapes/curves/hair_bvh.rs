//! AABB hierarchy over whole cubics with optional oriented bounds in its leaves.
use super::{CurveData, CurveSegment, CurvesCommon};
use crate::cpu::aggregates::bvh::{BVHTree, SplitMethod};
use crate::interaction::ShapeIntersection;
use crate::util::base::{gamma, Float, Vector3f};
use crate::util::geometry::{intersect_box, Bounds3f, Ray};
use crate::util::vecmath::coordinate_system;

pub(super) struct HairBvh {
    tree: BVHTree,
    leaves: Vec<HairLeaf>,
}

struct HairLeaf {
    curve: usize,
    segment_count: usize,
    oriented_bound: Option<OrientedBound>,
}

struct OrientedBound {
    axes: [Vector3f; 3],
    bounds: Bounds3f,
}

impl OrientedBound {
    fn new(curve: &CurveData) -> Option<Self> {
        let chord = curve.cp_obj[3] - curve.cp_obj[0];
        let length = chord.length();
        if length == 0.0 || !length.is_finite() {
            return None;
        }
        let axis = chord / length;
        let (u, v) = coordinate_system(&axis);
        let axes = [u, v, axis];
        let projected =
            |p: &Vector3f| Vector3f::new(axes[0].dot(p), axes[1].dot(p), axes[2].dot(p));

        let mut aligned = Bounds3f::default();
        let mut oriented = Bounds3f::default();
        let mut magnitude: Float = 0.0;
        for p in &curve.cp_obj {
            aligned = aligned.union_p(p);
            oriented = oriented.union_p(&projected(p));
            magnitude = magnitude.max(p.x.abs() + p.y.abs() + p.z.abs());
        }
        // The Bézier convex hull encloses the centerline. Expand its projection
        // by the maximum radius and round outward for control point evaluation.
        let radius = 0.5 * curve.width[0].max(curve.width[1]);
        let bounds = oriented.expand(radius + gamma(16.0) * (magnitude + radius));
        if bounds.surface_area() < aligned.expand(radius).surface_area() {
            Some(Self { axes, bounds })
        } else {
            None
        }
    }

    fn intersects(&self, ray: &Ray, t_max: Float) -> bool {
        let project = |p: &Vector3f| {
            Vector3f::new(
                self.axes[0].dot(p),
                self.axes[1].dot(p),
                self.axes[2].dot(p),
            )
        };
        let origin = project(&ray.o);
        let direction = project(&ray.d);
        // Dot product and slab rounding is significant for distant ray origins.
        let error = gamma(8.0)
            * (ray.o.x.abs()
                + ray.o.y.abs()
                + ray.o.z.abs()
                + self.bounds.min.abs().length()
                + self.bounds.max.abs().length());
        let bounds = self.bounds.expand(error);
        intersect_box(&bounds.min, &bounds.max, &origin, &direction, 0.0, t_max).0
    }
}

impl HairBvh {
    pub(super) fn new(
        group: &CurvesCommon,
        curves: &[CurveData],
        counts: &[usize],
        split_method: SplitMethod,
        max_prims_in_node: usize,
    ) -> Self {
        let (tree, order) = BVHTree::new(
            curves.len(),
            |curve| {
                CurveSegment {
                    common: &curves[curve],
                    group,
                    u_min: 0.0,
                    u_max: 1.0,
                }
                .world_bound()
            },
            max_prims_in_node,
            split_method,
        );
        let leaves = order
            .into_iter()
            .map(|curve| HairLeaf {
                curve,
                segment_count: counts[curve],
                oriented_bound: OrientedBound::new(&curves[curve]),
            })
            .collect();
        Self { tree, leaves }
    }

    pub(super) fn bounds(&self) -> Bounds3f {
        self.tree.bounds()
    }

    pub(super) fn intersect(
        &self,
        group: &CurvesCommon,
        curves: &[CurveData],
        ray: &Ray,
        t_max: Float,
    ) -> Option<ShapeIntersection> {
        let mut object_ray = None;
        self.tree.intersect(ray, t_max, |range, mut limit| {
            let mut closest = None;
            for leaf in &self.leaves[range] {
                let local_ray = object_ray.get_or_insert_with(|| group.object_ray(ray));
                if !leaf.passes_bound(local_ray, limit) {
                    continue;
                }
                for segment in leaf.segments(group, curves) {
                    if let Some(hit) = segment.intersect_object(local_ray, limit) {
                        limit = hit.t_hit;
                        closest = Some((limit, hit));
                    }
                }
            }
            closest
        })
    }

    pub(super) fn intersect_p(
        &self,
        group: &CurvesCommon,
        curves: &[CurveData],
        ray: &Ray,
        t_max: Float,
    ) -> bool {
        let mut object_ray = None;
        self.tree.intersect_p(ray, t_max, |range, limit| {
            self.leaves[range].iter().any(|leaf| {
                let local_ray = object_ray.get_or_insert_with(|| group.object_ray(ray));
                leaf.passes_bound(local_ray, limit)
                    && leaf
                        .segments(group, curves)
                        .any(|segment| segment.intersect_p_object(local_ray, limit))
            })
        })
    }
}

impl HairLeaf {
    fn passes_bound(&self, local_ray: &Ray, t_max: Float) -> bool {
        self.oriented_bound
            .as_ref()
            .is_none_or(|bound| bound.intersects(local_ray, t_max))
    }

    fn segments<'a>(
        &self,
        group: &'a CurvesCommon,
        curves: &'a [CurveData],
    ) -> impl Iterator<Item = CurveSegment<'a>> {
        let common = &curves[self.curve];
        let count = self.segment_count;
        (0..count).map(move |piece| CurveSegment {
            common,
            group,
            u_min: piece as Float / count as Float,
            u_max: (piece + 1) as Float / count as Float,
        })
    }
}
