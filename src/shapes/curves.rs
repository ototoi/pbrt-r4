//! Indexed CPU representation for a group of Bézier curves.
use super::curve::{
    curve_geometry_from_params, curve_split_count, CurveData, CurveSegment, CurveType, CurvesCommon,
};
use crate::cpu::aggregates::bvh::accel::BVHTree;
use crate::cpu::aggregates::bvh::build::SplitMethod;
use crate::interaction::ShapeIntersection;
use crate::paramdict::ParameterDictionary;
use crate::util::base::Float;
use crate::util::error::PbrtError;
use crate::util::geometry::{Bounds3f, Ray};
use crate::util::transform::transform::Transform;

pub struct CurvesShape {
    curves: IndexedCurves,
    tree: BVHTree,
}

impl CurvesShape {
    pub fn create(
        object_to_world: &Transform,
        world_to_object: &Transform,
        reverse_orientation: bool,
        shape_params: &[ParameterDictionary],
        split_method: SplitMethod,
        max_prims_in_node: usize,
    ) -> Result<Option<Self>, PbrtError> {
        let mut curves = IndexedCurves::create(
            object_to_world,
            world_to_object,
            reverse_orientation,
            shape_params,
        )?;
        if curves.is_empty() {
            return Ok(None);
        }
        let (tree, order) = BVHTree::new(
            curves.len(),
            |index| curves.segment(index).world_bound(),
            max_prims_in_node,
            split_method,
        );
        curves.reorder(&order);
        Ok(Some(Self { curves, tree }))
    }

    pub fn world_bound(&self) -> Bounds3f {
        self.tree.bounds()
    }

    pub fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        self.tree.intersect(ray, t_max, |range, mut limit| {
            let mut closest = None;
            for index in range {
                if let Some(hit) = self.curves.segment(index).intersect(ray, limit) {
                    limit = hit.t_hit;
                    closest = Some((limit, hit));
                }
            }
            closest
        })
    }

    pub fn intersect_p(&self, ray: &Ray, t_max: Float) -> bool {
        self.tree.intersect_p(ray, t_max, |mut range, limit| {
            range.any(|index| self.curves.segment(index).intersect_p(ray, limit))
        })
    }
}

struct IndexedCurves {
    group: CurvesCommon,
    data: Vec<CurveData>,
    segments: Vec<(usize, Float, Float)>,
}

impl IndexedCurves {
    fn create(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        shape_params: &[ParameterDictionary],
    ) -> Result<Self, PbrtError> {
        let Some(first) = shape_params.first() else {
            return Err(PbrtError::error(
                "create_curves_shape requires at least one curve parameter dictionary.",
            ));
        };
        let curve_type = CurveType::parse(&first.get_one_string("type", "flat"))?;
        let group = CurvesCommon::new(o2w, w2o, reverse_orientation, curve_type);
        let mut data = Vec::new();
        let mut segments = Vec::new();
        for params in shape_params {
            let geometry = curve_geometry_from_params(params)?;
            if geometry.curve_type != curve_type {
                return Err(PbrtError::error(
                    "Curve types in one curves shape must match.",
                ));
            }
            let count = curve_split_count(params)?;
            for (cp, w0, w1) in geometry.segments() {
                let index = data.len();
                data.push(CurveData::new(cp, w0, w1, &geometry.normals));
                segments.extend((0..count).map(|piece| {
                    (
                        index,
                        piece as Float / count as Float,
                        (piece + 1) as Float / count as Float,
                    )
                }));
            }
        }
        Ok(Self {
            group,
            data,
            segments,
        })
    }

    fn len(&self) -> usize {
        self.segments.len()
    }

    fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    fn reorder(&mut self, order: &[usize]) {
        self.segments = order.iter().map(|&index| self.segments[index]).collect();
    }

    fn segment(&self, index: usize) -> CurveSegment<'_> {
        let (data, u_min, u_max) = self.segments[index];
        CurveSegment {
            common: &self.data[data],
            group: &self.group,
            u_min,
            u_max,
        }
    }
}
