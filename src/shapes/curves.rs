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

mod hair_bvh;
use hair_bvh::HairBvh;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CurvesBvh {
    #[default]
    Segments,
    Hair,
}

impl CurvesBvh {
    pub fn from_env() -> Result<Self, PbrtError> {
        match std::env::var("PBRT_CURVES_BVH") {
            Ok(value) if value == "segments" => Ok(Self::Segments),
            Ok(value) if value == "hair" => Ok(Self::Hair),
            Err(std::env::VarError::NotPresent) => Ok(Self::default()),
            _ => Err(PbrtError::error(
                "PBRT_CURVES_BVH must be segments or hair.",
            )),
        }
    }
}

enum CurveTree {
    Segments(BVHTree, Vec<(usize, Float, Float)>),
    Hair(HairBvh),
}

pub struct CurvesShape {
    group: CurvesCommon,
    data: Vec<CurveData>,
    tree: CurveTree,
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
        Self::create_with_bvh(
            object_to_world,
            world_to_object,
            reverse_orientation,
            shape_params,
            split_method,
            max_prims_in_node,
            CurvesBvh::default(),
        )
    }

    pub fn create_with_bvh(
        object_to_world: &Transform,
        world_to_object: &Transform,
        reverse_orientation: bool,
        shape_params: &[ParameterDictionary],
        split_method: SplitMethod,
        max_prims_in_node: usize,
        backend: CurvesBvh,
    ) -> Result<Option<Self>, PbrtError> {
        let Some(first) = shape_params.first() else {
            return Err(PbrtError::error(
                "create_curves_shape requires at least one curve parameter dictionary.",
            ));
        };
        let curve_type = CurveType::parse(&first.get_one_string("type", "flat"))?;
        let group = CurvesCommon::new(
            object_to_world,
            world_to_object,
            reverse_orientation,
            curve_type,
        );
        let mut data = Vec::new();
        let mut counts = Vec::new();
        for params in shape_params {
            let geometry = curve_geometry_from_params(params)?;
            if geometry.curve_type != curve_type {
                return Err(PbrtError::error(
                    "Curve types in one curves shape must match.",
                ));
            }
            let count = curve_split_count(params)?;
            for (cp, w0, w1) in geometry.segments() {
                data.push(CurveData::new(cp, w0, w1, &geometry.normals));
                counts.push(count);
            }
        }
        if data.is_empty() {
            return Ok(None);
        }
        let tree = match backend {
            CurvesBvh::Segments => {
                let segments: Vec<_> = counts
                    .into_iter()
                    .enumerate()
                    .flat_map(|(index, count)| {
                        (0..count).map(move |piece| {
                            (
                                index,
                                piece as Float / count as Float,
                                (piece + 1) as Float / count as Float,
                            )
                        })
                    })
                    .collect();
                let (tree, order) = BVHTree::new(
                    segments.len(),
                    |index| curve_segment(&group, &data, segments[index]).world_bound(),
                    max_prims_in_node,
                    split_method,
                );
                CurveTree::Segments(
                    tree,
                    order.into_iter().map(|index| segments[index]).collect(),
                )
            }
            CurvesBvh::Hair => CurveTree::Hair(HairBvh::new(
                &group,
                &data,
                &counts,
                split_method,
                max_prims_in_node,
            )),
        };
        Ok(Some(Self { group, data, tree }))
    }

    pub fn world_bound(&self) -> Bounds3f {
        match &self.tree {
            CurveTree::Segments(tree, _) => tree.bounds(),
            CurveTree::Hair(tree) => tree.bounds(),
        }
    }

    pub fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        match &self.tree {
            CurveTree::Segments(tree, segments) => {
                tree.intersect(ray, t_max, |range, mut limit| {
                    let mut closest = None;
                    for &key in &segments[range] {
                        if let Some(hit) =
                            curve_segment(&self.group, &self.data, key).intersect(ray, limit)
                        {
                            limit = hit.t_hit;
                            closest = Some((limit, hit));
                        }
                    }
                    closest
                })
            }
            CurveTree::Hair(tree) => tree.intersect(&self.group, &self.data, ray, t_max),
        }
    }

    pub fn intersect_p(&self, ray: &Ray, t_max: Float) -> bool {
        match &self.tree {
            CurveTree::Segments(tree, segments) => tree.intersect_p(ray, t_max, |range, limit| {
                segments[range]
                    .iter()
                    .any(|&key| curve_segment(&self.group, &self.data, key).intersect_p(ray, limit))
            }),
            CurveTree::Hair(tree) => tree.intersect_p(&self.group, &self.data, ray, t_max),
        }
    }
}

fn curve_segment<'a>(
    group: &'a CurvesCommon,
    data: &'a [CurveData],
    (curve, u_min, u_max): (usize, Float, Float),
) -> CurveSegment<'a> {
    CurveSegment {
        common: &data[curve],
        group,
        u_min,
        u_max,
    }
}
