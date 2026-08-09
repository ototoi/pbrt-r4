use crate::cpu::aggregates::Accel;
use crate::cpu::primitive::*;
use crate::interaction::*;
use crate::paramdict::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;

use std::sync::Arc;

fn get_bounds(prims: &[Arc<Primitive>]) -> Bounds3f {
    let bounds: Vec<Bounds3f> = prims.iter().map(|p| p.bounds()).collect();
    let min = bounds
        .iter()
        .map(|b| b.min)
        .reduce(|a, b| {
            Vector3f::new(
                Float::min(a[0], b[0]),
                Float::min(a[1], b[1]),
                Float::min(a[2], b[2]),
            )
        })
        .unwrap();
    let max = bounds
        .iter()
        .map(|b| b.max)
        .reduce(|a, b| {
            Vector3f::new(
                Float::max(a[0], b[0]),
                Float::max(a[1], b[1]),
                Float::max(a[2], b[2]),
            )
        })
        .unwrap();
    Bounds3f::from(((min[0], min[1], min[2]), (max[0], max[1], max[2])))
}

#[derive(Clone)]
pub struct ExhaustiveAccel {
    pub prims: Vec<Arc<Primitive>>,
    pub bounds: Bounds3f,
    is_empty: bool,
}

impl ExhaustiveAccel {
    pub fn new(prims: &[Arc<Primitive>]) -> Self {
        let is_empty = prims.is_empty();
        let bounds = if is_empty {
            Bounds3f::default()
        } else {
            get_bounds(prims)
        };
        let prims = prims.to_vec();
        ExhaustiveAccel {
            prims,
            bounds,
            is_empty,
        }
    }

    pub fn bounds(&self) -> Bounds3f {
        self.bounds
    }

    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        if self.is_empty {
            return None;
        }
        if let Some((_, t_max)) = self.bounds.intersect_p(r, t_max) {
            let mut opt_isect = None;
            let mut t_max = t_max;
            for prim in self.prims.iter() {
                if let Some(isect) = prim.intersect(r, t_max) {
                    t_max = isect.t_hit;
                    opt_isect = Some(isect);
                }
            }
            return opt_isect;
        }
        None
    }

    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        if self.is_empty {
            return false;
        }
        if let Some((_, t_max)) = self.bounds.intersect_p(r, t_max) {
            for prim in self.prims.iter() {
                if prim.intersect_p(r, t_max) {
                    return true;
                }
            }
        }
        false
    }
}

pub fn create_exhaustive_accelerator(
    prims: &[Arc<Primitive>],
    _params: &ParameterDictionary,
) -> Result<Primitive, PbrtError> {
    Ok(Primitive::Accel(Accel::Exhaustive(Arc::new(
        ExhaustiveAccel::new(prims),
    ))))
}
