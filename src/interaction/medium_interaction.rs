use crate::media::*;
use crate::util::base::*;
use crate::util::geometry::*;

use std::fmt::{Debug, Formatter, Result};
use std::sync::Arc;

#[derive(Clone)]
pub struct MediumInteraction {
    pub p: Point3f,
    pub p_error: Vector3f,
    pub n: Normal3f,
    pub uv: Point2f,
    pub time: Float,
    pub wo: Vector3f,
    pub medium_interface: MediumInterface,
    pub phase: Arc<PhaseFunction>,
}

impl MediumInteraction {
    /// Build a MediumInteraction at the scattering event `p`, carrying the
    /// medium the ray was traveling through. Mirrors v4's
    /// `MediumInteraction(p, wo, time, medium, phase)` — without this, a
    /// shadow ray spawned from the event would lose the medium and skip
    /// the ratio-tracking transmittance estimate in `SampleLd`, leaving
    /// direct lighting un-attenuated and the render systematically too
    /// bright.
    pub fn new(
        p: &Point3f,
        wo: &Vector3f,
        time: Float,
        medium: &Option<Arc<Medium>>,
        phase: &Arc<PhaseFunction>,
    ) -> Self {
        let p = *p;
        let wo = *wo;
        let medium_interface = MediumInterface::from(medium);
        let phase = phase.clone();
        MediumInteraction {
            p,
            p_error: Vector3f::zero(),
            n: Normal3f::zero(),
            uv: Point2f::zero(),
            time,
            wo,
            medium_interface,
            phase,
        }
    }

    pub fn get_base_tuple(&self) -> (Point3f, Vector3f, Normal3f, Float) {
        return (self.p, self.p_error, self.n, self.time);
    }

    pub fn spawn_ray(&self, d: &Vector3f) -> Ray {
        let (p, p_error, n, time) = self.get_base_tuple();
        let o = offset_ray_origin(&p, &p_error, &n, d);
        let mut r = Ray::new(&o, d, Float::INFINITY, time);
        r.medium = self.get_medium(d);
        return r;
    }

    pub fn get_medium(&self, w: &Vector3f) -> Option<Arc<Medium>> {
        if Vector3f::dot(w, &self.n) > 0.0 {
            return self.medium_interface.get_outside();
        } else {
            return self.medium_interface.get_inside();
        }
    }
}

impl Debug for MediumInteraction {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("MediumInteraction")
            .field("p", &self.p)
            .field("p_error", &self.p_error)
            .field("n", &self.n)
            .field("uv", &self.uv)
            .field("time", &self.time)
            .field("wo", &self.wo)
            .field("medium_interface", &self.medium_interface)
            .finish()
    }
}
