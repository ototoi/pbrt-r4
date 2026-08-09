use super::base_interaction::BaseInteraction;
use super::medium_interaction::MediumInteraction;
use super::surface_interaction::SurfaceInteraction;
use crate::media::*;
use crate::util::base::*;
use crate::util::geometry::*;

use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum Interaction {
    Base(BaseInteraction),
    Surface(SurfaceInteraction),
    Medium(MediumInteraction),
}

impl Interaction {
    pub fn is_surface_interaction(&self) -> bool {
        match self {
            Self::Base(inter) => {
                return inter.n.length_squared() > 0.0;
            }
            Self::Surface(inter) => {
                return inter.n.length_squared() > 0.0;
            }
            _ => {
                return false;
            }
        }
    }

    pub fn as_surface_interaction(&self) -> Option<&SurfaceInteraction> {
        match self {
            Self::Base(_) => None,
            Self::Surface(inter) => {
                if inter.n.length_squared() > 0.0 {
                    Some(inter)
                } else {
                    return None;
                }
            }
            Self::Medium(_) => None,
        }
    }

    pub fn as_medium_interaction(&self) -> Option<&MediumInteraction> {
        match self {
            Self::Base(_) => None,
            Self::Surface(_) => None,
            Self::Medium(inter) => Some(inter),
        }
    }

    pub fn as_medium_interaction_mut(&mut self) -> Option<&mut MediumInteraction> {
        match self {
            Self::Base(_) => None,
            Self::Surface(_) => None,
            Self::Medium(inter) => Some(inter),
        }
    }

    pub fn get_p(&self) -> Point3f {
        match self {
            Self::Base(inter) => inter.p,
            Self::Surface(inter) => inter.p,
            Self::Medium(inter) => inter.p,
        }
    }

    pub fn get_p_error(&self) -> Vector3f {
        match self {
            Self::Base(inter) => inter.p_error,
            Self::Surface(inter) => inter.p_error,
            Self::Medium(inter) => inter.p_error,
        }
    }

    pub fn get_n(&self) -> Normal3f {
        match self {
            Self::Base(inter) => inter.n,
            Self::Surface(inter) => inter.n,
            Self::Medium(inter) => inter.n,
        }
    }

    pub fn get_uv(&self) -> Point2f {
        match self {
            Self::Base(inter) => inter.uv,
            Self::Surface(inter) => inter.uv,
            Self::Medium(inter) => inter.uv,
        }
    }

    pub fn get_time(&self) -> Float {
        match self {
            Self::Base(inter) => inter.time,
            Self::Surface(inter) => inter.time,
            Self::Medium(inter) => inter.time,
        }
    }

    pub fn set_time(&mut self, time: Float) {
        match self {
            Self::Base(inter) => inter.time = time,
            Self::Surface(inter) => inter.time = time,
            Self::Medium(inter) => inter.time = time,
        }
    }

    pub fn set_medium_interface(&mut self, medium_interface: &MediumInterface) {
        match self {
            Self::Base(inter) => inter.medium_interface = medium_interface.clone(),
            Self::Surface(inter) => inter.medium_interface = medium_interface.clone(),
            Self::Medium(inter) => inter.medium_interface = medium_interface.clone(),
        }
    }

    pub fn get_base_tuple(&self) -> (Point3f, Vector3f, Normal3f, Float) {
        match self {
            Self::Base(inter) => (inter.p, inter.p_error, inter.n, inter.time),
            Self::Surface(inter) => (inter.p, inter.p_error, inter.n, inter.time),
            Self::Medium(inter) => (inter.p, inter.p_error, inter.n, inter.time),
        }
    }

    pub fn spawn_ray(&self, d: &Vector3f) -> Ray {
        let (p, p_error, n, time) = self.get_base_tuple();
        let o = offset_ray_origin(&p, &p_error, &n, d);
        let mut r = Ray::new(&o, d, Float::INFINITY, time);
        r.medium = self.get_medium(d);
        return r;
    }

    pub fn spawn_ray_to(&self, it: &Interaction) -> Ray {
        let (ap, ap_error, an, atime) = self.get_base_tuple();
        let (bp, bp_error, bn, _btime) = it.get_base_tuple();
        let origin = offset_ray_origin(&ap, &ap_error, &an, &(bp - ap));
        let target = offset_ray_origin(&bp, &bp_error, &bn, &(origin - bp));
        let d = target - origin;
        let mut r = Ray::new(&origin, &d, Float::INFINITY, atime);
        r.medium = self.get_medium(&d);
        return r;
    }

    pub fn get_medium(&self, w: &Vector3f) -> Option<Arc<Medium>> {
        match self {
            Self::Base(_inter) => None,
            Self::Surface(inter) => inter.get_medium(w),
            Self::Medium(inter) => inter.get_medium(w),
        }
    }

    pub fn from_surface_sample(p: &Point3f, p_error: &Vector3f, n: &Normal3f) -> Self {
        let mut it = BaseInteraction::default();
        it.p = *p;
        it.p_error = *p_error;
        it.n = *n;
        return Self::Base(it);
    }

    pub fn from_light_sample(p: &Point3f, time: Float, medium_interface: &MediumInterface) -> Self {
        let mut it = BaseInteraction::default();
        it.p = *p;
        it.time = time;
        it.medium_interface = medium_interface.clone();
        return Self::Base(it);
    }
}

impl Default for Interaction {
    fn default() -> Self {
        Interaction::Base(BaseInteraction::default())
    }
}

impl From<BaseInteraction> for Interaction {
    fn from(value: BaseInteraction) -> Self {
        Self::Base(value)
    }
}

impl From<SurfaceInteraction> for Interaction {
    fn from(value: SurfaceInteraction) -> Self {
        Self::Surface(value)
    }
}

impl From<&SurfaceInteraction> for Interaction {
    fn from(value: &SurfaceInteraction) -> Self {
        Self::Surface(value.clone())
    }
}

impl From<&mut SurfaceInteraction> for Interaction {
    fn from(value: &mut SurfaceInteraction) -> Self {
        Self::Surface(value.clone())
    }
}

impl From<MediumInteraction> for Interaction {
    fn from(value: MediumInteraction) -> Self {
        Self::Medium(value)
    }
}

impl From<&MediumInteraction> for Interaction {
    fn from(value: &MediumInteraction) -> Self {
        Self::Medium(value.clone())
    }
}

impl
    From<(
        Point3f,
        Normal3f,
        Vector3f,
        Vector3f,
        Float,
        MediumInterface,
    )> for Interaction
{
    fn from(
        value: (
            Point3f,
            Normal3f,
            Vector3f,
            Vector3f,
            Float,
            MediumInterface,
        ),
    ) -> Self {
        let mut it = BaseInteraction::default();
        it.p = value.0;
        it.n = value.1;
        it.p_error = value.2;
        it.wo = value.3;
        it.time = value.4;
        it.medium_interface = value.5;
        return Self::Base(it);
    }
}
