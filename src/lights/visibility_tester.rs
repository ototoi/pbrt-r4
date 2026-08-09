use crate::cpu::integrators::IntegratorBase;
use crate::interaction::*;
use crate::util::base::*;

#[derive(Clone)]
pub struct VisibilityTester {
    pub p0: Interaction,
    pub p1: Interaction,
}

impl VisibilityTester {
    pub fn new() -> Self {
        VisibilityTester {
            p0: Interaction::default(),
            p1: Interaction::default(),
        }
    }

    pub fn unoccluded(&self, base: &IntegratorBase) -> bool {
        let ray = self.p0.spawn_ray_to(&self.p1);
        return !base.intersect_p(&ray, 1.0 - SHADOW_EPSILON);
    }
}

impl From<(&Interaction, &Interaction)> for VisibilityTester {
    fn from(value: (&Interaction, &Interaction)) -> Self {
        VisibilityTester {
            p0: value.0.clone(),
            p1: value.1.clone(),
        }
    }
}

impl From<(Interaction, Interaction)> for VisibilityTester {
    fn from(value: (Interaction, Interaction)) -> Self {
        VisibilityTester {
            p0: value.0,
            p1: value.1,
        }
    }
}
