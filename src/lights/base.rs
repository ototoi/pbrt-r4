use crate::media::*;
use crate::util::transform::*;

/// pbrt-v4 `class LightBase` collects the data every `Light`
/// subclass shares: the light type discriminant, the transform from
/// light space to render/world space, and the surrounding
/// `MediumInterface`.
#[derive(Clone, Default, Debug)]
pub struct LightBase {
    pub flags: u32,
    pub medium_interface: MediumInterface,
    pub render_from_light: Transform,
}

impl LightBase {
    pub fn new(
        flags: u32,
        render_from_light: &Transform,
        medium_interface: &MediumInterface,
    ) -> Self {
        LightBase {
            flags,
            medium_interface: medium_interface.clone(),
            render_from_light: render_from_light.clone(),
        }
    }

    pub fn world_to_light(&self) -> Transform {
        Transform::inverse(&self.render_from_light)
    }
}
