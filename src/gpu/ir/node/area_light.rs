use crate::paramdict::ParameterDictionary;

/// Declarative area-light information attached to the Shape node it emits.
///
/// The shape and its transform remain owned by the node.  This component only
/// preserves the area-light declaration until Flat IR resolves it to an
/// instance and a light handle.
#[derive(Clone)]
pub struct AreaLight {
    pub name: String,
    pub params: ParameterDictionary,
}

#[derive(Clone)]
pub struct AreaLightComponent {
    pub area_light: AreaLight,
}
