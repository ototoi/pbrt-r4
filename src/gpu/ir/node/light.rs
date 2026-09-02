use super::transform::Transform;
use crate::paramdict::ParameterDictionary;

#[derive(Clone)]
pub struct Light {
    pub name: String,
    pub params: ParameterDictionary,
    pub transform: Transform,
    pub medium: String,
}
