use super::transform::Transform;
use crate::paramdict::ParameterDictionary;

#[derive(Clone)]
pub struct Medium {
    pub name: String,
    pub params: ParameterDictionary,
    pub transform: Transform,
}
