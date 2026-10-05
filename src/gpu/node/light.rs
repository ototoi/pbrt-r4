use super::medium::Medium;
use super::transform::Transform;
use crate::paramdict::ParameterDictionary;
use std::sync::Arc;

#[derive(Clone)]
pub struct Light {
    pub name: String,
    pub params: ParameterDictionary,
    pub transform: Transform,
    pub medium: Option<Arc<Medium>>,
}
