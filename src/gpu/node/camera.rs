use super::medium::Medium;
use crate::paramdict::ParameterDictionary;
use std::sync::Arc;

#[derive(Clone)]
pub struct Camera {
    pub kind: String,
    pub params: ParameterDictionary,
    pub medium: Option<Arc<Medium>>,
}
