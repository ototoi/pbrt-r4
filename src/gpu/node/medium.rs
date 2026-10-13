use super::transform::Transform;
use crate::paramdict::ParameterDictionary;
use std::sync::Arc;

#[derive(Clone)]
pub struct Medium {
    pub name: String,
    pub kind: String,
    pub params: ParameterDictionary,
    pub transform: Transform,
}

#[derive(Clone, Default)]
pub struct MediumInterface {
    pub inside: Option<Arc<Medium>>,
    pub outside: Option<Arc<Medium>>,
}

impl MediumInterface {
    pub fn new(inside: Option<Arc<Medium>>, outside: Option<Arc<Medium>>) -> Self {
        Self { inside, outside }
    }

    pub fn is_medium_transition(&self) -> bool {
        match (&self.inside, &self.outside) {
            (Some(inside), Some(outside)) => !Arc::ptr_eq(inside, outside),
            (Some(_), None) | (None, Some(_)) => true,
            (None, None) => false,
        }
    }
}
