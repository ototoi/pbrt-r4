use super::transform::Transform;
use crate::paramdict::ParameterDictionary;

#[derive(Clone)]
pub struct Medium {
    pub name: String,
    pub kind: String,
    pub params: ParameterDictionary,
    pub transform: Transform,
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct MediumInterface {
    pub inside_medium: String,
    pub outside_medium: String,
}

impl MediumInterface {
    pub fn new(inside: impl Into<String>, outside: impl Into<String>) -> Self {
        Self {
            inside_medium: inside.into(),
            outside_medium: outside.into(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inside_medium.is_empty() && self.outside_medium.is_empty()
    }
}
