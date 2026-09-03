use crate::paramdict::ParameterDictionary;

#[derive(Clone)]
pub struct Material {
    pub name: String,
    pub kind: String,
    pub params: ParameterDictionary,
}
