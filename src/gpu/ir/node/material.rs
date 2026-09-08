use crate::paramdict::ParameterDictionary;
use std::sync::Arc;

#[derive(Clone)]
pub struct Material {
    pub name: String,
    pub kind: String,
    pub params: ParameterDictionary,
    pub material_attributes: Vec<(String, Arc<Material>)>,
}
