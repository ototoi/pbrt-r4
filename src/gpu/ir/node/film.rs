use crate::paramdict::ParameterDictionary;

#[derive(Clone)]
pub struct Film {
    pub name: String,
    pub params: ParameterDictionary,
}
