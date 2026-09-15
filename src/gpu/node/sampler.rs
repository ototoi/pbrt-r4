use crate::paramdict::ParameterDictionary;

#[derive(Clone)]
pub struct Sampler {
    pub name: String,
    pub params: ParameterDictionary,
}
