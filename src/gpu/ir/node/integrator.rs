use crate::paramdict::ParameterDictionary;

#[derive(Clone)]
pub struct Integrator {
    pub name: String,
    pub params: ParameterDictionary,
}
