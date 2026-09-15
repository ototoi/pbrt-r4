use crate::paramdict::ParameterDictionary;

#[derive(Clone)]
pub struct Accelerator {
    pub name: String,
    pub params: ParameterDictionary,
}
