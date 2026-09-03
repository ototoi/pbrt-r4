use crate::paramdict::ParameterDictionary;

#[derive(Clone)]
pub struct Filter {
    pub name: String,
    pub params: ParameterDictionary,
}
