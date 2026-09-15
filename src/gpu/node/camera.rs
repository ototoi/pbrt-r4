use crate::paramdict::ParameterDictionary;

#[derive(Clone)]
pub struct Camera {
    pub params: ParameterDictionary,
    pub medium: String,
}
