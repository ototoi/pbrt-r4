use crate::paramdict::ParameterDictionary;

#[derive(Clone)]
pub struct Camera {
    pub kind: String,
    pub params: ParameterDictionary,
    pub medium: String,
}
