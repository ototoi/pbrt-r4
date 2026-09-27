use super::Transform;

#[derive(Clone, Debug, PartialEq)]
pub struct Medium {
    pub name: String,
    pub kind: String,
    pub sigma_a: u32,
    pub sigma_s: u32,
    pub le: u32,
    pub g: f32,
    pub transform: Transform,
}
