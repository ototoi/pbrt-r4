#[derive(Clone, Debug, PartialEq)]
pub struct Transform {
    pub matrix: [f32; 16],
}

impl Default for Transform {
    fn default() -> Self {
        Transform {
            matrix: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }
}
