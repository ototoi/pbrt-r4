#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    pub filename: String,
}

impl Output {
    pub fn from_flat(output: crate::gpu::ir::flat::Output) -> Self {
        Self {
            filename: output.filename,
        }
    }
}
