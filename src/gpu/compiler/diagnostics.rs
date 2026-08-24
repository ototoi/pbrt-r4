#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuSourceLocation {
    pub filename: String,
    pub line: u32,
    pub column: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuCompileError {
    UnsupportedShape {
        name: String,
        source: GpuSourceLocation,
    },
    UnsupportedSceneFeature {
        feature: &'static str,
        source: GpuSourceLocation,
    },
    MissingParameter {
        parameter: &'static str,
        source: GpuSourceLocation,
    },
    InvalidParameter {
        parameter: &'static str,
        detail: String,
        source: GpuSourceLocation,
    },
}
