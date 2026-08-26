#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceLocation {
    pub filename: String,
    pub line: u32,
    pub column: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompileError {
    UnsupportedShape {
        name: String,
        source: SourceLocation,
    },
    UnsupportedSceneFeature {
        feature: &'static str,
        source: SourceLocation,
    },
    MissingParameter {
        parameter: &'static str,
        source: SourceLocation,
    },
    InvalidParameter {
        parameter: &'static str,
        detail: String,
        source: SourceLocation,
    },
}
