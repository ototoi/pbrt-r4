use std::{error::Error, num::ParseIntError};

#[derive(Debug, Copy, Clone)]
pub enum PbrtErrorKind {
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct PbrtError {
    pub kind: PbrtErrorKind,
    pub msg: String,
}

impl PbrtError {
    pub fn new(kind: PbrtErrorKind, msg: &str) -> Self {
        PbrtError {
            kind,
            msg: String::from(msg),
        }
    }
    pub fn warning(msg: &str) -> Self {
        PbrtError::new(PbrtErrorKind::Warning, msg)
    }
    pub fn error(msg: &str) -> Self {
        PbrtError::new(PbrtErrorKind::Error, msg)
    }
}

impl std::error::Error for PbrtError {}

impl std::fmt::Display for PbrtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "{}", &self.msg);
    }
}

impl From<&str> for PbrtError {
    fn from(value: &str) -> Self {
        return PbrtError::error(value);
    }
}

impl From<String> for PbrtError {
    fn from(value: String) -> Self {
        return PbrtError::error(&value);
    }
}

impl From<std::io::Error> for PbrtError {
    fn from(value: std::io::Error) -> Self {
        let msg = value.to_string();
        return PbrtError::error(&msg);
    }
}

impl From<ParseIntError> for PbrtError {
    fn from(value: ParseIntError) -> Self {
        let msg = value.to_string();
        return PbrtError::error(&msg);
    }
}

impl From<Box<dyn Error>> for PbrtError {
    fn from(value: Box<dyn Error>) -> Self {
        let msg = value.to_string();
        return PbrtError::error(&msg);
    }
}
