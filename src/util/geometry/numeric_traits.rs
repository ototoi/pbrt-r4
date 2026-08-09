pub trait NumberType {
    fn abs(x: Self) -> Self;
}
impl NumberType for f32 {
    fn abs(x: Self) -> Self {
        Self::abs(x)
    }
}
impl NumberType for f64 {
    fn abs(x: Self) -> Self {
        Self::abs(x)
    }
}
impl NumberType for i32 {
    fn abs(x: Self) -> Self {
        Self::abs(x)
    }
}

pub trait FloatType {
    fn sqrt(x: Self) -> Self;
}
impl FloatType for f32 {
    fn sqrt(x: Self) -> Self {
        Self::sqrt(x)
    }
}
impl FloatType for f64 {
    fn sqrt(x: Self) -> Self {
        Self::sqrt(x)
    }
}
