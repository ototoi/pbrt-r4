#[derive(Debug, PartialEq, Eq, Copy, Clone, Default)]
pub enum RandomizeStrategy {
    None,
    PermuteDigits,
    #[default]
    FastOwen,
    Owen,
}
