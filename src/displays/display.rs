use crate::util::error::*;

pub struct DisplayTile {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub buffer: Vec<f32>,
}

pub trait Display: Sync {
    fn start(
        &mut self,
        title: &str,
        resolution: &[usize; 2],
        channel_names: &[&str],
    ) -> Result<(), PbrtError>;
    fn update(&mut self, tile: &DisplayTile) -> Result<(), PbrtError>;
    fn end(&mut self) -> Result<(), PbrtError>;
}
