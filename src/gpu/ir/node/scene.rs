use super::texture::Texture;
use std::sync::Arc;

#[derive(Clone)]
pub struct Scene {
    pub textures: Vec<Arc<Texture>>,
}
