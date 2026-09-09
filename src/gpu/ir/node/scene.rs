use super::texture::TextureNode;
use std::sync::Arc;

#[derive(Clone)]
pub struct Scene {
    pub texture_nodes: Vec<Arc<TextureNode>>,
}
