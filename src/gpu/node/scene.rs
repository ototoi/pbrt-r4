use super::medium::Medium;
use super::texture::TextureNode;
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct Scene {
    pub texture_nodes: Vec<Arc<TextureNode>>,
    pub media: Vec<Arc<Medium>>,
}
