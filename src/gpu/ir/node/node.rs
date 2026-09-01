use super::component::Component;
use super::transform::Transform;

use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: u32,
    pub name: String,
    pub transform: Transform,
    pub components: Vec<Component>,
    pub children: Vec<Arc<Node>>,
}
