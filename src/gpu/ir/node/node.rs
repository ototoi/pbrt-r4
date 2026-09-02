use super::component::Component;
use super::transform::Transform;

use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub name: String,
    pub transform: Transform,
    pub components: Vec<Component>,
    pub children: Vec<Arc<Node>>,
}

impl Node {
    pub fn new(name: &str) -> Self {
        Node {
            name: name.to_string(),
            transform: Transform::default(),
            components: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, child: Arc<Node>) {
        self.children.push(child);
    }
}
