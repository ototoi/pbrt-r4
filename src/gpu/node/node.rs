use super::component::Component;
use super::transform::Transform;

use std::sync::{Arc, RwLock};

pub type NodeRef = Arc<RwLock<Node>>;

pub struct Node {
    pub name: String,
    pub transform: Transform,
    pub components: Vec<Component>,
    pub children: Vec<NodeRef>,
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

    pub fn add_child(&mut self, child: NodeRef) {
        self.children.push(child);
    }

    pub fn add_component(&mut self, component: Component) {
        self.components.push(component);
    }
}
