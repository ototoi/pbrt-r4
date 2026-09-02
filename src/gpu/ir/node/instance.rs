use super::node::NodeRef;
use super::transform::Transform;
#[derive(Clone)]
pub struct Instance {
    pub target: NodeRef,
    pub transform: Transform,
}
