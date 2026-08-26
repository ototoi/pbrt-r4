pub mod wavefront;

use super::fragment::{Fragment, FragmentId};
use super::ShaderStage;

pub struct ShaderRecipe {
    pub label: &'static str,
    pub fragments: Vec<Fragment>,
    pub roots: Vec<FragmentId>,
    pub stages: Vec<ShaderStage>,
}
