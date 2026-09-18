//! Backend-independent post-order representation of texture graphs.
//!
//! This is the first compilation seam between the variable-sized Node IR and
//! backend evaluators.  Operation-specific lowering is intentionally kept for
//! the next stage; this module only establishes deterministic child-first
//! scheduling and shared-node interning.

use std::collections::HashMap;
use std::sync::Arc;

use crate::gpu::node::TextureNode;
use crate::util::error::PbrtError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureInstruction {
    /// Index into [`TextureProgram::nodes`].
    pub node: u32,
    /// Range into [`TextureProgram::operands`].
    pub operand_offset: u32,
    pub operand_count: u32,
}

#[derive(Clone)]
pub struct TextureProgram {
    /// Node identity table retained until operation-specific lowering is
    /// implemented.  Instructions refer to this table by index.
    pub nodes: Vec<Arc<TextureNode>>,
    /// Child-first instructions.  Every shared Node IR node is emitted once.
    pub instructions: Vec<TextureInstruction>,
    /// Post-order operand table containing instruction indices.
    pub operands: Vec<u32>,
    pub result: u32,
}

impl TextureProgram {
    pub fn compile(root: &Arc<TextureNode>) -> Result<Self, PbrtError> {
        let mut compiler = Compiler {
            nodes: Vec::new(),
            instructions: Vec::new(),
            operands: Vec::new(),
            nodes_by_ptr: HashMap::new(),
            visiting: Vec::new(),
        };
        let result = compiler.emit(root)?;
        Ok(Self {
            nodes: compiler.nodes,
            instructions: compiler.instructions,
            operands: compiler.operands,
            result,
        })
    }
}

struct Compiler {
    nodes: Vec<Arc<TextureNode>>,
    instructions: Vec<TextureInstruction>,
    operands: Vec<u32>,
    nodes_by_ptr: HashMap<usize, u32>,
    visiting: Vec<usize>,
}

impl Compiler {
    fn emit(&mut self, node: &Arc<TextureNode>) -> Result<u32, PbrtError> {
        let key = Arc::as_ptr(node) as usize;
        if self.visiting.contains(&key) {
            return Err(PbrtError::error("Texture graph contains a cycle."));
        }
        if let Some(&instruction) = self.nodes_by_ptr.get(&key) {
            return Ok(instruction);
        }

        let node_index = u32::try_from(self.nodes.len())
            .map_err(|_| PbrtError::error("Texture program node table exceeds u32."))?;
        self.nodes.push(node.clone());
        self.nodes_by_ptr.insert(key, node_index);
        self.visiting.push(key);

        let operand_offset = u32::try_from(self.operands.len())
            .map_err(|_| PbrtError::error("Texture program operand table exceeds u32."))?;
        let mut child_instructions = Vec::with_capacity(node.children.len());
        for child in &node.children {
            child_instructions.push(self.emit(child)?);
        }
        self.operands.extend(child_instructions);
        let operand_count = u32::try_from(node.children.len())
            .map_err(|_| PbrtError::error("Texture program child count exceeds u32."))?;
        self.visiting.pop();

        let instruction = u32::try_from(self.instructions.len())
            .map_err(|_| PbrtError::error("Texture program instruction table exceeds u32."))?;
        self.instructions.push(TextureInstruction {
            node: node_index,
            operand_offset,
            operand_count,
        });
        self.nodes_by_ptr.insert(key, instruction);
        Ok(instruction)
    }
}

#[cfg(test)]
mod tests {
    use super::TextureProgram;
    use crate::gpu::node::TextureNode;
    use std::sync::Arc;

    #[test]
    fn compile_emits_children_before_parent() {
        let child = Arc::new(TextureNode::new("child"));
        let mut root = TextureNode::new("root");
        root.children.push(child);
        let program = TextureProgram::compile(&Arc::new(root)).unwrap();

        assert_eq!(program.instructions.len(), 2);
        assert_eq!(program.instructions[0].node, 1);
        assert_eq!(program.instructions[1].node, 0);
        assert_eq!(program.operands, vec![0]);
        assert_eq!(program.result, 1);
    }

    #[test]
    fn compile_interns_shared_children() {
        let child = Arc::new(TextureNode::new("shared"));
        let mut root = TextureNode::new("root");
        root.children.push(child.clone());
        root.children.push(child);
        let program = TextureProgram::compile(&Arc::new(root)).unwrap();

        assert_eq!(program.instructions.len(), 2);
        assert_eq!(program.operands, vec![0, 0]);
    }
}
