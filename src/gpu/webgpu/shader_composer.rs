use std::collections::{HashMap, HashSet};

use crate::util::error::PbrtError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderModuleSpec {
    pub id: String,
    pub source: String,
    pub dependencies: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComposedShader {
    pub source: String,
    pub modules: Vec<String>,
}

pub fn compose(modules: Vec<ShaderModuleSpec>, entry: &str) -> Result<ComposedShader, PbrtError> {
    let registry = modules
        .into_iter()
        .map(|module| (module.id.clone(), module))
        .collect::<HashMap<_, _>>();
    if !registry.contains_key(entry) {
        return Err(PbrtError::error(&format!(
            "Shader entry module \"{entry}\" is not registered."
        )));
    }
    let mut order = Vec::new();
    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    visit(entry, &registry, &mut visiting, &mut visited, &mut order)?;
    let mut source = String::new();
    for id in &order {
        if let Some(module) = registry.get(id) {
            source.push_str(&format!("// module: {id}\n"));
            source.push_str(&module.source);
            source.push('\n');
        }
    }
    Ok(ComposedShader {
        source,
        modules: order,
    })
}

fn visit(
    id: &str,
    registry: &HashMap<String, ShaderModuleSpec>,
    visiting: &mut HashSet<String>,
    visited: &mut HashSet<String>,
    order: &mut Vec<String>,
) -> Result<(), PbrtError> {
    if visited.contains(id) {
        return Ok(());
    }
    if !visiting.insert(id.to_string()) {
        return Err(PbrtError::error(&format!(
            "Cycle detected in shader module dependency at \"{id}\"."
        )));
    }
    let module = registry.get(id).ok_or_else(|| {
        PbrtError::error(&format!(
            "Shader module dependency \"{id}\" is not registered."
        ))
    })?;
    for dependency in &module.dependencies {
        visit(dependency, registry, visiting, visited, order)?;
    }
    visiting.remove(id);
    visited.insert(id.to_string());
    order.push(id.to_string());
    Ok(())
}
