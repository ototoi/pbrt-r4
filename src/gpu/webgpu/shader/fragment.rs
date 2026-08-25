use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FragmentId {
    Abi,
    Transform,
    Texture,
    Geometry,
    Sampling,
    AreaGeometry,
    AreaSampling,
    Emission,
    HardwareBindings,
    RayQueryTraversal,
    SoftwareBvhTraversal,
    EntryMain,
}

#[derive(Clone)]
pub struct Fragment {
    pub id: FragmentId,
    pub path: &'static str,
    pub source: &'static str,
    pub dependencies: Vec<FragmentId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FragmentError {
    Cycle(FragmentId),
    Unknown(FragmentId),
}

pub fn compose(fragments: &[Fragment], roots: &[FragmentId]) -> Result<String, FragmentError> {
    let mut output = String::new();
    let mut emitted = HashSet::new();
    let mut visiting = HashSet::new();

    for root in roots {
        emit(*root, fragments, &mut output, &mut emitted, &mut visiting)?;
    }
    Ok(output)
}

fn emit(
    id: FragmentId,
    fragments: &[Fragment],
    output: &mut String,
    emitted: &mut HashSet<FragmentId>,
    visiting: &mut HashSet<FragmentId>,
) -> Result<(), FragmentError> {
    if emitted.contains(&id) {
        return Ok(());
    }
    if !visiting.insert(id) {
        return Err(FragmentError::Cycle(id));
    }
    let fragment = fragments
        .iter()
        .find(|fragment| fragment.id == id)
        .ok_or(FragmentError::Unknown(id))?;
    for dependency in &fragment.dependencies {
        emit(*dependency, fragments, output, emitted, visiting)?;
    }
    output.push_str("// BEGIN pbrt-r4 shader fragment: ");
    output.push_str(fragment.path);
    output.push('\n');
    output.push_str(fragment.source);
    if !fragment.source.ends_with('\n') {
        output.push('\n');
    }
    output.push_str("// END pbrt-r4 shader fragment: ");
    output.push_str(fragment.path);
    output.push('\n');
    visiting.remove(&id);
    emitted.insert(id);
    Ok(())
}
