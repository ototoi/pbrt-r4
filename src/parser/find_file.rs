use super::parse_next_operation;
use super::read_file::read_file_source;
use super::session::resolve_include_path;
use crate::util::error::PbrtError;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum DependencyKind {
    Include,
    Import,
}

#[derive(Clone)]
struct Dependency {
    kind: DependencyKind,
    filename: String,
}

struct DependencyScanner<'a> {
    archive_dir: &'a Path,
    candidates: &'a HashSet<PathBuf>,
    active_files: HashSet<PathBuf>,
    parsed_dependencies: HashMap<PathBuf, Vec<Dependency>>,
    dependencies: HashSet<PathBuf>,
}

pub fn find_root_scene(archive_dir: &Path) -> Result<PathBuf, PbrtError> {
    let archive_dir = archive_dir.canonicalize()?;
    let mut scenes = Vec::new();
    collect_scene_files(&archive_dir, &mut scenes)?;
    scenes.sort();
    if scenes.is_empty() {
        return Err(PbrtError::from("Archive contains no .pbrt scene files."));
    }

    let candidates: HashSet<PathBuf> = scenes.iter().cloned().collect();
    let mut scanner = DependencyScanner {
        archive_dir: &archive_dir,
        candidates: &candidates,
        active_files: HashSet::new(),
        parsed_dependencies: HashMap::new(),
        dependencies: HashSet::new(),
    };
    for root in &scenes {
        let mut include_frames = vec![root.clone()];
        let mut work_dirs = vec![parent_dir(root)?];
        scanner.scan(root, &mut include_frames, &mut work_dirs)?;
    }

    let roots: Vec<PathBuf> = scenes
        .into_iter()
        .filter(|scene| !scanner.dependencies.contains(scene))
        .collect();
    match roots.as_slice() {
        [root] => Ok(root.clone()),
        [] => Err(PbrtError::from(
            "Archive contains no root .pbrt scene file; every scene file is a dependency.",
        )),
        _ => Err(PbrtError::from(format!(
            "Archive contains multiple root scene files: {}",
            roots
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
}

fn collect_scene_files(dir: &Path, scenes: &mut Vec<PathBuf>) -> Result<(), PbrtError> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_scene_files(&path, scenes)?;
        } else if file_type.is_file() && path.extension().is_some_and(|ext| ext == "pbrt") {
            scenes.push(path.canonicalize()?);
        }
    }
    Ok(())
}

impl DependencyScanner<'_> {
    fn scan(
        &mut self,
        path: &Path,
        include_frames: &mut Vec<PathBuf>,
        work_dirs: &mut Vec<PathBuf>,
    ) -> Result<(), PbrtError> {
        let path = path.canonicalize()?;
        if !path.starts_with(self.archive_dir) || !self.active_files.insert(path.clone()) {
            return Ok(());
        }

        let parsed = self.parse_dependencies(&path)?;
        for dependency in parsed {
            let resolved = match dependency.kind {
                DependencyKind::Include => {
                    resolve_include_path(&dependency.filename, include_frames.iter())
                }
                DependencyKind::Import => resolve_import_path(&dependency.filename, work_dirs),
            };
            let Some(resolved) = resolved else {
                continue;
            };
            let Ok(resolved) = resolved.canonicalize() else {
                continue;
            };
            if !resolved.starts_with(self.archive_dir) {
                continue;
            }
            if self.candidates.contains(&resolved) {
                self.dependencies.insert(resolved.clone());
            }

            let Some(parent) = resolved.parent() else {
                continue;
            };
            work_dirs.push(parent.to_path_buf());
            match dependency.kind {
                DependencyKind::Include => {
                    include_frames.push(resolved.clone());
                    self.scan(&resolved, include_frames, work_dirs)?;
                    include_frames.pop();
                }
                DependencyKind::Import => {
                    let mut import_frames = vec![resolved.clone()];
                    self.scan(&resolved, &mut import_frames, work_dirs)?;
                }
            }
            work_dirs.pop();
        }

        self.active_files.remove(&path);
        Ok(())
    }

    fn parse_dependencies(&mut self, path: &Path) -> Result<Vec<Dependency>, PbrtError> {
        if let Some(dependencies) = self.parsed_dependencies.get(path) {
            return Ok(dependencies.clone());
        }

        let source = read_file_source(path)?;
        let mut input = source.as_str();
        let mut dependencies = Vec::new();
        while let Some((remaining, _, operation)) = parse_next_operation(&source, input)? {
            let kind = match operation.name.as_str() {
                "Include" => Some(DependencyKind::Include),
                "Import" => Some(DependencyKind::Import),
                _ => None,
            };
            if let Some(kind) = kind {
                if operation.params.is_none() {
                    return Err(PbrtError::from(format!(
                        "{} requires parameters.",
                        operation.name
                    )));
                }
                let filename = operation
                    .args
                    .as_ref()
                    .and_then(|args| args.get_strings("arg1").into_iter().next())
                    .ok_or_else(|| {
                        PbrtError::from(format!("{} requires a filename.", operation.name))
                    })?;
                dependencies.push(Dependency { kind, filename });
            }
            input = remaining;
        }
        self.parsed_dependencies
            .insert(path.to_path_buf(), dependencies.clone());
        Ok(dependencies)
    }
}

fn resolve_import_path(filename: &str, work_dirs: &[PathBuf]) -> Option<PathBuf> {
    let filename = Path::new(filename);
    if filename.is_absolute() {
        return filename.exists().then(|| filename.to_path_buf());
    }
    work_dirs
        .iter()
        .rev()
        .map(|dir| dir.join(filename))
        .find(|path| path.exists())
        .or_else(|| filename.exists().then(|| filename.to_path_buf()))
}

fn parent_dir(path: &Path) -> Result<PathBuf, PbrtError> {
    path.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| PbrtError::from("scene file has no parent directory"))
}
