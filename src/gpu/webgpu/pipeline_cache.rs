use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub const MAX_PIPELINE_CACHE_BYTES: u64 = 256 * 1024 * 1024;

pub fn default_path(info: &wgpu::AdapterInfo) -> Option<PathBuf> {
    let key = wgpu::util::pipeline_cache_key(info)?;
    Some(
        dirs::cache_dir()?
            .join("pbrt-r4")
            .join("pipelines")
            .join(format!("{key}.bin")),
    )
}

pub fn load(path: &Path) -> Option<Vec<u8>> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return None,
        Err(error) => {
            log::warn!(
                "Failed to inspect pipeline cache {}: {error}",
                path.display()
            );
            return None;
        }
    };

    if !within_limit(metadata.len()) {
        log::warn!("Pipeline cache {} exceeds the size limit", path.display());
        return None;
    }

    match fs::read(path) {
        Ok(data) => Some(data),
        Err(error) => {
            log::warn!("Failed to read pipeline cache {}: {error}", path.display());
            None
        }
    }
}

pub fn save(path: &Path, data: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(parent)?;

    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(data)?;
    temporary
        .persist(path)
        .map(|_| ())
        .map_err(|error| error.error)
}

fn within_limit(len: u64) -> bool {
    len <= MAX_PIPELINE_CACHE_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_info() -> wgpu::AdapterInfo {
        wgpu::AdapterInfo::new(wgpu::DeviceType::Other, wgpu::Backend::Vulkan)
    }

    #[test]
    fn limit_is_inclusive() {
        assert!(within_limit(MAX_PIPELINE_CACHE_BYTES));
        assert!(!within_limit(MAX_PIPELINE_CACHE_BYTES + 1));
    }

    #[test]
    fn save_then_load_round_trips_without_leftovers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("cache.bin");
        save(&path, b"abc").unwrap();
        assert_eq!(load(&path).as_deref(), Some(&b"abc"[..]));
        let names: Vec<_> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(names, vec![std::ffi::OsString::from("cache.bin")]);
    }

    #[test]
    fn missing_file_loads_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load(&dir.path().join("none.bin")).is_none());
    }

    #[test]
    fn concurrent_saves_publish_one_complete_payload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.bin");
        let payloads: Vec<Vec<u8>> = (0..8u8).map(|i| vec![i; 1 << 20]).collect();
        std::thread::scope(|scope| {
            for payload in &payloads {
                let path = &path;
                scope.spawn(move || save(path, payload).unwrap());
            }
        });
        let saved = load(&path).unwrap();
        assert!(payloads.contains(&saved));
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn default_path_uses_user_cache_dir_and_key() {
        let info = wgpu::AdapterInfo {
            backend: wgpu::Backend::Vulkan,
            vendor: 0x10de,
            device: 0x2504,
            ..test_info()
        };
        let path = default_path(&info).unwrap();
        assert!(path.starts_with(dirs::cache_dir().unwrap().join("pbrt-r4").join("pipelines")));
        assert_eq!(
            path.file_name().unwrap(),
            format!("{}.bin", wgpu::util::pipeline_cache_key(&info).unwrap()).as_str()
        );
        assert!(default_path(&wgpu::AdapterInfo {
            backend: wgpu::Backend::Metal,
            ..info
        })
        .is_none());
    }
}
