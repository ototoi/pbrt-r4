use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use super::mipmap::{F32MIPMapImage, MIPMap, MIPMapLevelStorage};
use crate::base::texture::TexInfo;
use crate::util::base::Float;
use crate::util::error::PbrtError;
use crate::util::imageio::ColorEncoding;
use crate::util::spectrum::RGBSpectrum;

use crypto::digest::Digest;
use crypto::md5::Md5;
use serde::{Deserialize, Serialize};

pub const MIPMAP_CACHE_FORMAT_VERSION: u32 = 1;
const MAX_CACHE_LEVELS: usize = 64;
const MAX_CACHE_DIMENSION: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum CacheView {
    Float,
    Spectrum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum CacheStorageKind {
    F32,
    F16,
    U8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct CacheLevelInfo {
    width: usize,
    height: usize,
    channels: usize,
    byte_length: usize,
    checksum: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CacheMetadata {
    format_version: u32,
    texinfo: TexInfo,
    image_file_hash: String,
    view: CacheView,
    storage_kind: CacheStorageKind,
    encoding: ColorEncoding,
    channels: usize,
    channel_layout: String,
    levels: Vec<CacheLevelInfo>,
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut md5 = Md5::new();
    md5.input(bytes);
    md5.result_str()
}

fn hash(s: &str) -> String {
    hash_bytes(s.as_bytes())
}

pub fn get_mipmap_file_hash(path: &str) -> Result<String, PbrtError> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(hash_bytes(&bytes))
}

pub fn get_mipmap_cache_dir(texinfo: &TexInfo) -> String {
    let temp_dir = env::temp_dir();
    let fullpath = fs::canonicalize(&texinfo.filename)
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_else(|_| texinfo.filename.clone());
    let filename = Path::new(&fullpath)
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "mipmap".to_string());
    temp_dir
        .join("pbrt")
        .join("textures")
        .join(filename)
        .join(hash(&fullpath))
        .join(format!(
            "v{}-{}",
            MIPMAP_CACHE_FORMAT_VERSION,
            hash(&texinfo.to_string())
        ))
        .to_string_lossy()
        .into_owned()
}

pub fn remove_mipmap_cache(dir: &str) -> Result<bool, PbrtError> {
    let path = Path::new(dir);
    if !path.exists() {
        return Ok(false);
    }
    fs::remove_dir_all(path)?;
    Ok(true)
}

fn layout(channels: usize) -> Result<&'static str, PbrtError> {
    match channels {
        1 => Ok("Y"),
        3 => Ok("RGB"),
        4 => Ok("RGBA"),
        _ => Err(PbrtError::error("Invalid MIPMap cache channel count.")),
    }
}

fn storage_kind(image: &F32MIPMapImage) -> Result<CacheStorageKind, PbrtError> {
    Ok(match image.data {
        MIPMapLevelStorage::F32(_) => CacheStorageKind::F32,
        MIPMapLevelStorage::F16(_) => CacheStorageKind::F16,
        MIPMapLevelStorage::U8 { .. } => CacheStorageKind::U8,
    })
}

fn validate_level(image: &F32MIPMapImage) -> Result<(), PbrtError> {
    if image.channels == 0 || image.resolution.0 == 0 || image.resolution.1 == 0 {
        return Err(PbrtError::error("Invalid MIPMap cache level dimensions."));
    }
    Ok(())
}

fn write_level(file: &mut File, image: &F32MIPMapImage) -> Result<(usize, String), PbrtError> {
    const BUFFER_BYTES: usize = 64 * 1024;

    let mut digest = Md5::new();
    let byte_length = match &image.data {
        MIPMapLevelStorage::F32(data) => {
            let mut bytes = Vec::with_capacity(BUFFER_BYTES);
            for values in data.chunks(BUFFER_BYTES / std::mem::size_of::<f32>()) {
                bytes.clear();
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                digest.input(&bytes);
                file.write_all(&bytes)?;
            }
            data.len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| PbrtError::error("MIPMap cache level is too large."))?
        }
        MIPMapLevelStorage::F16(data) => {
            let mut bytes = Vec::with_capacity(BUFFER_BYTES);
            for values in data.chunks(BUFFER_BYTES / std::mem::size_of::<u16>()) {
                bytes.clear();
                for value in values {
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                digest.input(&bytes);
                file.write_all(&bytes)?;
            }
            data.len()
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| PbrtError::error("MIPMap cache level is too large."))?
        }
        MIPMapLevelStorage::U8 { data, .. } => {
            digest.input(data);
            file.write_all(data)?;
            data.len()
        }
    };

    Ok((byte_length, digest.result_str()))
}

fn cache_layout<T>(mipmap: &MIPMap<T>) -> Result<(CacheStorageKind, usize, String), PbrtError> {
    let first = mipmap
        .storage
        .pyramid
        .first()
        .ok_or_else(|| PbrtError::error("Cannot cache an empty MIPMap."))?;
    validate_level(first)?;
    Ok((
        storage_kind(first)?,
        first.channels,
        layout(first.channels)?.to_string(),
    ))
}

fn validate_pyramid<T>(
    mipmap: &MIPMap<T>,
    kind: CacheStorageKind,
    channels: usize,
) -> Result<(), PbrtError> {
    for image in &mipmap.storage.pyramid {
        validate_level(image)?;
        if image.channels != channels || storage_kind(image)? != kind {
            return Err(PbrtError::error(
                "MIPMap cache levels have inconsistent layout.",
            ));
        }
    }
    Ok(())
}

fn write_cache<T>(
    dir: &str,
    texinfo: &TexInfo,
    file_hash: &str,
    view: CacheView,
    mipmap: &MIPMap<T>,
) -> Result<(), PbrtError> {
    let (kind, channels, channel_layout) = cache_layout(mipmap)?;
    validate_pyramid(mipmap, kind, channels)?;
    let target = Path::new(dir);
    if target.exists() {
        if target.join("COMPLETE").exists() {
            return Ok(());
        }
        fs::remove_dir_all(target)?;
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| PbrtError::error("Invalid system clock for cache write."))?
        .as_nanos();
    let temporary = target.with_extension(format!("tmp.{}.{}", std::process::id(), nonce));
    fs::create_dir_all(&temporary)?;
    let result = (|| {
        let mut levels = Vec::with_capacity(mipmap.storage.pyramid.len());
        for (index, image) in mipmap.storage.pyramid.iter().enumerate() {
            let mut file = File::create(temporary.join(format!("level-{index:03}.bin")))?;
            let (byte_length, checksum) = write_level(&mut file, image)?;
            file.sync_all()?;
            levels.push(CacheLevelInfo {
                width: image.resolution.0,
                height: image.resolution.1,
                channels: image.channels,
                byte_length,
                checksum,
            });
        }
        let metadata = CacheMetadata {
            format_version: MIPMAP_CACHE_FORMAT_VERSION,
            texinfo: texinfo.clone(),
            image_file_hash: file_hash.to_string(),
            view,
            storage_kind: kind,
            encoding: texinfo.encoding,
            channels,
            channel_layout,
            levels,
        };
        let metadata_path = temporary.join("metadata.json");
        let mut file = File::create(metadata_path)?;
        serde_json::to_writer_pretty(&mut file, &metadata)
            .map_err(|e| PbrtError::error(&format!("Cannot write MIPMap cache metadata: {e}")))?;
        file.sync_all()?;
        File::create(temporary.join("COMPLETE"))?.sync_all()?;
        fs::rename(&temporary, target)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&temporary);
    }
    result
}

fn read_level(
    info: &CacheLevelInfo,
    path: &Path,
    kind: CacheStorageKind,
    encoding: ColorEncoding,
) -> Result<F32MIPMapImage, PbrtError> {
    if info.width == 0
        || info.height == 0
        || info.width > MAX_CACHE_DIMENSION
        || info.height > MAX_CACHE_DIMENSION
        || !matches!(info.channels, 1 | 3 | 4)
    {
        return Err(PbrtError::error("Invalid MIPMap cache level metadata."));
    }
    let elements = info
        .width
        .checked_mul(info.height)
        .and_then(|v| v.checked_mul(info.channels))
        .ok_or_else(|| PbrtError::error("MIPMap cache level is too large."))?;
    let element_size = match kind {
        CacheStorageKind::F32 => 4,
        CacheStorageKind::F16 => 2,
        CacheStorageKind::U8 => 1,
    };
    if info.byte_length
        != elements
            .checked_mul(element_size)
            .ok_or_else(|| PbrtError::error("MIPMap cache level is too large."))?
    {
        return Err(PbrtError::error(
            "MIPMap cache byte length is inconsistent.",
        ));
    }
    let byte_length = u64::try_from(info.byte_length)
        .map_err(|_| PbrtError::error("MIPMap cache level is too large."))?;
    let file = File::open(path)?;
    if file.metadata()?.len() != byte_length {
        return Err(PbrtError::error("MIPMap cache level checksum mismatch."));
    }
    let max_read = byte_length
        .checked_add(1)
        .ok_or_else(|| PbrtError::error("MIPMap cache level is too large."))?;
    let mut bytes = Vec::with_capacity(info.byte_length);
    file.take(max_read).read_to_end(&mut bytes)?;
    if bytes.len() != info.byte_length || hash_bytes(&bytes) != info.checksum {
        return Err(PbrtError::error("MIPMap cache level checksum mismatch."));
    }
    let data = match kind {
        CacheStorageKind::F32 => MIPMapLevelStorage::F32(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        ),
        CacheStorageKind::F16 => MIPMapLevelStorage::F16(
            bytes
                .chunks_exact(2)
                .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())))
                .collect(),
        ),
        CacheStorageKind::U8 => MIPMapLevelStorage::U8 {
            data: bytes,
            encoding,
        },
    };
    Ok(F32MIPMapImage {
        resolution: (info.width, info.height),
        channels: info.channels,
        data,
    })
}

fn load_cache<T>(
    dir: &str,
    file_hash: &str,
    expected: &TexInfo,
    view: CacheView,
) -> Result<MIPMap<T>, PbrtError>
where
    T: Default
        + std::fmt::Debug
        + Copy
        + std::ops::Add<T, Output = T>
        + std::ops::Mul<Float, Output = T>,
    F32MIPMapImage: super::mipmap::MIPMapImage<T> + for<'a> From<(&'a [T], (usize, usize))>,
{
    let dir = Path::new(dir);
    let metadata_path = dir.join("metadata.json");
    if !dir.join("COMPLETE").exists() {
        return Err(PbrtError::error("MIPMap cache is incomplete."));
    }
    let metadata: CacheMetadata = serde_json::from_reader(File::open(metadata_path)?)
        .map_err(|e| PbrtError::error(&format!("Invalid MIPMap cache metadata: {e}")))?;
    if metadata.format_version != MIPMAP_CACHE_FORMAT_VERSION
        || metadata.image_file_hash != file_hash
        || metadata.texinfo != *expected
        || metadata.encoding != expected.encoding
        || metadata.view != view
        || metadata.levels.is_empty()
        || metadata.levels.len() > MAX_CACHE_LEVELS
        || metadata.channels != metadata.levels[0].channels
        || metadata.channel_layout != layout(metadata.channels)?
    {
        return Err(PbrtError::error("MIPMap cache metadata mismatch."));
    }
    let mut pyramid = Vec::with_capacity(metadata.levels.len());
    for (index, info) in metadata.levels.iter().enumerate() {
        if info.channels != metadata.channels {
            return Err(PbrtError::error("MIPMap cache channel layout mismatch."));
        }
        pyramid.push(read_level(
            info,
            &dir.join(format!("level-{index:03}.bin")),
            metadata.storage_kind,
            metadata.encoding,
        )?);
    }
    Ok(MIPMap::make_from_pyramid(
        pyramid,
        expected.filter,
        expected.max_aniso,
        expected.swrap_mode,
        expected.twrap_mode,
    ))
}

pub fn load_float_mipmap_cache(
    dir: &str,
    file_hash: &str,
    expected: &TexInfo,
) -> Result<MIPMap<Float>, PbrtError> {
    load_cache(dir, file_hash, expected, CacheView::Float)
}

pub fn save_float_mipmap_cache(
    dir: &str,
    texinfo: &TexInfo,
    file_hash: &str,
    mipmap: &MIPMap<Float>,
) -> Result<(), PbrtError> {
    write_cache(dir, texinfo, file_hash, CacheView::Float, mipmap)
}

pub fn load_spectrum_mipmap_cache(
    dir: &str,
    file_hash: &str,
    expected: &TexInfo,
) -> Result<MIPMap<RGBSpectrum>, PbrtError> {
    load_cache(dir, file_hash, expected, CacheView::Spectrum)
}

pub fn save_spectrum_mipmap_cache(
    dir: &str,
    texinfo: &TexInfo,
    file_hash: &str,
    mipmap: &MIPMap<RGBSpectrum>,
) -> Result<(), PbrtError> {
    write_cache(dir, texinfo, file_hash, CacheView::Spectrum, mipmap)
}
