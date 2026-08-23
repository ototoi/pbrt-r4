use crate::options::PbrtOptions;
use crate::paramdict::*;

use crate::shapes::*;
use crate::textures::*;
use crate::util::error::*;
use crate::util::imageio::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.
use crate::util::spectrum::*;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Copy, Clone)]
enum MipmapCacheValueType {
    Float,
    Spectrum,
}

fn get_typed_mipmap_cache_dir(texinfo: &TexInfo, value_type: MipmapCacheValueType) -> String {
    let suffix = match value_type {
        MipmapCacheValueType::Float => "float",
        MipmapCacheValueType::Spectrum => "spectrum",
    };
    format!("{}/{}", get_mipmap_cache_dir(texinfo), suffix)
}

fn texture_cache_enabled() -> bool {
    PbrtOptions::get().texture_cache
}

#[derive(Default)]
pub struct ImageMapMIPMapCache {
    float: HashMap<String, Arc<MIPMap<Float>>>,
    float_build_locks: HashMap<String, Arc<Mutex<()>>>,
    spectrum: HashMap<String, Arc<MIPMap<RGBSpectrum>>>,
    spectrum_build_locks: HashMap<String, Arc<Mutex<()>>>,
}

impl ImageMapMIPMapCache {
    fn float_build_lock(&mut self, key: &str) -> Arc<Mutex<()>> {
        Arc::clone(
            self.float_build_locks
                .entry(key.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }

    fn spectrum_build_lock(&mut self, key: &str) -> Arc<Mutex<()>> {
        Arc::clone(
            self.spectrum_build_locks
                .entry(key.to_string())
                .or_insert_with(|| Arc::new(Mutex::new(()))),
        )
    }
}

fn mipmap_texinfo(texinfo: &TexInfo) -> TexInfo {
    let mut texinfo = texinfo.clone();
    texinfo.scale = 1.0;
    texinfo
}

fn mipmap_key(texinfo: &TexInfo) -> String {
    mipmap_texinfo(texinfo).to_string()
}

pub struct ImageTexture<Tmemory, Treturn> {
    mapping: TextureMapping2D,
    mipmap: Arc<MIPMap<Tmemory>>,
    spectrum_type: SpectrumType,
    scale: Float,
    invert: bool,
    phantom: PhantomData<Treturn>, //non allocate
}

impl<Tmemory, Treturn> ImageTexture<Tmemory, Treturn> {
    pub fn mipmap_channel_count(&self) -> usize {
        self.mipmap.channel_count()
    }
}

impl ImageTexture<Float, Float> {
    pub fn new(
        mapping: TextureMapping2D,
        mipmap: MIPMap<Float>,
        scale: Float,
        invert: bool,
    ) -> Self {
        Self::new_shared(mapping, Arc::new(mipmap), scale, invert)
    }

    pub fn new_shared(
        mapping: TextureMapping2D,
        mipmap: Arc<MIPMap<Float>>,
        scale: Float,
        invert: bool,
    ) -> Self {
        Self {
            mapping,
            mipmap,
            spectrum_type: SpectrumType::Albedo,
            scale,
            invert,
            phantom: PhantomData,
        }
    }

    pub fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let (mut st, dstdx, dstdy) = self.mapping.map(ctx);
        st[1] = 1.0 - st[1];
        let v = self.scale * self.mipmap.lookup_delta(&st, &dstdx, &dstdy);
        // pbrt-v4 textures.h FloatImageTexture::Evaluate: `invert ? max(0, 1-v) : v`.
        if self.invert {
            (1.0 - v).max(0.0)
        } else {
            v
        }
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
    ) -> Result<FloatTexture, PbrtError> {
        Self::create_with_mipmap_cache(render_from_texture, parameters, None)
    }

    pub fn create_with_mipmap_cache(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        mipmap_cache: Option<&Mutex<ImageMapMIPMapCache>>,
    ) -> Result<FloatTexture, PbrtError> {
        let mapping =
            TextureMapping2D::create(render_from_texture, parameters.parameter_dictionary())?;
        let texinfo = create_texinfo(parameters)?
            .ok_or_else(|| PbrtError::error("Image texture requires a non-empty filename."))?;
        let scale = texinfo.scale;
        let encoding = texinfo.encoding;
        let invert = parameters.get_one_bool("invert", false);
        let mipmap_texinfo = mipmap_texinfo(&texinfo);
        let mipmap_key = mipmap_key(&texinfo);

        let build_lock = if let Some(cache) = mipmap_cache {
            let mut cache = cache.lock().unwrap();
            if let Some(mipmap) = cache.float.get(&mipmap_key).cloned() {
                return Ok(FloatTexture::ImageMap(FloatImageTexture::new_shared(
                    mapping, mipmap, scale, invert,
                )));
            }
            Some(cache.float_build_lock(&mipmap_key))
        } else {
            None
        };
        let _build_guard = build_lock.as_ref().map(|lock| lock.lock().unwrap());
        if let Some(cache) = mipmap_cache {
            if let Some(mipmap) = cache.lock().unwrap().float.get(&mipmap_key).cloned() {
                return Ok(FloatTexture::ImageMap(FloatImageTexture::new_shared(
                    mapping, mipmap, scale, invert,
                )));
            }
        }

        let cache_entry = if texture_cache_enabled() {
            let cache_path =
                get_typed_mipmap_cache_dir(&mipmap_texinfo, MipmapCacheValueType::Float);
            let file_hash = get_mipmap_file_hash(&mipmap_texinfo.filename)?;
            let cache_exists = Path::new(&cache_path).exists();
            match load_float_mipmap_cache(&cache_path, &file_hash, &mipmap_texinfo) {
                Ok(mipmap) => {
                    let mipmap = Arc::new(mipmap);
                    if let Some(cache) = mipmap_cache {
                        cache
                            .lock()
                            .unwrap()
                            .float
                            .insert(mipmap_key, Arc::clone(&mipmap));
                    }
                    return Ok(FloatTexture::ImageMap(FloatImageTexture::new_shared(
                        mapping, mipmap, scale, invert,
                    )));
                }
                Err(err) if cache_exists => match remove_mipmap_cache(&cache_path) {
                    Ok(_) => log::warn!(
                        "Invalid float MIPMap cache removed ({}): {}",
                        cache_path, err
                    ),
                    Err(remove_err) => log::warn!(
                        "Invalid float MIPMap cache could not be removed ({}): {}; removal failed: {}",
                        cache_path, err, remove_err
                    ),
                },
                Err(_) => {}
            }
            Some((cache_path, file_hash))
        } else {
            None
        };

        let raw = read_raw_image_with_encoding(&texinfo.filename, encoding)?;
        let mipmap = if matches!(raw.channels, 1 | 3) {
            MIPMap::<Float>::new_from_raw_image(
                &raw,
                texinfo.filter,
                texinfo.max_aniso,
                texinfo.swrap_mode,
                texinfo.twrap_mode,
            )
        } else {
            let (data, channels) = normalize_raw_image_for_float(&raw)?;
            MIPMap::<Float>::new_with_raw_channels_and_storage(
                &raw.resolution,
                &data,
                channels,
                texinfo.filter,
                texinfo.max_aniso,
                texinfo.swrap_mode,
                texinfo.twrap_mode,
                MIPMapStorageKind::from_raw_image(&raw),
            )
        };
        if let Some((cache_path, file_hash)) = cache_entry {
            if let Err(err) =
                save_float_mipmap_cache(&cache_path, &mipmap_texinfo, &file_hash, &mipmap)
            {
                log::warn!(
                    "Could not write float MIPMap cache ({}): {}",
                    cache_path,
                    err
                );
            }
        }
        let mipmap = Arc::new(mipmap);
        if let Some(cache) = mipmap_cache {
            cache
                .lock()
                .unwrap()
                .float
                .insert(mipmap_key, Arc::clone(&mipmap));
        }
        Ok(FloatTexture::ImageMap(FloatImageTexture::new_shared(
            mapping, mipmap, scale, invert,
        )))
    }
}

impl ImageTexture<RGBSpectrum, Spectrum> {
    pub fn new(
        mapping: TextureMapping2D,
        mipmap: MIPMap<RGBSpectrum>,
        spectrum_type: SpectrumType,
        scale: Float,
        invert: bool,
    ) -> Self {
        Self::new_shared(mapping, Arc::new(mipmap), spectrum_type, scale, invert)
    }

    pub fn new_shared(
        mapping: TextureMapping2D,
        mipmap: Arc<MIPMap<RGBSpectrum>>,
        spectrum_type: SpectrumType,
        scale: Float,
        invert: bool,
    ) -> Self {
        Self {
            mapping,
            mipmap,
            spectrum_type,
            scale,
            invert,
            phantom: PhantomData,
        }
    }

    pub fn create_variants(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_types: &[SpectrumType],
    ) -> Result<Vec<(SpectrumType, Self)>, PbrtError> {
        Self::create_variants_with_mipmap_cache(
            render_from_texture,
            parameters,
            spectrum_types,
            None,
        )
    }

    pub fn create_variants_with_mipmap_cache(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_types: &[SpectrumType],
        mipmap_cache: Option<&Mutex<ImageMapMIPMapCache>>,
    ) -> Result<Vec<(SpectrumType, Self)>, PbrtError> {
        let mapping =
            TextureMapping2D::create(render_from_texture, parameters.parameter_dictionary())?;
        let (mipmap, scale, invert) = Self::create_mipmap(parameters, mipmap_cache)?;
        let textures = spectrum_types
            .iter()
            .map(|&spectrum_type| {
                (
                    spectrum_type,
                    SpectrumImageTexture::new_shared(
                        mapping.clone(),
                        Arc::clone(&mipmap),
                        spectrum_type,
                        scale,
                        invert,
                    ),
                )
            })
            .collect();
        Ok(textures)
    }

    /// pbrt-v4 verbatim `ImageTextureBase::Evaluate` — returns a
    /// `SampledSpectrum` directly. Applies `invert` as `ClampZero(1-rgb)`.
    pub fn evaluate(
        &self,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        let (mut st, dstdx, dstdy) = self.mapping.map(ctx);
        st[1] = 1.0 - st[1];
        let mut rgb = self.mipmap.lookup_delta(&st, &dstdx, &dstdy).to_rgb();
        for c in rgb.iter_mut() {
            *c *= self.scale;
        }
        if self.invert {
            for c in rgb.iter_mut() {
                *c = (1.0 - *c).max(0.0);
            }
        } else {
            for c in rgb.iter_mut() {
                *c = c.max(0.0);
            }
        }
        Spectrum::rgb_to_sampled(rgb, self.spectrum_type, lambda)
    }

    pub fn convert_out(from: &RGBSpectrum, spectrum_type: SpectrumType) -> Spectrum {
        return Spectrum::from_rgb(&from.to_rgb(), spectrum_type);
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_type: SpectrumType,
    ) -> Result<Self, PbrtError> {
        Self::create_with_mipmap_cache(render_from_texture, parameters, spectrum_type, None)
    }

    pub fn create_with_mipmap_cache(
        render_from_texture: &Transform,
        parameters: &TextureParameterDictionary,
        spectrum_type: SpectrumType,
        mipmap_cache: Option<&Mutex<ImageMapMIPMapCache>>,
    ) -> Result<Self, PbrtError> {
        let mapping =
            TextureMapping2D::create(render_from_texture, parameters.parameter_dictionary())?;
        let (mipmap, scale, invert) = Self::create_mipmap(parameters, mipmap_cache)?;
        Ok(SpectrumImageTexture::new_shared(
            mapping,
            mipmap,
            spectrum_type,
            scale,
            invert,
        ))
    }

    fn create_mipmap(
        parameters: &TextureParameterDictionary,
        mipmap_cache: Option<&Mutex<ImageMapMIPMapCache>>,
    ) -> Result<(Arc<MIPMap<RGBSpectrum>>, Float, bool), PbrtError> {
        let texinfo = create_texinfo(parameters)?
            .ok_or_else(|| PbrtError::error("Image texture requires a non-empty filename."))?;
        let scale = texinfo.scale;
        let encoding = texinfo.encoding;
        let invert = parameters.get_one_bool("invert", false);
        let mipmap_texinfo = mipmap_texinfo(&texinfo);
        let mipmap_key = mipmap_key(&texinfo);

        let build_lock = if let Some(cache) = mipmap_cache {
            let mut cache = cache.lock().unwrap();
            if let Some(mipmap) = cache.spectrum.get(&mipmap_key).cloned() {
                return Ok((mipmap, scale, invert));
            }
            Some(cache.spectrum_build_lock(&mipmap_key))
        } else {
            None
        };
        let _build_guard = build_lock.as_ref().map(|lock| lock.lock().unwrap());
        if let Some(cache) = mipmap_cache {
            if let Some(mipmap) = cache.lock().unwrap().spectrum.get(&mipmap_key).cloned() {
                return Ok((mipmap, scale, invert));
            }
        }

        let cache_entry = if texture_cache_enabled() {
            let cache_path =
                get_typed_mipmap_cache_dir(&mipmap_texinfo, MipmapCacheValueType::Spectrum);
            let file_hash = get_mipmap_file_hash(&mipmap_texinfo.filename)?;
            let cache_exists = Path::new(&cache_path).exists();
            match load_spectrum_mipmap_cache(&cache_path, &file_hash, &mipmap_texinfo) {
                Ok(mipmap) => {
                    let mipmap = Arc::new(mipmap);
                    if let Some(cache) = mipmap_cache {
                        cache
                            .lock()
                            .unwrap()
                            .spectrum
                            .insert(mipmap_key, Arc::clone(&mipmap));
                    }
                    return Ok((mipmap, scale, invert));
                }
                Err(err) if cache_exists => match remove_mipmap_cache(&cache_path) {
                    Ok(_) => log::warn!(
                        "Invalid spectrum MIPMap cache removed ({}): {}",
                        cache_path, err
                    ),
                    Err(remove_err) => log::warn!(
                        "Invalid spectrum MIPMap cache could not be removed ({}): {}; removal failed: {}",
                        cache_path, err, remove_err
                    ),
                },
                Err(_) => {}
            }
            Some((cache_path, file_hash))
        } else {
            None
        };

        let raw = read_raw_image_with_encoding(&texinfo.filename, encoding)?;
        let (data, channels, resolution) = normalize_raw_image_for_spectrum(&raw)?;
        let mipmap = MIPMap::<RGBSpectrum>::new_with_raw_channels_and_storage(
            &resolution,
            &data,
            channels,
            texinfo.filter,
            texinfo.max_aniso,
            texinfo.swrap_mode,
            texinfo.twrap_mode,
            MIPMapStorageKind::from_raw_image(&raw),
        );
        if let Some((cache_path, file_hash)) = cache_entry {
            if let Err(err) =
                save_spectrum_mipmap_cache(&cache_path, &mipmap_texinfo, &file_hash, &mipmap)
            {
                log::warn!(
                    "Could not write spectrum MIPMap cache ({}): {}",
                    cache_path,
                    err
                );
            }
        }
        let mipmap = Arc::new(mipmap);
        if let Some(cache) = mipmap_cache {
            cache
                .lock()
                .unwrap()
                .spectrum
                .insert(mipmap_key, Arc::clone(&mipmap));
        }
        Ok((mipmap, scale, invert))
    }
}

fn has_extension(path: &str, ext: &str) -> bool {
    if let Some(e) = Path::new(path).extension() {
        if let Some(s) = e.to_str() {
            if s == ext {
                return true;
            }
        }
    }
    return false;
}

fn get_wrap_mode(mode: &str) -> Result<ImageWrap, PbrtError> {
    match mode {
        "repeat" => Ok(ImageWrap::Repeat),
        "black" => Ok(ImageWrap::Black),
        "clamp" => Ok(ImageWrap::Clamp),
        "octahedralsphere" => Ok(ImageWrap::OctahedralSphere),
        _ => Err(PbrtError::error(&format!(
            "Unknown image texture wrap mode \"{}\".",
            mode
        ))),
    }
}

fn get_filter_mode(mode: &str) -> Result<ImageFilter, PbrtError> {
    match mode {
        "point" => Ok(ImageFilter::Point),
        "bilinear" => Ok(ImageFilter::Bilinear),
        "trilinear" => Ok(ImageFilter::Trilinear),
        "ewa" | "EWA" => Ok(ImageFilter::EWA),
        _ => Err(PbrtError::error(&format!(
            "Unknown image texture filter \"{}\".",
            mode
        ))),
    }
}

pub fn create_texinfo(
    parameters: &TextureParameterDictionary,
) -> Result<Option<TexInfo>, PbrtError> {
    // Initialize _ImageTexture_ parameters
    let filename = parameters.get_one_filename("filename", "");
    if filename.is_empty() {
        return Ok(None);
    }

    let max_aniso = parameters.get_one_float("maxanisotropy", 8.0);
    let filter = {
        let filter_name = parameters.get_one_string("filter", "");
        if !filter_name.is_empty() {
            get_filter_mode(&filter_name)?
        } else if parameters.get_one_bool("trilinear", false) {
            ImageFilter::Trilinear
        } else {
            ImageFilter::Bilinear
        }
    };
    let wrap = parameters.get_one_string("wrap", "repeat");
    let swrap = parameters.get_one_string("swrap", &wrap);
    let twrap = parameters.get_one_string("twrap", &wrap);

    let swrap_mode = get_wrap_mode(&swrap)?;
    let twrap_mode = get_wrap_mode(&twrap)?;
    if (swrap_mode == ImageWrap::OctahedralSphere) != (twrap_mode == ImageWrap::OctahedralSphere) {
        return Err(PbrtError::error(
            "Image texture octahedralsphere wrap requires both axes.",
        ));
    }

    let scale = parameters.get_one_float("scale", 1.0);
    let encoding = {
        let default_encoding = if has_extension(&filename, "png") {
            "sRGB"
        } else {
            "linear"
        };
        let encoding_name = parameters.get_one_string("encoding", "");
        if !encoding_name.is_empty() {
            ColorEncoding::parse(&encoding_name)?
        } else if parameters.parameter_dictionary().has_parameter("gamma") {
            if !parameters
                .parameter_dictionary()
                .get_floats("gamma")
                .is_empty()
            {
                ColorEncoding::parse(&format!(
                    "gamma {}",
                    parameters
                        .parameter_dictionary()
                        .get_one_float("gamma", 0.0)
                ))?
            } else {
                ColorEncoding::from_legacy_gamma(
                    parameters
                        .parameter_dictionary()
                        .get_one_bool("gamma", false),
                )
            }
        } else {
            ColorEncoding::parse(default_encoding)?
        }
    };

    Ok(Some(TexInfo {
        cache_version: 3,
        filename,
        filter,
        max_aniso,
        swrap_mode,
        twrap_mode,
        scale,
        encoding,
    }))
}

fn alpha_all_one(raw: &RawImage) -> bool {
    let alpha_offset = raw.channels - 1;
    let pixels = (raw.resolution.x * raw.resolution.y) as usize;
    (0..pixels).all(|i| raw.channel(i, alpha_offset) == 1.0)
}

fn normalize_raw_image_for_float(raw: &RawImage) -> Result<(Vec<Float>, usize), PbrtError> {
    let pixels = (raw.resolution.x * raw.resolution.y) as usize;
    let normalized = match raw.channels {
        1 => (raw.data_f32(), 1),
        2 => {
            // Match pbrt-v4 Image::Read: treat Y+A images as a Y-only image.
            let mut y = Vec::with_capacity(pixels);
            for pixel in 0..pixels {
                y.push(raw.channel(pixel, 0));
            }
            (y, 1)
        }
        3 => (raw.data_f32(), 3),
        4 => {
            if alpha_all_one(raw) {
                let mut rgb = Vec::with_capacity(3 * pixels);
                for i in 0..pixels {
                    rgb.extend_from_slice(&[
                        raw.channel(i, 0),
                        raw.channel(i, 1),
                        raw.channel(i, 2),
                    ]);
                }
                (rgb, 3)
            } else {
                (raw.data_f32(), 4)
            }
        }
        _ => {
            return Err(PbrtError::error(&format!(
                "Unsupported image channel count for float imagemap: {}",
                raw.channels
            )));
        }
    };
    Ok(normalized)
}

fn normalize_raw_image_for_spectrum(
    raw: &RawImage,
) -> Result<(Vec<Float>, usize, Point2i), PbrtError> {
    let pixels = (raw.resolution.x * raw.resolution.y) as usize;
    match raw.channels {
        1 => Ok((raw.data_f32(), 1, raw.resolution)),
        2 => {
            let mut data = Vec::with_capacity(3 * pixels);
            for pixel in 0..pixels {
                let value = raw.channel(pixel, 0);
                data.extend_from_slice(&[value, value, value]);
            }
            Ok((data, 3, raw.resolution))
        }
        3 => Ok((raw.data_f32(), 3, raw.resolution)),
        4 => {
            if alpha_all_one(raw) {
                let mut data = Vec::with_capacity(3 * pixels);
                for pixel in 0..pixels {
                    data.extend_from_slice(&[
                        raw.channel(pixel, 0),
                        raw.channel(pixel, 1),
                        raw.channel(pixel, 2),
                    ]);
                }
                Ok((data, 3, raw.resolution))
            } else {
                Ok((raw.data_f32(), 4, raw.resolution))
            }
        }
        _ => Err(PbrtError::error(&format!(
            "Unsupported image channel count for Spectrum imagemap: {}",
            raw.channels
        ))),
    }
}

pub type FloatImageTexture = ImageTexture<Float, Float>;
pub type SpectrumImageTexture = ImageTexture<RGBSpectrum, Spectrum>;
