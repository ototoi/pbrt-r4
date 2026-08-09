use super::parameter_dictionary::ParameterDictionary;
use crate::base::typed_spectrum_texture_name;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::PbrtError;
use crate::util::spectrum::*;

use std::collections::HashMap;
use std::sync::Arc;

use log::*;

type FloatTextureMap = HashMap<String, Arc<FloatTexture>>;
type SpectrumTextureMap = HashMap<String, Arc<SpectrumTexture>>;

pub struct TextureParameterDictionary<'a> {
    pub params: &'a ParameterDictionary,
    pub f_tex: &'a FloatTextureMap,
    pub s_tex: &'a SpectrumTextureMap,
}

impl<'a> TextureParameterDictionary<'a> {
    pub fn new(
        params: &'a ParameterDictionary,
        f_tex: &'a FloatTextureMap,
        s_tex: &'a SpectrumTextureMap,
    ) -> Self {
        TextureParameterDictionary::<'a> {
            params,
            f_tex,
            s_tex,
        }
    }

    /// Returns the underlying parameter dictionary used by v4-style factory
    /// functions that do not need named texture lookup.
    pub fn parameter_dictionary(&self) -> &'a ParameterDictionary {
        self.params
    }

    fn get_float(&self, key: &str) -> Option<Float> {
        if let Some(c) = self.params.get_floats_ref(key) {
            if c.len() >= 1 {
                if c.len() > 1 {
                    warn!("More than one texture present in parameter file for key \"{}\". Using first.", key);
                }
                return Some(c[0]);
            }
        }
        None
    }

    fn get_spectrum_source(&self, key: &str, spectrum_type: SpectrumType) -> Option<Spectrum> {
        if let Some(stored_key) = self.params.get_keys().iter().find(|stored_key| {
            self.params.get_key_name(stored_key) == self.params.get_key_name(key)
        }) {
            return Some(self.params.get_one_spectrum_typed(
                stored_key,
                &Spectrum::zero(),
                spectrum_type,
            ));
        }
        None
    }

    pub fn get_spectrum_or_null(&self, key: &str) -> Option<Spectrum> {
        self.get_spectrum_or_null_typed(key, SpectrumType::Albedo)
    }

    pub fn get_spectrum_or_null_typed(
        &self,
        key: &str,
        spectrum_type: SpectrumType,
    ) -> Option<Spectrum> {
        self.get_spectrum_source(key, spectrum_type)
    }

    pub fn get_float_array(&self, key: &str) -> Vec<Float> {
        if let Some(values) = self.params.get_floats_ref(key) {
            return values.clone();
        }
        Vec::new()
    }

    fn get_texture_from(params: &ParameterDictionary, key: &str) -> Option<String> {
        let keys = params.get_keys();
        let Some(stored_key) = keys.iter().find(|stored_key| {
            params.get_key_type(stored_key) == "texture" && params.get_key_name(stored_key) == key
        }) else {
            return None;
        };
        if let Some(s) = params.get_textures_ref(stored_key) {
            if s.len() >= 1 {
                if s.len() > 1 {
                    warn!("More than one texture present in parameter file for key \"{}\". Using first.", key);
                }
                return Some(s[0].clone());
            }
        }
        return None;
    }

    fn get_texture(&self, key: &str) -> Option<String> {
        Self::get_texture_from(self.params, key)
    }

    pub fn has_texture_name(&self, key: &str) -> bool {
        Self::get_texture_from(self.params, key).is_some()
    }

    fn unresolved_texture_error(key: &str, texture_kind: &str) -> PbrtError {
        PbrtError::error(&format!(
            "Couldn't find {} texture named \"{}\" for parameter \"{}\"",
            texture_kind, key, key
        ))
    }

    pub fn get_float_texture_or_null(
        &self,
        key: &str,
    ) -> Result<Option<Arc<FloatTexture>>, PbrtError> {
        if let Some(name) = self.get_texture(key) {
            if let Some(tex) = self.f_tex.get(&name) {
                return Ok(Some(Arc::clone(tex)));
            }
            return Err(Self::unresolved_texture_error(key, "float"));
        } else if let Some(s) = self.get_float(key) {
            return Ok(Some(Arc::new(FloatTexture::Constant(
                ConstantTexture::new(&s),
            ))));
        }
        Ok(None)
    }

    pub fn get_float_texture(
        &self,
        key: &str,
        value: Float,
    ) -> Result<Arc<FloatTexture>, PbrtError> {
        match self.get_float_texture_or_null(key)? {
            Some(tex) => Ok(tex),
            None => Ok(Arc::new(FloatTexture::Constant(ConstantTexture::new(
                &value,
            )))),
        }
    }

    pub fn get_spectrum_texture_or_null(
        &self,
        key: &str,
    ) -> Result<Option<Arc<SpectrumTexture>>, PbrtError> {
        self.get_spectrum_texture_or_null_typed(key, SpectrumType::Albedo)
    }

    pub fn get_spectrum_texture_or_null_typed(
        &self,
        key: &str,
        spectrum_type: SpectrumType,
    ) -> Result<Option<Arc<SpectrumTexture>>, PbrtError> {
        if let Some(name) = self.get_texture(key) {
            let typed_name = typed_spectrum_texture_name(&name, spectrum_type);
            if let Some(tex) = self.s_tex.get(&typed_name) {
                return Ok(Some(Arc::clone(tex)));
            }
            if let Some(tex) = self.s_tex.get(&name) {
                return Ok(Some(Arc::clone(tex)));
            }
            if let Some(source) = self.get_spectrum_source(&name, spectrum_type) {
                return Ok(Some(Arc::new(spectrum_texture_from_constant(source))));
            }
            return Err(Self::unresolved_texture_error(key, "spectrum"));
        }
        if let Some(source) = self.get_spectrum_source(key, spectrum_type) {
            return Ok(Some(Arc::new(spectrum_texture_from_constant(source))));
        }
        Ok(None)
    }

    pub fn get_spectrum_texture(
        &self,
        key: &str,
        value: &Spectrum,
    ) -> Result<Arc<SpectrumTexture>, PbrtError> {
        self.get_spectrum_texture_typed(key, value, SpectrumType::Albedo)
    }

    pub fn get_spectrum_texture_typed(
        &self,
        key: &str,
        value: &Spectrum,
        spectrum_type: SpectrumType,
    ) -> Result<Arc<SpectrumTexture>, PbrtError> {
        match self.get_spectrum_texture_or_null_typed(key, spectrum_type)? {
            Some(tex) => Ok(tex),
            None => Ok(Arc::new(SpectrumTexture::Constant(ConstantTexture::new(
                value,
            )))),
        }
    }

    pub fn get_one_float(&self, key: &str, value: Float) -> Float {
        self.params.get_one_float(key, value)
    }

    pub fn get_one_int(&self, key: &str, value: i32) -> i32 {
        self.params.get_one_int(key, value)
    }

    pub fn get_one_bool(&self, key: &str, value: bool) -> bool {
        self.params.get_one_bool(key, value)
    }

    pub fn get_one_vector3f(&self, key: &str, value: &Vector3f) -> Vector3f {
        self.params.get_one_vector3f(key, value)
    }

    pub fn get_one_spectrum(&self, key: &str, value: &Spectrum) -> Spectrum {
        self.get_one_spectrum_typed(key, value, SpectrumType::Albedo)
    }

    pub fn get_one_spectrum_typed(
        &self,
        key: &str,
        value: &Spectrum,
        spectrum_type: SpectrumType,
    ) -> Spectrum {
        self.params
            .get_one_spectrum_typed(key, value, spectrum_type)
    }

    pub fn get_one_string(&self, key: &str, value: &str) -> String {
        self.params.get_one_string(key, value)
    }

    pub fn get_one_filename(&self, key: &str, value: &str) -> String {
        return self.get_one_string(key, value);
    }
}
