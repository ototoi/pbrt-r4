use super::wellknown_params;
use crate::util::base::*;
use crate::util::spectrum::rgb_to_spectrum::{RGBColorSpace, SRGB};
use crate::util::spectrum::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Clone, Debug, PartialEq)]
pub struct SampledSpectrumParam {
    pub lambda: Vec<Float>,
    pub values: Vec<Float>,
}

pub type ParameterReadGuard<'a, T> = RwLockReadGuard<'a, Vec<T>>;
pub type ParameterWriteGuard<'a, T> = RwLockWriteGuard<'a, Vec<T>>;

type SharedValues<T> = RwLock<Vec<T>>;

fn read_values<T>(values: &SharedValues<T>) -> ParameterReadGuard<'_, T> {
    values
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_values<T>(values: &SharedValues<T>) -> ParameterWriteGuard<'_, T> {
    values
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn clone_values_map<T: Clone>(
    values: &HashMap<String, SharedValues<T>>,
) -> HashMap<String, SharedValues<T>> {
    values
        .iter()
        .map(|(key, values)| (key.clone(), RwLock::new(read_values(values).clone())))
        .collect()
}

pub struct ParameterDictionary {
    bools: HashMap<String, SharedValues<bool>>,
    ints: HashMap<String, SharedValues<i32>>,
    floats: HashMap<String, SharedValues<Float>>,
    strings: HashMap<String, SharedValues<String>>,
    spectrums: HashMap<String, SharedValues<Spectrum>>,
    sampled_spectra: HashMap<String, SharedValues<SampledSpectrumParam>>,
    points: HashMap<String, SharedValues<Float>>,
    keys: Vec<String>,
    color_space: &'static RGBColorSpace,
}

fn get_key_type(key: &str) -> String {
    let ss: Vec<&str> = key.split_ascii_whitespace().collect();
    match ss.len() {
        2 => {
            return String::from(ss[0]);
        }
        1 => {
            let name = ss[0];
            if let Some(t) = wellknown_params::find_type_from_key(name) {
                return String::from(t);
            }
        }
        _ => {}
    }
    return String::from("");
}

fn get_key_name(key: &str) -> String {
    let ss: Vec<&str> = key.split_ascii_whitespace().collect();
    match ss.len() {
        2 => String::from(ss[1]),
        _ => String::from(key),
    }
}

fn add_value<T: Clone>(
    k: &mut Vec<String>,
    m: &mut HashMap<String, SharedValues<T>>,
    key: &str,
    v: T,
) {
    k.push(key.to_string());
    let keyname = get_key_name(key);
    match m.get(&keyname) {
        Some(r) => {
            let mut items = write_values(r);
            items.push(v);
        }
        _ => {
            let r = RwLock::<Vec<T>>::new(Vec::<T>::new());
            {
                let mut items = write_values(&r);
                items.push(v);
            }
            m.insert(keyname, r);
        }
    }
}

fn add_values<T: Clone>(
    k: &mut Vec<String>,
    m: &mut HashMap<String, SharedValues<T>>,
    key: &str,
    v: &[T],
) {
    k.push(key.to_string());
    let keyname = get_key_name(key);
    if let Some(r) = m.get(&keyname) {
        let mut items = write_values(r);
        items.clear(); //
        for x in v.iter() {
            items.push(x.clone());
        }
    } else {
        let r = RwLock::<Vec<T>>::new(Vec::<T>::new());
        {
            let mut items = write_values(&r);
            for x in v.iter() {
                items.push(x.clone());
            }
        }
        m.insert(keyname, r);
    }
}

fn add_owned_values<T>(
    k: &mut Vec<String>,
    m: &mut HashMap<String, SharedValues<T>>,
    key: &str,
    values: Vec<T>,
) {
    k.push(key.to_string());
    let keyname = get_key_name(key);
    if let Some(storage) = m.get(&keyname) {
        *write_values(storage) = values;
    } else {
        m.insert(keyname, RwLock::new(values));
    }
}

fn add_owned_values_with_type<T>(
    k: &mut Vec<String>,
    m: &mut HashMap<String, SharedValues<T>>,
    parameter_type: &str,
    name: &str,
    values: Vec<T>,
) {
    let key = if parameter_type.is_empty() {
        name.to_string()
    } else {
        format!("{parameter_type} {name}")
    };
    k.push(key);
    if let Some(storage) = m.get(name) {
        *write_values(storage) = values;
    } else {
        m.insert(name.to_string(), RwLock::new(values));
    }
}

fn add_value_no_key<T: Clone>(m: &mut HashMap<String, SharedValues<T>>, key: &str, v: T) {
    let keyname = get_key_name(key);
    match m.get(&keyname) {
        Some(r) => {
            let mut items = write_values(r);
            items.push(v);
        }
        _ => {
            let r = RwLock::<Vec<T>>::new(Vec::<T>::new());
            {
                let mut items = write_values(&r);
                items.push(v);
            }
            m.insert(keyname, r);
        }
    }
}

fn get_values<T: Clone>(m: &HashMap<String, SharedValues<T>>, key: &str) -> Vec<T> {
    let keyname = get_key_name(key);
    match m.get(&keyname) {
        Some(r) => read_values(r).clone(),
        _ => Vec::<T>::new(),
    }
}

fn get_values_ref<'a, T>(
    m: &'a HashMap<String, SharedValues<T>>,
    key: &str,
) -> Option<ParameterReadGuard<'a, T>> {
    let keyname = get_key_name(key);
    let r = m.get(&keyname);
    match r {
        Some(r) => {
            return Some(read_values(r));
        }
        _ => {
            return None;
        }
    }
}

fn get_values_mut<'a, T>(
    m: &'a HashMap<String, SharedValues<T>>,
    key: &str,
) -> Option<ParameterWriteGuard<'a, T>> {
    let keyname = get_key_name(key);
    let r = m.get(&keyname);
    match r {
        Some(r) => {
            return Some(write_values(r));
        }
        _ => {
            return None;
        }
    }
}

impl Default for ParameterDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl ParameterDictionary {
    pub fn new() -> Self {
        ParameterDictionary {
            bools: HashMap::new(),
            ints: HashMap::new(),
            floats: HashMap::new(),
            strings: HashMap::new(),
            spectrums: HashMap::new(),
            sampled_spectra: HashMap::new(),
            points: HashMap::new(),
            keys: Vec::<String>::new(),
            color_space: &SRGB,
        }
    }

    pub fn color_space(&self) -> &'static RGBColorSpace {
        self.color_space
    }

    pub fn set_color_space(&mut self, color_space: &'static RGBColorSpace) {
        self.color_space = color_space;
    }

    //--------------------

    pub fn add_bool(&mut self, key: &str, v: bool) {
        add_value(&mut self.keys, &mut self.bools, key, v);
    }

    pub fn add_bools(&mut self, key: &str, v: &[bool]) {
        add_values(&mut self.keys, &mut self.bools, key, v);
    }

    pub fn add_int(&mut self, key: &str, v: i32) {
        add_value(&mut self.keys, &mut self.ints, key, v);
    }

    pub fn add_ints(&mut self, key: &str, v: &[i32]) {
        add_values(&mut self.keys, &mut self.ints, key, v);
    }

    pub fn add_owned_bools(&mut self, key: &str, values: Vec<bool>) {
        add_owned_values(&mut self.keys, &mut self.bools, key, values);
    }

    pub fn add_owned_bools_typed(&mut self, parameter_type: &str, name: &str, values: Vec<bool>) {
        add_owned_values_with_type(
            &mut self.keys,
            &mut self.bools,
            parameter_type,
            name,
            values,
        );
    }

    pub fn add_owned_ints(&mut self, key: &str, values: Vec<i32>) {
        add_owned_values(&mut self.keys, &mut self.ints, key, values);
    }

    pub fn add_owned_ints_typed(&mut self, parameter_type: &str, name: &str, values: Vec<i32>) {
        add_owned_values_with_type(&mut self.keys, &mut self.ints, parameter_type, name, values);
    }

    pub fn add_float(&mut self, key: &str, v: Float) {
        add_value(&mut self.keys, &mut self.floats, key, v);
    }

    pub fn add_floats(&mut self, key: &str, v: &[Float]) {
        let t = get_key_type(key);
        match &t as &str {
            "point" | "point2" | "point3" | "point4" => {
                add_values(&mut self.keys, &mut self.points, key, v)
            }
            "normal" => add_values(&mut self.keys, &mut self.points, key, v),
            "vector" | "vector2" | "vector3" | "vector4" => {
                add_values(&mut self.keys, &mut self.points, key, v)
            }
            "color" => self.add_color(key, v),
            "rgb" => self.add_color(key, v),
            "blackbody" => add_values(&mut self.keys, &mut self.floats, key, v),
            _ => add_values(&mut self.keys, &mut self.floats, key, v),
        }
    }

    pub fn add_owned_floats(&mut self, key: &str, values: Vec<Float>) {
        let t = get_key_type(key);
        match t.as_str() {
            "point" | "point2" | "point3" | "point4" | "normal" | "vector" | "vector2"
            | "vector3" | "vector4" | "color" | "rgb" => {
                add_owned_values(&mut self.keys, &mut self.points, key, values)
            }
            "xyz" => self.add_xyz(key, &values),
            "blackbody" => add_owned_values(&mut self.keys, &mut self.floats, key, values),
            _ => add_owned_values(&mut self.keys, &mut self.floats, key, values),
        }
    }

    pub fn add_owned_floats_typed(&mut self, parameter_type: &str, name: &str, values: Vec<Float>) {
        match parameter_type {
            "point" | "point2" | "point3" | "point4" | "normal" | "vector" | "vector2"
            | "vector3" | "vector4" | "color" | "rgb" => add_owned_values_with_type(
                &mut self.keys,
                &mut self.points,
                parameter_type,
                name,
                values,
            ),
            "xyz" => {
                let xyz = RGBSpectrum::rgb_from_xyz(&values);
                let rgb = xyz.to_rgb();
                add_owned_values_with_type(
                    &mut self.keys,
                    &mut self.points,
                    parameter_type,
                    name,
                    rgb.to_vec(),
                );
            }
            "blackbody" => add_owned_values_with_type(
                &mut self.keys,
                &mut self.floats,
                parameter_type,
                name,
                values,
            ),
            _ => add_owned_values_with_type(
                &mut self.keys,
                &mut self.floats,
                parameter_type,
                name,
                values,
            ),
        }
    }

    pub fn add_string(&mut self, key: &str, v: &str) {
        add_value(&mut self.keys, &mut self.strings, key, String::from(v));
    }

    pub fn add_strings(&mut self, key: &str, v: &[&str]) {
        let vv: Vec<String> = v.iter().map(|s| String::from(*s)).collect();
        add_values(&mut self.keys, &mut self.strings, key, &vv);
    }

    pub fn add_owned_strings(&mut self, key: &str, values: Vec<String>) {
        add_owned_values(&mut self.keys, &mut self.strings, key, values);
    }

    pub fn add_owned_strings_typed(
        &mut self,
        parameter_type: &str,
        name: &str,
        values: Vec<String>,
    ) {
        add_owned_values_with_type(
            &mut self.keys,
            &mut self.strings,
            parameter_type,
            name,
            values,
        );
    }

    pub fn add_spectrum(&mut self, key: &str, v: &Spectrum) {
        add_value(&mut self.keys, &mut self.spectrums, key, v.clone());
    }

    pub fn add_owned_spectrums_typed(
        &mut self,
        parameter_type: &str,
        name: &str,
        values: Vec<Spectrum>,
    ) {
        add_owned_values_with_type(
            &mut self.keys,
            &mut self.spectrums,
            parameter_type,
            name,
            values,
        );
    }

    pub fn add_spectrums(&mut self, key: &str, v: &[Spectrum]) {
        add_values(&mut self.keys, &mut self.spectrums, key, v);
    }

    pub fn add_sampled_spectrum(&mut self, key: &str, lambda: &[Float], values: &[Float]) {
        let sampled = SampledSpectrumParam {
            lambda: lambda.to_vec(),
            values: values.to_vec(),
        };
        add_value(&mut self.keys, &mut self.sampled_spectra, key, sampled);
    }

    pub fn add_owned_sampled_spectra_typed(
        &mut self,
        parameter_type: &str,
        name: &str,
        values: Vec<SampledSpectrumParam>,
    ) {
        add_owned_values_with_type(
            &mut self.keys,
            &mut self.sampled_spectra,
            parameter_type,
            name,
            values,
        );
    }

    pub fn add_point(&mut self, key: &str, v: &[Float]) {
        add_values(&mut self.keys, &mut self.points, key, v);
    }

    pub fn add_owned_points(&mut self, key: &str, values: Vec<Float>) {
        add_owned_values(&mut self.keys, &mut self.points, key, values);
    }

    pub fn add_owned_points_typed(&mut self, parameter_type: &str, name: &str, values: Vec<Float>) {
        add_owned_values_with_type(
            &mut self.keys,
            &mut self.points,
            parameter_type,
            name,
            values,
        );
    }

    //--------------------

    pub fn add_color(&mut self, key: &str, v: &[Float]) {
        add_values(&mut self.keys, &mut self.points, key, v);
    }

    pub fn add_rgb(&mut self, key: &str, v: &[Float]) {
        add_values(&mut self.keys, &mut self.points, key, v);
    }

    pub fn add_xyz(&mut self, key: &str, v: &[Float]) {
        let xyz = RGBSpectrum::rgb_from_xyz(v);
        let rgb = xyz.to_rgb();
        add_values(&mut self.keys, &mut self.points, key, &rgb);
    }

    pub fn add_blackbody(&mut self, key: &str, v: &[Float]) {
        add_values(&mut self.keys, &mut self.floats, key, v);
    }

    pub fn add_spectrum_no_key(&mut self, key: &str, v: &Spectrum) {
        add_value_no_key(&mut self.spectrums, key, v.clone());
    }

    pub fn add_sampled_spectrum_no_key(&mut self, key: &str, lambda: &[Float], values: &[Float]) {
        let sampled = SampledSpectrumParam {
            lambda: lambda.to_vec(),
            values: values.to_vec(),
        };
        add_value_no_key(&mut self.sampled_spectra, key, sampled);
    }

    //--------------------

    pub fn add_point2f(&mut self, key: &str, v: &Point2f) {
        let tv = [v.x, v.y];
        self.add_floats(key, &tv);
    }

    pub fn add_vector2f(&mut self, key: &str, v: &Vector2f) {
        let tv = [v.x, v.y];
        self.add_floats(key, &tv);
    }

    //--------------------

    pub fn add_point3f(&mut self, key: &str, v: &Point3f) {
        let tv = [v.x, v.y, v.z];
        self.add_point(key, &tv);
    }

    pub fn add_vector3f(&mut self, key: &str, v: &Vector3f) {
        let tv = [v.x, v.y, v.z];
        self.add_point(key, &tv);
    }

    pub fn add_normal3f(&mut self, key: &str, v: &Normal3f) {
        let tv = [v.x, v.y, v.z];
        self.add_point(key, &tv);
    }

    //--------------------

    pub fn get_bools(&self, key: &str) -> Vec<bool> {
        return get_values(&self.bools, key);
    }

    pub fn get_ints(&self, key: &str) -> Vec<i32> {
        return get_values(&self.ints, key);
    }

    pub fn get_floats(&self, key: &str) -> Vec<Float> {
        return get_values(&self.floats, key);
    }

    pub fn get_strings(&self, key: &str) -> Vec<String> {
        return get_values(&self.strings, key);
    }

    pub fn get_spectrums(&self, key: &str) -> Vec<Spectrum> {
        return get_values(&self.spectrums, key);
    }

    pub fn get_sampled_spectra(&self, key: &str) -> Vec<SampledSpectrumParam> {
        return get_values(&self.sampled_spectra, key);
    }

    pub fn get_points(&self, key: &str) -> Vec<Float> {
        return get_values(&self.points, key);
    }

    //--------------------
    pub fn get_bools_ref(&self, key: &str) -> Option<ParameterReadGuard<'_, bool>> {
        return get_values_ref(&self.bools, key);
    }

    pub fn get_ints_ref(&self, key: &str) -> Option<ParameterReadGuard<'_, i32>> {
        return get_values_ref(&self.ints, key);
    }

    pub fn get_floats_ref(&self, key: &str) -> Option<ParameterReadGuard<'_, Float>> {
        return get_values_ref(&self.floats, key);
    }

    pub fn get_strings_ref(&self, key: &str) -> Option<ParameterReadGuard<'_, String>> {
        return get_values_ref(&self.strings, key);
    }

    pub fn get_textures_ref(&self, key: &str) -> Option<ParameterReadGuard<'_, String>> {
        return get_values_ref(&self.strings, key);
    }

    pub fn get_spectrums_ref(&self, key: &str) -> Option<ParameterReadGuard<'_, Spectrum>> {
        return get_values_ref(&self.spectrums, key);
    }

    pub fn get_sampled_spectra_ref(
        &self,
        key: &str,
    ) -> Option<ParameterReadGuard<'_, SampledSpectrumParam>> {
        return get_values_ref(&self.sampled_spectra, key);
    }

    pub fn get_points_ref(&self, key: &str) -> Option<ParameterReadGuard<'_, Float>> {
        return get_values_ref(&self.points, key);
    }

    //--------------------

    pub fn get_bools_mut(&self, key: &str) -> Option<ParameterWriteGuard<'_, bool>> {
        return get_values_mut(&self.bools, key);
    }

    pub fn get_ints_mut(&self, key: &str) -> Option<ParameterWriteGuard<'_, i32>> {
        return get_values_mut(&self.ints, key);
    }

    pub fn get_strings_mut(&self, key: &str) -> Option<ParameterWriteGuard<'_, String>> {
        return get_values_mut(&self.strings, key);
    }

    //--------------------
    pub fn get_one_bool(&self, key: &str, value: bool) -> bool {
        let r = self.get_bools_ref(key);
        match r {
            Some(v) => v[0],
            None => value,
        }
    }

    pub fn get_one_int(&self, key: &str, value: i32) -> i32 {
        let r = self.get_ints_ref(key);
        match r {
            Some(v) => v[0],
            None => value,
        }
    }

    pub fn get_one_float(&self, key: &str, value: Float) -> Float {
        let r = self.get_floats_ref(key);
        match r {
            Some(v) => v[0],
            None => value,
        }
    }

    pub fn get_one_string(&self, key: &str, value: &str) -> String {
        let r = self.get_strings_ref(key);
        match r {
            Some(v) => v[0].clone(),
            None => String::from(value),
        }
    }

    pub fn get_one_filename(&self, key: &str, value: &str) -> String {
        let r = self.get_strings_ref(key);
        match r {
            Some(v) => v[0].clone(),
            None => String::from(value),
        }
    }

    fn is_blackbody_parameter(&self, key: &str) -> bool {
        if get_key_type(key) == "blackbody" {
            return true;
        }
        let key_name = get_key_name(key);
        self.keys.iter().any(|stored_key| {
            get_key_type(stored_key) == "blackbody" && get_key_name(stored_key) == key_name
        })
    }

    pub fn get_one_spectrum_typed(
        &self,
        key: &str,
        value: &Spectrum,
        spectrum_type: SpectrumType,
    ) -> Spectrum {
        if let Some(r) = self.get_points_ref(key) {
            if r.len() >= 3 {
                return Spectrum::from_rgb_in_color_space(
                    self.color_space(),
                    &[r[0], r[1], r[2]],
                    spectrum_type,
                );
            } else if r.len() == 1 {
                return Spectrum::from(r[0]);
            }
        } else if let Some(r) = self.get_sampled_spectra_ref(key) {
            if !r.is_empty() {
                return Spectrum::from_sampled(&r[0].lambda, &r[0].values);
            }
        } else if let Some(r) = self.get_spectrums_ref(key) {
            if !r.is_empty() {
                return r[0].clone();
            }
        } else if let Some(r) = self.get_floats_ref(key) {
            if !r.is_empty() {
                if self.is_blackbody_parameter(key) {
                    return blackbody_spectrum(&r);
                }
                return Spectrum::from(r[0]);
            }
        } else if let Some(r) = self.get_strings_ref(key) {
            if !r.is_empty() {
                let name = &r[0];
                if let Some(alias) = self.get_sampled_spectra_ref(name) {
                    if !alias.is_empty() {
                        return Spectrum::from_sampled(&alias[0].lambda, &alias[0].values);
                    }
                }
                if let Some(alias) = self.get_spectrums_ref(name) {
                    if !alias.is_empty() {
                        return alias[0].clone();
                    }
                }
                if let Some(spec) = spectrum_from_named(name) {
                    return spec;
                }
                if let Some(spec) = spectrum_from_file(name) {
                    return spec;
                }
            }
        }
        let _ = spectrum_type;
        value.clone()
    }

    pub fn get_one_spectrum(&self, key: &str, value: &Spectrum) -> Spectrum {
        self.get_one_spectrum_typed(key, value, SpectrumType::Albedo)
    }

    pub fn get_one_point(&self, key: &str, value: &[Float]) -> Vec<Float> {
        let r = self.get_points_ref(key);
        match r {
            Some(v) => {
                return v.clone();
            }
            None => value.to_vec(),
        }
    }

    pub fn get_one_point3f(&self, key: &str, value: &Point3f) -> Point3f {
        let v = vec![value.x, value.y, value.z];
        let a: &[Float] = &self.get_one_point(key, &v);
        return Point3f::from(a);
    }

    pub fn get_one_vector3f(&self, key: &str, value: &Vector3f) -> Vector3f {
        let v = vec![value.x, value.y, value.z];
        let a: &[Float] = &self.get_one_point(key, &v);
        return Vector3f::from(a);
    }

    pub fn get_one_normal3f(&self, key: &str, value: &Normal3f) -> Normal3f {
        let v = vec![value.x, value.y, value.z];
        let a: &[Float] = &self.get_one_point(key, &v);
        return Normal3f::from(a);
    }

    //--------------------

    pub fn replace_one_bool(&mut self, key: &str, value: bool) {
        let mut replaced = false;
        if let Some(mut r) = self.get_bools_mut(key) {
            if !r.is_empty() {
                r[0] = value;
                replaced = true;
            }
        }
        if !replaced {
            self.add_bool(key, value);
        }
    }

    pub fn replace_one_int(&mut self, key: &str, value: i32) {
        let mut replaced = false;
        if let Some(mut r) = self.get_ints_mut(key) {
            if !r.is_empty() {
                r[0] = value;
                replaced = true;
            }
        }
        if !replaced {
            self.add_int(key, value);
        }
    }

    pub fn replace_one_string(&mut self, key: &str, value: &str) {
        let mut replaced = false;
        if let Some(mut r) = self.get_strings_mut(key) {
            if !r.is_empty() {
                r[0] = String::from(value);
                replaced = true;
            }
        }
        if !replaced {
            self.add_string(key, value);
        }
    }

    pub fn replace_floats(&mut self, key: &str, values: &[Float]) {
        self.add_floats(key, values);
    }

    //--------------------
    pub fn set(&mut self, other: &ParameterDictionary) {
        self.bools = clone_values_map(&other.bools);
        self.ints = clone_values_map(&other.ints);
        self.floats = clone_values_map(&other.floats);
        self.strings = clone_values_map(&other.strings);
        self.spectrums = clone_values_map(&other.spectrums);
        self.sampled_spectra = clone_values_map(&other.sampled_spectra);
        self.points = clone_values_map(&other.points);
        self.keys = other.keys.clone();
    }

    /// Returns `true` if a parameter with the given (type-stripped) name
    /// is already present in any of the typed maps.
    fn has_key_name(&self, name: &str) -> bool {
        self.bools.contains_key(name)
            || self.ints.contains_key(name)
            || self.floats.contains_key(name)
            || self.strings.contains_key(name)
            || self.spectrums.contains_key(name)
            || self.sampled_spectra.contains_key(name)
            || self.points.contains_key(name)
    }

    /// Adds every parameter from `other` that is not already present in
    /// `self`. Parameters already set on `self` take precedence, mirroring
    /// pbrt-v4 where a directive's own parameters win over inherited
    /// `Attribute` parameters.
    pub fn merge_missing(&mut self, other: &ParameterDictionary) {
        for key in other.keys.iter() {
            let name = get_key_name(key);
            if self.has_key_name(&name) {
                continue;
            }
            if let Some(v) = other.bools.get(&name) {
                self.bools
                    .insert(name.clone(), RwLock::new(read_values(v).clone()));
            } else if let Some(v) = other.ints.get(&name) {
                self.ints
                    .insert(name.clone(), RwLock::new(read_values(v).clone()));
            } else if let Some(v) = other.floats.get(&name) {
                self.floats
                    .insert(name.clone(), RwLock::new(read_values(v).clone()));
            } else if let Some(v) = other.strings.get(&name) {
                self.strings
                    .insert(name.clone(), RwLock::new(read_values(v).clone()));
            } else if let Some(v) = other.spectrums.get(&name) {
                self.spectrums
                    .insert(name.clone(), RwLock::new(read_values(v).clone()));
            } else if let Some(v) = other.sampled_spectra.get(&name) {
                self.sampled_spectra
                    .insert(name.clone(), RwLock::new(read_values(v).clone()));
            } else if let Some(v) = other.points.get(&name) {
                self.points
                    .insert(name.clone(), RwLock::new(read_values(v).clone()));
            } else {
                continue;
            }
            self.keys.push(key.clone());
        }
    }
    //--------------------

    pub fn get_keys(&self) -> Vec<String> {
        return self.keys.clone();
    }

    pub fn get_key_name(&self, key: &str) -> String {
        return get_key_name(key);
    }

    pub fn get_key_type(&self, key: &str) -> String {
        return get_key_type(key);
    }

    /// Returns whether a parameter with the given type-stripped name exists.
    pub fn has_parameter(&self, key: &str) -> bool {
        let name = get_key_name(key);
        self.has_key_name(&name)
    }

    /// Renames a parameter in every typed store while preserving its value and
    /// declared type. Callers are responsible for checking name collisions.
    pub fn rename_parameter(&mut self, before: &str, after: &str) -> bool {
        let before = get_key_name(before);
        let after = get_key_name(after);
        let mut found = false;

        macro_rules! rename_map {
            ($map:expr) => {
                if let Some(values) = $map.remove(&before) {
                    $map.insert(after.clone(), values);
                    found = true;
                }
            };
        }
        rename_map!(self.bools);
        rename_map!(self.ints);
        rename_map!(self.floats);
        rename_map!(self.strings);
        rename_map!(self.spectrums);
        rename_map!(self.sampled_spectra);
        rename_map!(self.points);

        for key in &mut self.keys {
            if get_key_name(key) != before {
                continue;
            }
            let ty = get_key_type(key);
            *key = if ty.is_empty() {
                after.clone()
            } else {
                format!("{ty} {after}")
            };
        }
        found
    }

    /// Removes a parameter from every typed store and from the declaration list.
    pub fn remove_parameter(&mut self, key: &str) -> bool {
        let name = get_key_name(key);
        let mut found = false;

        macro_rules! remove_map {
            ($map:expr) => {
                found |= $map.remove(&name).is_some();
            };
        }
        remove_map!(self.bools);
        remove_map!(self.ints);
        remove_map!(self.floats);
        remove_map!(self.strings);
        remove_map!(self.spectrums);
        remove_map!(self.sampled_spectra);
        remove_map!(self.points);
        self.keys.retain(|key| get_key_name(key) != name);
        found
    }

    /// Replaces a legacy blackbody parameter while keeping its declared type.
    pub fn replace_blackbody(&mut self, key: &str, temperature: Float) {
        self.remove_parameter(key);
        self.add_blackbody(&format!("blackbody {key}"), &[temperature]);
    }

    /// Renames references stored in `texture` parameters without changing
    /// ordinary string-valued parameters.
    pub fn rename_texture_references(&mut self, before: &str, after: &str) {
        let names: Vec<String> = self
            .keys
            .iter()
            .filter(|key| get_key_type(key) == "texture")
            .map(|key| get_key_name(key))
            .collect();
        for name in names {
            if let Some(values) = self.strings.get(&name) {
                let mut values = write_values(values);
                for value in values.iter_mut() {
                    if value == before {
                        *value = after.to_string();
                    }
                }
            }
        }
    }

    // Array getters for Point2f/Vector2f/Point3f/Vector3f/Normal3f
    pub fn get_point2f_array(&self, key: &str) -> Vec<Point2f> {
        let mut points = self.get_points(key);
        if points.is_empty() {
            points = self.get_floats(key);
        }
        let mut result = Vec::new();
        for i in (0..points.len()).step_by(2) {
            if i + 1 < points.len() {
                result.push(Point2f::from([points[i], points[i + 1]]));
            }
        }
        result
    }

    pub fn get_vector2f_array(&self, key: &str) -> Vec<Vector2f> {
        let mut points = self.get_points(key);
        if points.is_empty() {
            points = self.get_floats(key);
        }
        let mut result = Vec::new();
        for i in (0..points.len()).step_by(2) {
            if i + 1 < points.len() {
                result.push(Vector2f::from([points[i], points[i + 1]]));
            }
        }
        result
    }

    pub fn get_point3f_array(&self, key: &str) -> Vec<Point3f> {
        let points = self.get_points(key);
        let mut result = Vec::new();
        for i in (0..points.len()).step_by(3) {
            if i + 2 < points.len() {
                result.push(Point3f::from([points[i], points[i + 1], points[i + 2]]));
            }
        }
        result
    }

    pub fn get_vector3f_array(&self, key: &str) -> Vec<Vector3f> {
        let points = self.get_points(key);
        let mut result = Vec::new();
        for i in (0..points.len()).step_by(3) {
            if i + 2 < points.len() {
                result.push(Vector3f::from([points[i], points[i + 1], points[i + 2]]));
            }
        }
        result
    }

    pub fn get_normal3f_array(&self, key: &str) -> Vec<Normal3f> {
        let points = self.get_points(key);
        let mut result = Vec::new();
        for i in (0..points.len()).step_by(3) {
            if i + 2 < points.len() {
                result.push(Normal3f::from([points[i], points[i + 1], points[i + 2]]));
            }
        }
        result
    }

    pub fn get_rgb_array(&self, key: &str) -> Vec<[Float; 3]> {
        let points = self.get_points(key);
        if points.len() % 3 != 0 {
            return Vec::new();
        }
        points
            .chunks_exact(3)
            .map(|rgb| [rgb[0], rgb[1], rgb[2]])
            .collect()
    }

    pub fn get_spectrum_array(&self, key: &str) -> Vec<Spectrum> {
        if let Some(v) = self.get_sampled_spectra_ref(key) {
            return v
                .iter()
                .map(|sampled| Spectrum::from_sampled(&sampled.lambda, &sampled.values))
                .collect();
        }
        if let Some(v) = self.get_spectrums_ref(key) {
            return v.clone();
        }
        if let Some(v) = self.get_points_ref(key) {
            if v.len() == 1 {
                return vec![Spectrum::from(v[0])];
            }
            if v.len() % 3 != 0 {
                return Vec::new();
            }
            let key_name = get_key_name(key);
            let spectrum_type = self
                .keys
                .iter()
                .find(|stored_key| get_key_name(stored_key) == key_name)
                .map(|stored_key| match get_key_type(stored_key).as_str() {
                    "rgb" | "xyz" => SpectrumType::Unbounded,
                    _ => SpectrumType::Albedo,
                })
                .unwrap_or(SpectrumType::Albedo);
            let mut spectra = Vec::with_capacity(v.len() / 3);
            for rgb in v.chunks_exact(3) {
                spectra.push(Spectrum::from_rgb(rgb, spectrum_type));
            }
            return spectra;
        }
        if let Some(v) = self.get_floats_ref(key) {
            return v.iter().map(|x| Spectrum::from(*x)).collect();
        }
        if let Some(v) = self.get_strings_ref(key) {
            let mut spectra = Vec::with_capacity(v.len());
            for name in v.iter() {
                if let Some(spd) = lookup_named_spectrum(name) {
                    spectra.push(spd);
                    continue;
                }
                if Path::new(name).exists() {
                    if let Ok(sampled) = DenseSampledSpectrum::load_sampled_spectrum_file(name) {
                        spectra.push(Spectrum::from(&sampled));
                        continue;
                    }
                }
                return Vec::new();
            }
            return spectra;
        }
        Vec::new()
    }

    pub fn get_one_point2f(&self, key: &str, value: &Point2f) -> Point2f {
        let v = vec![value.x, value.y];
        let a: &[Float] = &self.get_one_point(key, &v);
        return Point2f::from(a);
    }

    pub fn get_one_vector2f(&self, key: &str, value: &Vector2f) -> Vector2f {
        let v = vec![value.x, value.y];
        let a: &[Float] = &self.get_one_point(key, &v);
        return Vector2f::from(a);
    }
}

impl Clone for ParameterDictionary {
    fn clone(&self) -> Self {
        ParameterDictionary {
            bools: clone_values_map(&self.bools),
            ints: clone_values_map(&self.ints),
            floats: clone_values_map(&self.floats),
            strings: clone_values_map(&self.strings),
            spectrums: clone_values_map(&self.spectrums),
            sampled_spectra: clone_values_map(&self.sampled_spectra),
            points: clone_values_map(&self.points),
            keys: self.keys.clone(),
            color_space: self.color_space,
        }
    }
}
