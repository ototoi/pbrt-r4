//! Parse-time state for [`SceneBuilder`] — equivalent to pbrt-v4
//! `BasicSceneBuilder::GraphicsState` plus the transform stack.
//!
//! Unlike [`SceneBuilder`], the only things stored here are **indices
//! into the entity vectors and entity names**. Actual objects
//! (Material / Light / Texture / Medium) are not realised until build
//! time.
//!
//! For simplicity `named_materials` / `named_*_textures` are not
//! scope-tracked; SceneBuilder keeps them in a single global
//! `HashMap`. pbrt-v4's build-time resolution is effectively global
//! too, so the visible difference is minimal.

use crate::paramdict::ParameterDictionary;
use crate::util::spectrum::rgb_to_spectrum::{RGBColorSpace, SRGB};

/// Graphics state for a single `AttributeBegin`/`End` (or `WorldBegin`)
/// scope. Snapshotted on every Shape push so the entity vector can
/// remember which state should drive the eventual Shape / Material /
/// Light realisation.
#[derive(Clone)]
pub struct BuilderGraphicsState {
    /// Most recent anonymous material from `Material "<name>" ...`.
    /// `Some(idx)` indexes into `SceneBuilder.materials`. When both
    /// `current_material_index` and `current_material_name` are `None`,
    /// `current_material_is_default` distinguishes the default-material
    /// case from `interface`.
    pub current_material_index: Option<usize>,
    /// Most recent `NamedMaterial "<name>"` reference (look-up by name).
    pub current_material_name: Option<String>,
    /// `true` initially (no `Material` / `NamedMaterial` set yet).
    /// `Material ""` / `Material "interface"` sets it to `false`
    /// (= no surface material). Setting any other `Material` /
    /// `NamedMaterial` also drops it to `false`. This mirrors
    /// `SceneBuilder::GraphicsState::new()`, which assigns the default
    /// diffuse material to `current_material`; we re-create that
    /// behaviour by checking the flag during build.
    pub current_material_is_default: bool,

    /// Name + parameters of the most recent `AreaLightSource`, to be
    /// bound to the next Shape. Empty = no emissive material.
    pub area_light_name: String,
    pub area_light_params: ParameterDictionary,

    /// `ReverseOrientation` flag. Baked into the entity per Shape.
    pub reverse_orientation: bool,

    /// Medium names set by `MediumInterface` (empty = none). Stored on
    /// the `medium_interface` field of Shape / Light entities.
    pub current_inside_medium: String,
    pub current_outside_medium: String,

    /// Current RGB color space, set by the `ColorSpace` directive.
    /// Applied to every parameter dictionary so RGB values convert to
    /// spectra in the right space. Mirrors pbrt-v4
    /// `BasicSceneBuilder::GraphicsState::colorSpace`.
    pub color_space: &'static RGBColorSpace,

    /// Pending per-target parameters accumulated by the `Attribute`
    /// directive. Merged into the matching directive's own parameters
    /// when the entity is created. Mirrors pbrt-v4
    /// `GraphicsState::shapeAttributes` / `lightAttributes` / etc.
    pub shape_attributes: ParameterDictionary,
    pub light_attributes: ParameterDictionary,
    pub material_attributes: ParameterDictionary,
    pub medium_attributes: ParameterDictionary,
    pub texture_attributes: ParameterDictionary,
}

impl BuilderGraphicsState {
    pub fn new() -> Self {
        Self {
            current_material_index: None,
            current_material_name: None,
            current_material_is_default: true,
            area_light_name: String::new(),
            area_light_params: ParameterDictionary::default(),
            reverse_orientation: false,
            current_inside_medium: String::new(),
            current_outside_medium: String::new(),
            color_space: &SRGB,
            shape_attributes: ParameterDictionary::default(),
            light_attributes: ParameterDictionary::default(),
            material_attributes: ParameterDictionary::default(),
            medium_attributes: ParameterDictionary::default(),
            texture_attributes: ParameterDictionary::default(),
        }
    }
}

impl Default for BuilderGraphicsState {
    fn default() -> Self {
        Self::new()
    }
}

/// API phase (`Options block` ↔ `World block`); same semantics as
/// SceneBuilder's equivalent flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiState {
    OptionsBlock,
    WorldBlock,
}

/// Identifies which kind of push (`AttributeBegin` /
/// `TransformBegin` / `ObjectBegin`) is active so the matching `End`
/// directive pops the correct stacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushKind {
    Attribute,
    Transform,
    Object,
}
