//! pbrt-v4 `*SceneEntity` family of types (scene.h).
//!
//! When the parser encounters a `Shape` / `Material` / `LightSource`
//! token, v4's load strategy is to **not realise the object yet** and
//! instead push the matching `*SceneEntity` onto the `SceneBuilder`.
//! Realisation (PLY load, TriangleMesh build, BVH build, ...) is
//! deferred until after `WorldEnd` and realized in the scene build phase.

use crate::paramdict::ParameterDictionary;
use crate::util::base::Float;
use crate::util::transform::Transform;

/// pbrt-v4 `FileLoc` (parser.h) — source location captured at parse
/// time. Used for error messages and to surface the responsible scope
/// in deferred realisation steps.
#[derive(Clone, Debug, Default)]
pub struct FileLoc {
    pub filename: String,
    pub line: u32,
    pub column: u32,
}

impl FileLoc {
    pub fn new(filename: impl Into<String>, line: u32, column: u32) -> Self {
        Self {
            filename: filename.into(),
            line,
            column,
        }
    }
}

/// Rust counterpart of pbrt-v4 `SceneEntity` (scene.h): a minimal
/// triple of token name, parameter bundle, and source location.
///
/// v4 shares names via `InternedString`; r4 uses a plain `String`.
#[derive(Clone)]
pub struct SceneEntity {
    pub name: String,
    pub params: ParameterDictionary,
    pub loc: FileLoc,
}

impl SceneEntity {
    pub fn new(name: impl Into<String>, params: ParameterDictionary, loc: FileLoc) -> Self {
        Self {
            name: name.into(),
            params,
            loc,
        }
    }
}

/// pbrt-v4 `TransformedSceneEntity` (scene.h) — `SceneEntity` plus a
/// `renderFromObject: AnimatedTransform`. The light entities derive
/// from this in v4.
///
/// In r4 we don't have a dedicated `AnimatedTransform`; the
/// `RenderFromObject` enum carries either a static transform or an
/// animated pair.
#[derive(Clone)]
pub struct TransformedSceneEntity {
    pub base: SceneEntity,
    pub render_from_object: RenderFromObject,
}

/// Name-based MediumInterface — media declared via `MakeNamedMedium`
/// are resolved by name later. An empty string means "no medium"
/// (same convention as v4).
///
/// The realised [`crate::media::MediumInterface`] (a pair of
/// `Arc<Medium>`) is used by the SceneBuilder path and at render
/// time, but for the deferred-build intermediate it is safer to hold
/// only the names (Medium itself becomes an entity in SceneBuilder
/// too).
#[derive(Clone, Default, PartialEq, Eq)]
pub struct MediumInterfaceNames {
    pub inside_medium: String,
    pub outside_medium: String,
}

impl MediumInterfaceNames {
    pub fn new(inside: impl Into<String>, outside: impl Into<String>) -> Self {
        Self {
            inside_medium: inside.into(),
            outside_medium: outside.into(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inside_medium.is_empty() && self.outside_medium.is_empty()
    }
}

/// Transform attached to a Shape / Light at parse time. `Static` holds
/// a single matrix; `Animated` holds the two-matrix interpolation
/// pair plus the time interval. This is the source data for building
/// v4's `AnimatedTransform` equivalent in the end.
#[derive(Clone, PartialEq)]
pub enum RenderFromObject {
    Static(Transform),
    Animated {
        from: Transform,
        to: Transform,
        start_time: Float,
        end_time: Float,
    },
}

impl Default for RenderFromObject {
    fn default() -> Self {
        Self::Static(Transform::identity())
    }
}

impl RenderFromObject {
    pub fn is_animated(&self) -> bool {
        matches!(self, Self::Animated { .. })
    }

    /// Returns a single `Transform` for the static case. For an
    /// animated transform, returns the start-time matrix.
    pub fn primary(&self) -> Transform {
        match self {
            Self::Static(t) => *t,
            Self::Animated { from, .. } => *from,
        }
    }
}

/// pbrt-v4 `ShapeSceneEntity`. One `Shape "<name>"` directive's
/// parameters plus a snapshot of the graphics state at that point.
///
/// `material_index` / `area_light_index` / `medium_interface` are
/// designed to **index other entities inside the same `SceneBuilder`**
/// (like v4). `MakeNamedMaterial` look-ups, etc. are resolved in
/// `SceneBuilder::done()`.
///
/// The constructor helper stores the entity parameters without realizing
/// the referenced scene object.
/// creates these yet.
#[derive(Clone)]
pub struct ShapeSceneEntity {
    pub base: SceneEntity,
    /// Parameters for consecutive child shapes that share this entity's
    /// transform and shading state. Empty for shapes that are not grouped.
    pub child_params: Vec<ParameterDictionary>,
    pub render_from_object: RenderFromObject,
    pub reverse_orientation: bool,
    /// Resolved later as `materials[material_index]`. `usize::MAX`
    /// means unresolved (linked by `NamedMaterial` name instead).
    pub material_index: usize,
    pub material_name: Option<String>,
    /// Resolved later as `area_lights[area_light_index]`. `None` =
    /// no emissive material.
    pub area_light_index: Option<usize>,
    /// When `material_index == usize::MAX && material_name == None`,
    /// this flag indicates that the **default diffuse material**
    /// should be assigned. SceneBuilder initialises
    /// `current_material` to the default diffuse in
    /// `GraphicsState::new()`, so shapes with no `Material` /
    /// `NamedMaterial` directive still get the default. Conversely,
    /// shapes that explicitly use `Material ""` or
    /// `Material "interface"` carry no surface material, so the
    /// flag is `false`.
    pub material_is_default: bool,
    /// MediumInterface is held by name (media declared with
    /// `MakeNamedMedium` are resolved at build time).
    pub medium_interface: MediumInterfaceNames,
    /// Set when inside an `ObjectBegin/End` scope: the corresponding
    /// instance name. `None` means the shape goes straight into the
    /// main scene.
    pub instance_name: Option<String>,
}

/// pbrt-v4 `MaterialSceneEntity` (scene.h). Stores both
/// `MakeNamedMaterial` and `Material` in the same type. Anonymous
/// (`Material`) entries have `name == ""`.
#[derive(Clone)]
pub struct MaterialSceneEntity {
    pub base: SceneEntity,
}

/// pbrt-v4 `LightSceneEntity` (scene.h) — one `LightSource`
/// directive. This is a direct light (`infinite` / `distant` /
/// `point` / ...), not an AreaLight.
#[derive(Clone)]
pub struct LightSceneEntity {
    pub base: TransformedSceneEntity,
    pub medium: String,
}

/// One `AreaLightSource` directive. Only meaningful once bound to a
/// Shape, so it is reached via `ShapeSceneEntity::area_light_index`.
#[derive(Clone)]
pub struct AreaLightSceneEntity {
    pub base: SceneEntity,
    pub render_from_light: RenderFromObject,
    pub medium_interface: MediumInterfaceNames,
}

/// pbrt-v4 `MediumSceneEntity` (scene.h). One `MakeNamedMedium`
/// directive.
#[derive(Clone)]
pub struct MediumSceneEntity {
    pub base: SceneEntity,
    pub render_from_medium: RenderFromObject,
}

/// pbrt-v4 `TextureSceneEntity` (scene.h).
#[derive(Clone)]
pub struct TextureSceneEntity {
    pub base: SceneEntity,
    pub texture_kind: TextureKind,
    /// Transform captured at the Texture directive, matching v4's
    /// renderFromTexture. Mapping implementations invert it as needed.
    pub render_from_texture: Transform,
}

/// Second argument of the `Texture` directive (`spectrum` /
/// `float`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureKind {
    Float,
    Spectrum,
}

/// pbrt-v4 `InstanceSceneEntity` — one `ObjectInstance` directive.
#[derive(Clone)]
pub struct InstanceSceneEntity {
    pub name: String,
    pub loc: FileLoc,
    pub render_from_instance: RenderFromObject,
}

/// One `ObjectBegin..ObjectEnd` block. Collects the Shape entities
/// and light entities declared inside, so they can be aggregated
/// later in one go.
#[derive(Clone, Default)]
pub struct InstanceDefinitionSceneEntity {
    pub name: String,
    pub loc: FileLoc,
    pub shapes: Vec<ShapeSceneEntity>,
    pub animated_shapes: Vec<ShapeSceneEntity>,
}
