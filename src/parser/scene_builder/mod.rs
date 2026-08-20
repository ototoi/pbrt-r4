//! Counterpart to pbrt-v4 `BasicSceneBuilder` / `BasicScene` (scene.h).
//!
//! Responsibilities:
//!  - On every parser event (`Shape` / `Material` / `LightSource` /
//!    `MakeNamedMedium` / `Texture` / ...) **do not realise** the
//!    object; instead push the matching `*SceneEntity`.
//!  - After `WorldEnd` (in practice once the whole parse is finished)
//!    run a single realisation phase. Mesh / Curve build can be
//!    parallelised with ParallelFor, and material / texture / light
//!    creation has room to become async jobs (v4 `RunAsync` equivalent).
//!
//! Design notes: see `docs/scene_loader_refactor_ja.md`. SceneBuilder
//! is intentionally limited to entity accumulation + parallel build —
//! it does **not** mimic v4 `BasicScene`'s hybrid struct. It coexists
//! with the current `SceneBuilder` (eager build) and the loader can be
//! switched at the CLI level.

pub mod build;
pub mod parse_target_impl;
pub mod path_resolver;
pub mod scene_entity;
pub mod state;

pub use scene_entity::{
    AreaLightSceneEntity, FileLoc, InstanceDefinitionSceneEntity, InstanceSceneEntity,
    LightSceneEntity, MaterialSceneEntity, MediumInterfaceNames, MediumSceneEntity,
    RenderFromObject, SceneEntity, ShapeSceneEntity, TextureKind, TextureSceneEntity,
    TransformedSceneEntity,
};
pub use state::{ApiState, BuilderGraphicsState, PushKind};

use crate::paramdict::ParameterDictionary;
use crate::util::base::Float;
use crate::util::transform::transform_set::{TransformSet, ALL_TRANSFORM_BITS};

use std::collections::HashMap;
use std::sync::OnceLock;

const CURVES_SHAPE_NAME: &str = "curves";

/// Parse-time state for the SceneBuilder loader, equivalent to the
/// `BasicScene` + `BasicSceneBuilder` pair in pbrt-v4. The entity
/// vectors accumulate immutably until `build()`; the other fields
/// evolve as parsing progresses.
pub struct SceneBuilder {
    // === Entity output (not mutated until build) =============================
    pub shapes: Vec<ShapeSceneEntity>,
    pub animated_shapes: Vec<ShapeSceneEntity>,
    pub materials: Vec<MaterialSceneEntity>,
    pub named_materials: HashMap<String, usize>,
    pub lights: Vec<LightSceneEntity>,
    pub area_lights: Vec<AreaLightSceneEntity>,
    pub media: HashMap<String, MediumSceneEntity>,
    pub float_textures: Vec<TextureSceneEntity>,
    pub spectrum_textures: Vec<TextureSceneEntity>,
    pub named_float_textures: HashMap<String, usize>,
    pub named_spectrum_textures: HashMap<String, usize>,
    pub instance_definitions: HashMap<String, InstanceDefinitionSceneEntity>,
    pub instance_uses: Vec<InstanceSceneEntity>,

    // === Render-level options (set inside the `Options` block) ===============
    pub filter_name: String,
    pub filter_params: ParameterDictionary,
    pub film_name: String,
    pub film_params: ParameterDictionary,
    pub sampler_name: String,
    pub sampler_params: ParameterDictionary,
    pub accelerator_name: String,
    pub accelerator_params: ParameterDictionary,
    pub integrator_name: String,
    pub integrator_params: ParameterDictionary,
    pub camera_name: String,
    pub camera_params: ParameterDictionary,
    pub camera_to_world: TransformSet,
    pub transform_start_time: Float,
    pub transform_end_time: Float,

    // === Parse-time state ====================================================
    pub api_state: ApiState,
    pub transforms: Vec<TransformSet>,
    pub transform_bits: Vec<u32>,
    pub graphics_states: Vec<BuilderGraphicsState>,
    pub named_coordinate_systems: HashMap<String, TransformSet>,
    pub push_stack: Vec<PushKind>,
    pub work_dirs: Vec<String>,
    /// Every directory ever pushed via `work_dir_begin`, in arrival
    /// order. Reused at `build()` as the search path for
    /// `SceneBuilder::get_filepath`.
    pub seen_work_dirs: Vec<String>,
    /// While inside an `ObjectBegin "<name>"` scope, shapes are pushed
    /// to the matching `InstanceDefinitionSceneEntity` instead of the
    /// top-level scene.
    pub current_instance_name: Option<String>,
    pub import_errors: Vec<String>,
    pub option_errors: Vec<String>,
}

impl SceneBuilder {
    fn default_transform_set() -> &'static TransformSet {
        static DEFAULT: OnceLock<TransformSet> = OnceLock::new();
        DEFAULT.get_or_init(TransformSet::new)
    }

    fn default_graphics_state() -> &'static BuilderGraphicsState {
        static DEFAULT: OnceLock<BuilderGraphicsState> = OnceLock::new();
        DEFAULT.get_or_init(BuilderGraphicsState::new)
    }

    pub fn new() -> Self {
        let accelerator_name =
            std::env::var("PBRT_ACCELERATOR").unwrap_or_else(|_| "bvh".to_string());

        let mut builder = Self {
            shapes: Vec::new(),
            animated_shapes: Vec::new(),
            materials: Vec::new(),
            named_materials: HashMap::new(),
            lights: Vec::new(),
            area_lights: Vec::new(),
            media: HashMap::new(),
            float_textures: Vec::new(),
            spectrum_textures: Vec::new(),
            named_float_textures: HashMap::new(),
            named_spectrum_textures: HashMap::new(),
            instance_definitions: HashMap::new(),
            instance_uses: Vec::new(),

            // pbrt-v4 BasicSceneBuilder uses "gaussian" (scene.cpp:96).
            filter_name: String::from("gaussian"),
            filter_params: ParameterDictionary::default(),
            film_name: String::from("rgb"),
            film_params: ParameterDictionary::default(),
            sampler_name: String::from("zsobol"),
            sampler_params: ParameterDictionary::default(),
            accelerator_name,
            accelerator_params: ParameterDictionary::default(),
            // pbrt-v4 BasicSceneBuilder uses "volpath" as the default
            // (scene.cpp:95). With "path", BSSRDF and medium code paths are
            // skipped, so scenes that don't specify an integrator
            // (sssdragon, dambreak, ...) diverge visibly from v4.
            integrator_name: String::from("volpath"),
            integrator_params: ParameterDictionary::default(),
            camera_name: String::from("perspective"),
            camera_params: ParameterDictionary::default(),
            camera_to_world: TransformSet::new(),
            transform_start_time: 0.0,
            transform_end_time: 1.0,

            api_state: ApiState::OptionsBlock,
            transforms: Vec::new(),
            transform_bits: Vec::new(),
            graphics_states: Vec::new(),
            named_coordinate_systems: HashMap::new(),
            push_stack: Vec::new(),
            work_dirs: Vec::new(),
            seen_work_dirs: Vec::new(),
            current_instance_name: None,
            import_errors: Vec::new(),
            option_errors: Vec::new(),
        };
        builder.initialize_stacks();
        builder
    }

    fn initialize_stacks(&mut self) {
        self.transforms.clear();
        self.transform_bits.clear();
        self.graphics_states.clear();
        self.transforms.push(TransformSet::new());
        self.transform_bits.push(ALL_TRANSFORM_BITS);
        self.graphics_states.push(BuilderGraphicsState::new());
    }

    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
            && self.animated_shapes.is_empty()
            && self.materials.is_empty()
            && self.named_materials.is_empty()
            && self.lights.is_empty()
            && self.area_lights.is_empty()
            && self.media.is_empty()
            && self.float_textures.is_empty()
            && self.spectrum_textures.is_empty()
            && self.instance_definitions.is_empty()
            && self.instance_uses.is_empty()
    }

    // === Parse-time helpers ==================================================

    pub fn push_transform(&mut self) {
        if let Some(top) = self.transforms.last().copied() {
            self.transforms.push(top);
        }
        if let Some(bits) = self.transform_bits.last().copied() {
            self.transform_bits.push(bits);
        }
    }

    pub fn pop_transform(&mut self) {
        self.transforms.pop();
        self.transform_bits.pop();
    }

    pub fn push_graphics_state(&mut self) {
        if let Some(top) = self.graphics_states.last().cloned() {
            self.graphics_states.push(top);
        }
    }

    pub fn pop_graphics_state(&mut self) {
        self.graphics_states.pop();
    }

    pub fn top_transform(&self) -> &TransformSet {
        if self.transforms.is_empty() {
            return Self::default_transform_set();
        }
        self.transforms
            .last()
            .expect("transform stack is non-empty after the fallback check")
    }

    pub fn top_transform_mut(&mut self) -> &mut TransformSet {
        if self.transforms.is_empty() {
            self.transforms.push(TransformSet::new());
        }
        self.transforms
            .last_mut()
            .expect("transform stack is non-empty after fallback initialization")
    }

    pub fn top_transform_bits(&self) -> u32 {
        self.transform_bits
            .last()
            .copied()
            .unwrap_or(ALL_TRANSFORM_BITS)
    }

    pub fn set_top_transform_bits(&mut self, bits: u32) {
        if let Some(b) = self.transform_bits.last_mut() {
            *b = bits;
        }
    }

    pub fn top_graphics_state(&self) -> &BuilderGraphicsState {
        if self.graphics_states.is_empty() {
            return Self::default_graphics_state();
        }
        self.graphics_states
            .last()
            .expect("graphics state stack is non-empty after the fallback check")
    }

    pub fn top_graphics_state_mut(&mut self) -> &mut BuilderGraphicsState {
        if self.graphics_states.is_empty() {
            self.graphics_states.push(BuilderGraphicsState::new());
        }
        self.graphics_states
            .last_mut()
            .expect("graphics state stack is non-empty after fallback initialization")
    }

    /// Build a `RenderFromObject` from the current transform. Returns
    /// `Animated` when the `TransformSet` holds two distinct matrices.
    pub fn render_from_object(&self) -> RenderFromObject {
        let cur = self.top_transform();
        if cur.is_animated() {
            RenderFromObject::Animated {
                from: cur[0],
                to: cur[1],
                start_time: self.transform_start_time,
                end_time: self.transform_end_time,
            }
        } else {
            RenderFromObject::Static(cur[0])
        }
    }

    pub fn current_medium_interface(&self) -> MediumInterfaceNames {
        let gs = self.top_graphics_state();
        MediumInterfaceNames::new(&gs.current_inside_medium, &gs.current_outside_medium)
    }

    pub fn current_file_loc(&self) -> FileLoc {
        FileLoc::default()
    }
}

impl Default for SceneBuilder {
    fn default() -> Self {
        Self::new()
    }
}
