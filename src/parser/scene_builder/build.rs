//! [`SceneBuilder::build`] — realise the accumulated entities directly
//! into an `Integrator`.
//!
//! `build` calls
//! each `create_*` free function in order to construct Texture /
//! Material / Light / Shape / Aggregate / Camera / Integrator without
//! re-routing through SceneBuilder.
//!
//! Phase 1: create float / spectrum textures in declaration order.
//! Phase 2: create named media (`MakeNamedMedium`).
//! Phase 3: create materials in declaration order (the "mix" case
//!          references previously created named materials).
//! Phase 4: gather InstanceDefinition primitives into per-instance
//!          aggregates.
//! Phase 5: turn top-level shapes into primitives and emit any area
//!          lights they own.
//! Phase 6: wrap InstanceUse into TransformedPrimitive and add to the
//!          top-level scene.
//! Phase 7: create top-level lights.
//! Phase 8: build the top-level Aggregate (BVH) and the `Scene`.
//! Phase 9: assemble Camera / Filter / Film / Sampler / Integrator.
//!
//! Design notes: see `docs/scene_loader_refactor_ja.md`.
//!
//! Texture / named-material names live in a global namespace across
//! `AttributeBegin/End` (duplicate names error). This matches pbrt-v4
//! (`pbrt-v4/src/pbrt/scene.cpp` `BasicSceneBuilder::Texture` /
//! `MakeNamedMaterial`); only `graphicsState` (ctm, colorSpace, *Attributes,
//! currentMaterialIndex, areaLightName, inside/outside medium, etc.) is
//! stacked.

use super::path_resolver::make_absolute_path;
use super::scene_entity::{
    InstanceDefinitionSceneEntity, InstanceSceneEntity, MediumInterfaceNames, RenderFromObject,
    ShapeSceneEntity,
};
use super::{SceneBuilder, CURVES_SHAPE_NAME};
use crate::base::camera::Camera;
use crate::base::filter::Filter;
use crate::base::light::Light;
use crate::base::material::Material;
use crate::base::sampler::Sampler;
use crate::base::shape::Shape;
use crate::base::texture::typed_spectrum_texture_name;
use crate::cpu::aggregates::create_accelerator;
use crate::cpu::integrators::create_integrator;
use crate::cpu::integrators::integrator::Integrator;
use crate::cpu::primitive::*;
use crate::film::Film;
use crate::materials::MixMaterial;
use crate::media::medium::Medium;
use crate::media::medium_interface::MediumInterface;
use crate::options::pbrt_options::{PbrtOptions, RenderingCoordinateSystem};
use crate::paramdict::{ParameterDictionary, TextureParameterDictionary};
use crate::scene::Scene;
use crate::textures::{
    FloatImageTexture, FloatTexture, ImageMapMIPMapCache, SpectrumImageTexture, SpectrumTexture,
};
use crate::util::error::PbrtError;
use crate::util::spectrum::source::{
    canonical_spectrum_type, SPECTRUM_CLASS_ALBEDO, SPECTRUM_CLASS_ILLUMINANT,
    SPECTRUM_CLASS_UNBOUNDED,
};
use crate::util::transform::animated_transform::AnimatedTransform;
use crate::util::transform::camera_transform::CameraTransform;
use crate::util::transform::transform::Transform;
use crate::util::transform::transform_set::TransformSet;

use log::{error, warn};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

type FloatTextureMap = HashMap<String, Arc<FloatTexture>>;
type SpectrumTextureMap = HashMap<String, Arc<SpectrumTexture>>;
type MediumMap = HashMap<String, Arc<Medium>>;

impl SceneBuilder {
    /// pbrt-v4 builds `CameraTransform` near the start of `RenderCpu`
    /// from the CTM that was active at the `Camera` directive. r4
    /// captured that CTM into `self.camera_to_world` (the name is
    /// historical — it actually holds `cameraFromWorld`, see
    /// `parse_target_impl.rs::camera`), so we invert here to obtain
    /// `worldFromCamera` and feed it through `CameraTransform::new`.
    fn build_camera_transform(&self) -> Result<CameraTransform, PbrtError> {
        let mut world_from_camera = TransformSet::new();
        world_from_camera[0] = self.camera_to_world[0].inverse();
        world_from_camera[1] = self.camera_to_world[1].inverse();
        let world_from_camera_animated = AnimatedTransform::new(
            &world_from_camera[0],
            self.transform_start_time,
            &world_from_camera[1],
            self.transform_end_time,
        )
        .ok_or_else(|| PbrtError::error("camera transform decomposition failed"))?;
        CameraTransform::new(
            &world_from_camera_animated,
            PbrtOptions::get().rendering_space,
        )
        .ok_or_else(|| PbrtError::error("camera transform decomposition failed"))
    }

    /// Realise the accumulated entities directly into an `Integrator`
    /// (SceneBuilder is not involved).
    pub fn build(&self) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
        if let Some(error) = self.import_errors.first() {
            return Err(PbrtError::error(error));
        }
        if let Some(error) = self.option_errors.first() {
            return Err(PbrtError::error(error));
        }
        // pbrt-v4 computes `CameraTransform` before realising any
        // scene entity; `worldFromRender` is then pre-multiplied into
        // every shape / light / medium / instance transform so the
        // entire pipeline operates in render space (cameras.cpp:27).
        // For `RenderingCoordinateSystem::World` this collapses to
        // identity and the established scene-space behavior is preserved.
        let camera_transform = self.build_camera_transform()?;
        let render_from_world = camera_transform.render_from_world();

        // Phase 1: Textures
        let (float_tex, spectrum_tex) = self.realize_textures(&render_from_world)?;

        // Phase 2: Media
        let named_media = self.realize_media(&render_from_world)?;

        // Phase 3: Materials (in declaration order; mix references resolve from
        // already-built materials, matching SceneBuilder eager semantics).
        let (materials, named_materials, default_material) =
            self.realize_materials(&float_tex, &spectrum_tex)?;
        let scene_build_pool = Self::scene_build_thread_pool()?;
        let scene_build_pool = scene_build_pool.as_ref();

        // Phase 4: Instance definitions. Instance internals are kept
        // in instance-local space; only the per-use placement (Phase 6)
        // gets `renderFromWorld` applied.
        let instance_aggregates = self.realize_instance_definitions(
            &float_tex,
            &spectrum_tex,
            &materials,
            &named_materials,
            &default_material,
            &named_media,
            scene_build_pool,
        )?;

        // Phase 5: top-level shapes + area lights
        let (mut top_primitives, mut top_lights) = self.realize_top_level_shapes(
            &float_tex,
            &spectrum_tex,
            &materials,
            &named_materials,
            &default_material,
            &named_media,
            &render_from_world,
            scene_build_pool,
        )?;

        // Phase 6: instance uses
        for inst in &self.instance_uses {
            if let Some(prim) =
                self.realize_instance_use(inst, &instance_aggregates, &render_from_world)
            {
                top_primitives.push(prim);
            }
        }

        // Phase 7: top-level lights
        let lights = self.realize_lights(&named_media, &mut top_lights, &render_from_world)?;

        // Phase 8: top-level Aggregate → Scene
        let aggregate = create_accelerator(
            &self.accelerator_name,
            &top_primitives,
            &self.accelerator_params,
        )
        .map_err(|e| PbrtError::error(&format!("create_accelerator failed: {}", e.msg)))?;
        let aggregate = Arc::new(aggregate);
        let scene = Arc::new(Scene::new(&aggregate, &lights));

        // Phase 9: Camera / Filter / Film / Sampler / Integrator
        let filter = self.realize_filter()?;
        let film = self.realize_film(&filter)?;
        let camera = self.realize_camera(&film, &named_media, &camera_transform)?;
        let camera = Arc::new(camera);
        let sampler = self.realize_sampler(&film)?;
        let have_scattering_media = !named_media.is_empty();
        if have_scattering_media
            && self.integrator_name != "volpath"
            && self.integrator_name != "bdpt"
            && self.integrator_name != "mlt"
        {
            warn!(
                "Scene has scattering media but \"{}\" integrator doesn't support volume scattering. Use \"volpath\", \"bdpt\" or \"mlt\".",
                self.integrator_name
            );
        }
        if lights.is_empty() {
            warn!("No light sources defined in scene; rendering a black image.");
        }
        create_integrator(
            &self.integrator_name,
            &self.integrator_params,
            &sampler,
            &camera,
            scene.as_ref(),
        )
        .map_err(|e| PbrtError::error(&format!("create_integrator failed: {}", e.msg)))
    }

    /// Realise the accumulated entities directly into an `Integrator` on GPU.
    pub fn build_gpu(&self) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
        return Err(PbrtError::error("GPU build not implemented yet"));
    }

    // ===== Phase helpers ====================================================

    fn scene_build_thread_pool() -> Result<Option<ThreadPool>, PbrtError> {
        let options = PbrtOptions::get();
        if !options.parallel_scene_build {
            return Ok(None);
        }

        let jobs = options.scene_build_jobs.max(1);
        ThreadPoolBuilder::new()
            .num_threads(jobs)
            .thread_name(|idx| format!("pbrt-scene-build-{idx}"))
            .build()
            .map(Some)
            .map_err(|e| {
                PbrtError::error(&format!(
                    "failed to create scene-build thread pool with {} jobs: {}",
                    jobs, e
                ))
            })
    }

    fn realize_textures(
        &self,
        render_from_world: &Transform,
    ) -> Result<(FloatTextureMap, SpectrumTextureMap), PbrtError> {
        use rayon::prelude::*;

        let render_from_world = *render_from_world;

        let mut float_tex: FloatTextureMap = HashMap::new();
        let mut spectrum_tex: SpectrumTextureMap = HashMap::new();
        let inv_named_float = invert_named_map(&self.named_float_textures);
        let inv_named_spectrum = invert_named_map(&self.named_spectrum_textures);

        // Split into "leaf" textures (no `"texture <name>"` parameter references)
        // and "chain" textures. Leaves can be built in parallel (each is
        // self-contained), chains must be built in declaration order so their
        // referenced textures are already in the map.
        let float_leaf_idxs: Vec<usize> = (0..self.float_textures.len())
            .filter(|&i| {
                inv_named_float.contains_key(&i)
                    && !has_texture_ref(&self.float_textures[i].base.params)
            })
            .collect();
        let float_chain_idxs: Vec<usize> = (0..self.float_textures.len())
            .filter(|&i| {
                inv_named_float.contains_key(&i)
                    && has_texture_ref(&self.float_textures[i].base.params)
            })
            .collect();
        let spectrum_leaf_idxs: Vec<usize> = (0..self.spectrum_textures.len())
            .filter(|&i| {
                inv_named_spectrum.contains_key(&i)
                    && !has_texture_ref(&self.spectrum_textures[i].base.params)
            })
            .collect();
        let spectrum_chain_idxs: Vec<usize> = (0..self.spectrum_textures.len())
            .filter(|&i| {
                inv_named_spectrum.contains_key(&i)
                    && has_texture_ref(&self.spectrum_textures[i].base.params)
            })
            .collect();

        // Empty maps used as TextureParameterDictionary placeholders during leaf build.
        // Leaves have no texture refs, so the maps are never consulted.
        let empty_float: FloatTextureMap = HashMap::new();
        let empty_spectrum: SpectrumTextureMap = HashMap::new();

        let build_parallel = PbrtOptions::get().parallel_texture_build;
        let imagemap_mipmaps = std::sync::Mutex::new(ImageMapMIPMapCache::default());
        let build_leaf_float = |idx: usize| -> Result<(String, Arc<FloatTexture>), PbrtError> {
            let tex = &self.float_textures[idx];
            let user_name = inv_named_float
                .get(&idx)
                .cloned()
                .ok_or_else(|| PbrtError::error("unnamed leaf float texture"))?;
            let params = make_absolute_path(&tex.base.params, &self.seen_work_dirs);
            let tp = TextureParameterDictionary::new(&params, &empty_float, &empty_spectrum);
            let render_from_texture = render_from_world * tex.render_from_texture;
            let created = if tex.base.name == "imagemap" {
                FloatImageTexture::create_with_mipmap_cache(
                    &render_from_texture,
                    &tp,
                    Some(&imagemap_mipmaps),
                )
            } else {
                FloatTexture::create(&tex.base.name, &render_from_texture, &tp)
            };
            created.map(|t| (user_name, Arc::new(t)))
        };

        // ---- Phase A: build leaf float textures -----------------------------
        let leaf_float: Vec<(String, Arc<FloatTexture>)> = if build_parallel {
            float_leaf_idxs
                .par_iter()
                .map(|&idx| build_leaf_float(idx))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            float_leaf_idxs
                .iter()
                .map(|&idx| build_leaf_float(idx))
                .collect::<Result<Vec<_>, _>>()?
        };
        for (name, t) in leaf_float {
            float_tex.insert(name, t);
        }

        // ---- Phase B: build leaf spectrum textures --------------------------
        // 3 spectrum classes per entity; each class needs its own
        // SpectrumTexture::create. Leaf spectrum textures may legitimately
        // reference float textures (already in `float_tex`), so pass that map.
        let build_leaf_spectrum =
            |idx: usize| -> Result<Vec<(String, Arc<SpectrumTexture>)>, PbrtError> {
                let tex = &self.spectrum_textures[idx];
                let user_name = match inv_named_spectrum.get(&idx) {
                    Some(n) => n.clone(),
                    None => return Ok(Vec::new()),
                };
                let params = make_absolute_path(&tex.base.params, &self.seen_work_dirs);
                let mut out = Vec::with_capacity(3);
                if tex.base.name == "imagemap" {
                    let spectrum_types = [
                        canonical_spectrum_type(SPECTRUM_CLASS_ALBEDO).ok_or_else(|| {
                            PbrtError::error("built-in spectrum class ALBEDO missing")
                        })?,
                        canonical_spectrum_type(SPECTRUM_CLASS_UNBOUNDED).ok_or_else(|| {
                            PbrtError::error("built-in spectrum class UNBOUNDED missing")
                        })?,
                        canonical_spectrum_type(SPECTRUM_CLASS_ILLUMINANT).ok_or_else(|| {
                            PbrtError::error("built-in spectrum class ILLUMINANT missing")
                        })?,
                    ];
                    let tp = TextureParameterDictionary::new(&params, &float_tex, &empty_spectrum);
                    let render_from_texture = render_from_world * tex.render_from_texture;
                    match SpectrumImageTexture::create_variants_with_mipmap_cache(
                        &render_from_texture,
                        &tp,
                        &spectrum_types,
                        Some(&imagemap_mipmaps),
                    ) {
                        Ok(textures) => {
                            for (spectrum_type, texture) in textures {
                                out.push((
                                    typed_spectrum_texture_name(&user_name, spectrum_type),
                                    Arc::new(SpectrumTexture::ImageMap(texture)),
                                ));
                            }
                        }
                        Err(e) => return Err(e),
                    }
                    return Ok(out);
                }
                for class in [
                    SPECTRUM_CLASS_ALBEDO,
                    SPECTRUM_CLASS_UNBOUNDED,
                    SPECTRUM_CLASS_ILLUMINANT,
                ] {
                    let spectrum_type = canonical_spectrum_type(class).ok_or_else(|| {
                        PbrtError::error(&format!("built-in spectrum class {} missing", class))
                    })?;
                    let tp = TextureParameterDictionary::new(&params, &float_tex, &empty_spectrum);
                    let render_from_texture = render_from_world * tex.render_from_texture;
                    match SpectrumTexture::create(
                        &tex.base.name,
                        &render_from_texture,
                        &tp,
                        spectrum_type,
                    ) {
                        Ok(t) => out.push((
                            typed_spectrum_texture_name(&user_name, spectrum_type),
                            Arc::new(t),
                        )),
                        Err(e) => return Err(e),
                    }
                }
                Ok(out)
            };
        let leaf_spectrum: Vec<Vec<(String, Arc<SpectrumTexture>)>> = if build_parallel {
            spectrum_leaf_idxs
                .par_iter()
                .map(|&idx| build_leaf_spectrum(idx))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            spectrum_leaf_idxs
                .iter()
                .map(|&idx| build_leaf_spectrum(idx))
                .collect::<Result<Vec<_>, _>>()?
        };
        for group in leaf_spectrum {
            for (name, t) in group {
                spectrum_tex.insert(name, t);
            }
        }

        // ---- Phase C: sequential build chain float textures -----------------
        // Declaration order matters: a chain texture may reference a
        // previously-declared (possibly chain) texture.
        for idx in float_chain_idxs {
            let tex = &self.float_textures[idx];
            let user_name = match inv_named_float.get(&idx) {
                Some(n) => n.clone(),
                None => continue,
            };
            let params = make_absolute_path(&tex.base.params, &self.seen_work_dirs);
            let tp = TextureParameterDictionary::new(&params, &float_tex, &spectrum_tex);
            let render_from_texture = render_from_world * tex.render_from_texture;
            let created = if tex.base.name == "imagemap" {
                FloatImageTexture::create_with_mipmap_cache(
                    &render_from_texture,
                    &tp,
                    Some(&imagemap_mipmaps),
                )
            } else {
                FloatTexture::create(&tex.base.name, &render_from_texture, &tp)
            };
            match created {
                Ok(t) => {
                    float_tex.insert(user_name, Arc::new(t));
                }
                Err(e) => return Err(e),
            }
        }

        // ---- Phase D: sequential build chain spectrum textures --------------
        for idx in spectrum_chain_idxs {
            let tex = &self.spectrum_textures[idx];
            let user_name = match inv_named_spectrum.get(&idx) {
                Some(n) => n.clone(),
                None => continue,
            };
            let params = make_absolute_path(&tex.base.params, &self.seen_work_dirs);
            if tex.base.name == "imagemap" {
                let spectrum_types = [
                    canonical_spectrum_type(SPECTRUM_CLASS_ALBEDO).ok_or_else(|| {
                        PbrtError::error("built-in spectrum class ALBEDO missing")
                    })?,
                    canonical_spectrum_type(SPECTRUM_CLASS_UNBOUNDED).ok_or_else(|| {
                        PbrtError::error("built-in spectrum class UNBOUNDED missing")
                    })?,
                    canonical_spectrum_type(SPECTRUM_CLASS_ILLUMINANT).ok_or_else(|| {
                        PbrtError::error("built-in spectrum class ILLUMINANT missing")
                    })?,
                ];
                let tp = TextureParameterDictionary::new(&params, &float_tex, &spectrum_tex);
                let render_from_texture = render_from_world * tex.render_from_texture;
                match SpectrumImageTexture::create_variants_with_mipmap_cache(
                    &render_from_texture,
                    &tp,
                    &spectrum_types,
                    Some(&imagemap_mipmaps),
                ) {
                    Ok(textures) => {
                        for (spectrum_type, texture) in textures {
                            spectrum_tex.insert(
                                typed_spectrum_texture_name(&user_name, spectrum_type),
                                Arc::new(SpectrumTexture::ImageMap(texture)),
                            );
                        }
                    }
                    Err(e) => return Err(e),
                }
                continue;
            }
            for class in [
                SPECTRUM_CLASS_ALBEDO,
                SPECTRUM_CLASS_UNBOUNDED,
                SPECTRUM_CLASS_ILLUMINANT,
            ] {
                let spectrum_type = canonical_spectrum_type(class).ok_or_else(|| {
                    PbrtError::error(&format!("built-in spectrum class {} missing", class))
                })?;
                let tp = TextureParameterDictionary::new(&params, &float_tex, &spectrum_tex);
                let render_from_texture = render_from_world * tex.render_from_texture;
                match SpectrumTexture::create(
                    &tex.base.name,
                    &render_from_texture,
                    &tp,
                    spectrum_type,
                ) {
                    Ok(t) => {
                        spectrum_tex.insert(
                            typed_spectrum_texture_name(&user_name, spectrum_type),
                            Arc::new(t),
                        );
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        Ok((float_tex, spectrum_tex))
    }

    fn realize_media(&self, render_from_world: &Transform) -> Result<MediumMap, PbrtError> {
        let mut out: MediumMap = HashMap::new();
        for (name, m) in &self.media {
            let params = make_absolute_path(&m.base.params, &self.seen_work_dirs);
            let t = params.get_one_string("type", "");
            if t.is_empty() {
                return Err(PbrtError::error(&format!(
                    "No parameter string \"type\" found in MakeNamedMedium \"{}\".",
                    name
                )));
            }
            let render_from_medium =
                compose_with_render_from_world(&m.render_from_medium, render_from_world);
            let transform = render_from_object_to_transform(&render_from_medium);
            let medium = Medium::create(&t, &params, &transform).map_err(|e| {
                PbrtError::error(&format!(
                    "Unable to create medium \"{}\" \"{}\": {}",
                    name, t, e
                ))
            })?;
            out.insert(name.clone(), Arc::new(medium));
        }
        Ok(out)
    }

    /// Materials are built in **declaration order**, matching SceneBuilder's
    /// eager semantics. `mix` material references resolve against the
    /// `named_materials_so_far` map (only previously-defined named materials).
    ///
    /// Returns `(materials, named_materials, default_material)`. The default is
    /// applied to shapes that never saw an explicit `Material` or
    /// `NamedMaterial` (SceneBuilder initializes `current_material` to a
    /// `"diffuse"` matte at `GraphicsState::new()`; we have to mimic that here).
    #[allow(clippy::type_complexity)]
    fn realize_materials(
        &self,
        float_tex: &FloatTextureMap,
        spectrum_tex: &SpectrumTextureMap,
    ) -> Result<
        (
            Vec<Arc<Material>>,
            HashMap<String, Arc<Material>>,
            Arc<Material>,
        ),
        PbrtError,
    > {
        let inv_named_mat = invert_named_map(&self.named_materials);
        let mut materials: Vec<Arc<Material>> = Vec::with_capacity(self.materials.len());
        let mut named_materials: HashMap<String, Arc<Material>> = HashMap::new();

        for (idx, mat) in self.materials.iter().enumerate() {
            let params = make_absolute_path(&mat.base.params, &self.seen_work_dirs);
            let tp = TextureParameterDictionary::new(&params, float_tex, spectrum_tex);
            let mat_arc =
                build_material(&mat.base.name, &tp, &named_materials, &self.integrator_name)?;
            materials.push(Arc::clone(&mat_arc));
            if let Some(name) = inv_named_mat.get(&idx) {
                named_materials.insert(name.clone(), Arc::clone(&mat_arc));
            }
        }
        // Default material (SceneBuilder initializes graphics_state.current_material
        // to a `"diffuse"` matte with empty params at GraphicsState::new()).
        let empty = ParameterDictionary::new();
        let tp = TextureParameterDictionary::new(&empty, float_tex, spectrum_tex);
        let default_material = Arc::new(make_material("diffuse", &tp)?);
        Ok((materials, named_materials, default_material))
    }

    #[allow(clippy::too_many_arguments)]
    fn realize_instance_definitions(
        &self,
        float_tex: &FloatTextureMap,
        spectrum_tex: &SpectrumTextureMap,
        materials: &[Arc<Material>],
        named_materials: &HashMap<String, Arc<Material>>,
        default_material: &Arc<Material>,
        named_media: &MediumMap,
        scene_build_pool: Option<&ThreadPool>,
    ) -> Result<HashMap<String, Arc<Primitive>>, PbrtError> {
        use rayon::prelude::*;
        let build_parallel = scene_build_pool.is_some();
        let entries: Vec<(&String, &InstanceDefinitionSceneEntity)> =
            self.instance_definitions.iter().collect();
        // Instance-definition shapes live in instance-local space; the
        // per-use placement applies `renderFromWorld` in Phase 6.
        let id_transform = Transform::identity();
        let build_definition = |&(name, def): &(&String, &InstanceDefinitionSceneEntity)| {
            let mut prims: Vec<Arc<Primitive>> = Vec::new();
            let mut area_lights: Vec<Arc<Light>> = Vec::new();

            if build_parallel {
                let realize_static_shape = |shape| {
                    self.realize_shape_owned(
                        shape,
                        float_tex,
                        spectrum_tex,
                        materials,
                        named_materials,
                        default_material,
                        named_media,
                        &id_transform,
                    )
                };
                let static_results: Vec<Result<(Vec<Arc<Primitive>>, Vec<Arc<Light>>), PbrtError>> =
                    def.shapes.par_iter().map(realize_static_shape).collect();
                let realize_animated_shape = |shape| {
                    self.realize_shape_owned(
                        shape,
                        float_tex,
                        spectrum_tex,
                        materials,
                        named_materials,
                        default_material,
                        named_media,
                        &id_transform,
                    )
                };
                let animated_results: Vec<
                    Result<(Vec<Arc<Primitive>>, Vec<Arc<Light>>), PbrtError>,
                > = def
                    .animated_shapes
                    .par_iter()
                    .map(realize_animated_shape)
                    .collect();

                for result in static_results
                    .into_iter()
                    .chain(animated_results.into_iter())
                {
                    let (p, l) = match result {
                        Ok(value) => value,
                        Err(e) => return ((*name).clone(), Err(e)),
                    };
                    prims.extend(p);
                    area_lights.extend(l);
                }
            } else {
                for shape in &def.shapes {
                    if let Err(e) = self.realize_shape(
                        shape,
                        float_tex,
                        spectrum_tex,
                        materials,
                        named_materials,
                        default_material,
                        named_media,
                        &id_transform,
                        &mut prims,
                        &mut area_lights,
                    ) {
                        return ((*name).clone(), Err(e));
                    }
                }
                for shape in &def.animated_shapes {
                    if let Err(e) = self.realize_shape(
                        shape,
                        float_tex,
                        spectrum_tex,
                        materials,
                        named_materials,
                        default_material,
                        named_media,
                        &id_transform,
                        &mut prims,
                        &mut area_lights,
                    ) {
                        return ((*name).clone(), Err(e));
                    }
                }
            }

            if !area_lights.is_empty() {
                return (
                    (*name).clone(),
                    Err(PbrtError::error(&format!(
                        "Area lights inside object instance \"{}\" are not supported.",
                        name
                    ))),
                );
            }
            if prims.is_empty() {
                warn!(
                    "Skipping empty instance definition \"{}\": no valid primitives remain",
                    name
                );
                return ((*name).clone(), Ok(None));
            }
            let res = create_accelerator(&self.accelerator_name, &prims, &self.accelerator_params)
                .map(Arc::new)
                .map_err(|e| {
                    PbrtError::error(&format!(
                        "create_accelerator for instance \"{}\" failed: {}",
                        name, e.msg
                    ))
                });
            ((*name).clone(), res.map(Some))
        };

        let mut aggregates = HashMap::new();
        if build_parallel {
            let Some(scene_build_pool) = scene_build_pool else {
                return Err(PbrtError::error(
                    "parallel scene build requested without a thread pool",
                ));
            };
            let built: Vec<(String, Result<Option<Arc<Primitive>>, PbrtError>)> =
                scene_build_pool.install(|| entries.par_iter().map(build_definition).collect());
            for (name, res) in built {
                if let Some(aggregate) = res? {
                    aggregates.insert(name, aggregate);
                }
            }
        } else {
            for entry in &entries {
                let (name, res) = build_definition(entry);
                if let Some(aggregate) = res? {
                    aggregates.insert(name, aggregate);
                }
            }
        }
        Ok(aggregates)
    }

    #[allow(clippy::too_many_arguments)]
    fn realize_top_level_shapes(
        &self,
        float_tex: &FloatTextureMap,
        spectrum_tex: &SpectrumTextureMap,
        materials: &[Arc<Material>],
        named_materials: &HashMap<String, Arc<Material>>,
        default_material: &Arc<Material>,
        named_media: &MediumMap,
        render_from_world: &Transform,
        scene_build_pool: Option<&ThreadPool>,
    ) -> Result<(Vec<Arc<Primitive>>, Vec<Arc<Light>>), PbrtError> {
        use rayon::prelude::*;
        let build_parallel = scene_build_pool.is_some();
        let mut prims: Vec<Arc<Primitive>> = Vec::new();
        let mut area_lights: Vec<Arc<Light>> = Vec::new();

        if build_parallel {
            // Process static and animated shape entities in declaration order.
            // The parallel path uses par_iter, whose collect preserves order.
            let realize_static_shape = |shape| {
                self.realize_shape_owned(
                    shape,
                    float_tex,
                    spectrum_tex,
                    materials,
                    named_materials,
                    default_material,
                    named_media,
                    render_from_world,
                )
            };
            let realize_animated_shape = |shape| {
                self.realize_shape_owned(
                    shape,
                    float_tex,
                    spectrum_tex,
                    materials,
                    named_materials,
                    default_material,
                    named_media,
                    render_from_world,
                )
            };
            let (static_results, animated_results): (
                Vec<Result<(Vec<Arc<Primitive>>, Vec<Arc<Light>>), PbrtError>>,
                Vec<Result<(Vec<Arc<Primitive>>, Vec<Arc<Light>>), PbrtError>>,
            ) = {
                let Some(scene_build_pool) = scene_build_pool else {
                    return Err(PbrtError::error(
                        "parallel scene build requested without a thread pool",
                    ));
                };
                scene_build_pool.install(|| {
                    (
                        self.shapes.par_iter().map(realize_static_shape).collect(),
                        self.animated_shapes
                            .par_iter()
                            .map(realize_animated_shape)
                            .collect(),
                    )
                })
            };

            for result in static_results
                .into_iter()
                .chain(animated_results.into_iter())
            {
                let (p, l) = result?;
                prims.extend(p);
                area_lights.extend(l);
            }
        } else {
            for shape in &self.shapes {
                self.realize_shape(
                    shape,
                    float_tex,
                    spectrum_tex,
                    materials,
                    named_materials,
                    default_material,
                    named_media,
                    render_from_world,
                    &mut prims,
                    &mut area_lights,
                )?;
            }
            for shape in &self.animated_shapes {
                self.realize_shape(
                    shape,
                    float_tex,
                    spectrum_tex,
                    materials,
                    named_materials,
                    default_material,
                    named_media,
                    render_from_world,
                    &mut prims,
                    &mut area_lights,
                )?;
            }
        }

        Ok((prims, area_lights))
    }

    /// Owned-output variant of `realize_shape` for parallel collection. Each
    /// parallel task gets its own buffers.
    #[allow(clippy::too_many_arguments)]
    fn realize_shape_owned(
        &self,
        shape: &ShapeSceneEntity,
        float_tex: &FloatTextureMap,
        spectrum_tex: &SpectrumTextureMap,
        materials: &[Arc<Material>],
        named_materials: &HashMap<String, Arc<Material>>,
        default_material: &Arc<Material>,
        named_media: &MediumMap,
        render_from_world: &Transform,
    ) -> Result<(Vec<Arc<Primitive>>, Vec<Arc<Light>>), PbrtError> {
        let mut prims = Vec::new();
        let mut area_lights = Vec::new();
        self.realize_shape(
            shape,
            float_tex,
            spectrum_tex,
            materials,
            named_materials,
            default_material,
            named_media,
            render_from_world,
            &mut prims,
            &mut area_lights,
        )?;
        Ok((prims, area_lights))
    }

    #[allow(clippy::too_many_arguments)]
    fn realize_curves(
        &self,
        shape: &ShapeSceneEntity,
        float_tex: &FloatTextureMap,
        _spectrum_tex: &SpectrumTextureMap,
        materials: &[Arc<Material>],
        named_materials: &HashMap<String, Arc<Material>>,
        default_material: &Arc<Material>,
        named_media: &MediumMap,
        render_from_world: &Transform,
        out_prims: &mut Vec<Arc<Primitive>>,
        out_area_lights: &mut Vec<Arc<Light>>,
    ) -> Result<(), PbrtError> {
        let rfo = compose_with_render_from_world(&shape.render_from_object, render_from_world);
        let animated = matches!(rfo, RenderFromObject::Animated { .. });
        let object_to_world = match &rfo {
            RenderFromObject::Static(t) => *t,
            RenderFromObject::Animated { from, .. } => *from,
        };
        let shape_object_to_world = if animated {
            Transform::identity()
        } else {
            object_to_world
        };
        let shape_sets = Shape::create_curves(
            &shape_object_to_world,
            &shape_object_to_world.inverse(),
            shape.reverse_orientation,
            &shape.child_params,
            float_tex,
        )?;
        for shapes in shape_sets {
            self.emit_shape_primitives(
                shape,
                rfo.clone(),
                object_to_world,
                shapes,
                materials,
                named_materials,
                default_material,
                named_media,
                out_prims,
                out_area_lights,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn realize_shape(
        &self,
        shape: &ShapeSceneEntity,
        float_tex: &FloatTextureMap,
        spectrum_tex: &SpectrumTextureMap,
        materials: &[Arc<Material>],
        named_materials: &HashMap<String, Arc<Material>>,
        default_material: &Arc<Material>,
        named_media: &MediumMap,
        render_from_world: &Transform,
        out_prims: &mut Vec<Arc<Primitive>>,
        out_area_lights: &mut Vec<Arc<Light>>,
    ) -> Result<(), PbrtError> {
        if shape.base.name == CURVES_SHAPE_NAME && shape.child_params.is_empty() {
            return Err(PbrtError::error(
                "Shape \"curves\" is reserved for SceneBuilder's internal representation.",
            ));
        }

        if shape.base.name == CURVES_SHAPE_NAME {
            return self.realize_curves(
                shape,
                float_tex,
                spectrum_tex,
                materials,
                named_materials,
                default_material,
                named_media,
                render_from_world,
                out_prims,
                out_area_lights,
            );
        }

        // pbrt-v4 pre-composes shape transforms with `renderFromWorld`
        // before handing them to the shape ctor; for World mode this
        // collapses to identity (the helper short-circuits).
        let rfo = compose_with_render_from_world(&shape.render_from_object, render_from_world);
        let animated = matches!(rfo, RenderFromObject::Animated { .. });
        let object_to_world = match &rfo {
            RenderFromObject::Static(t) => *t,
            RenderFromObject::Animated { from, .. } => *from,
        };
        let shape_object_to_world = if animated {
            Transform::identity()
        } else {
            object_to_world
        };

        // Build the underlying Shape(s). For triangle/curve meshes, params may
        // contain a relative filename which SceneBuilder's create_shapes
        // selectively did NOT absolute-resolve. Match that exclusion list.
        let params_resolved;
        let params_ref: &ParameterDictionary =
            if shape.base.name == "trianglemesh" || shape.base.name == "curve" {
                &shape.base.params
            } else {
                params_resolved = make_absolute_path(&shape.base.params, &self.seen_work_dirs);
                &params_resolved
            };

        // For area-light-bound shapes, force the "twosided" param along the
        // path used by SceneBuilder.make_shapes.
        let mut twosided_params;
        let mut shape_params_for_create: &ParameterDictionary = params_ref;
        if let Some(al_idx) = shape.area_light_index {
            let al = &self.area_lights[al_idx];
            let two_sided = al.base.params.get_one_bool("twosided", false);
            let two_sided = params_ref.get_one_bool("twosided", two_sided);
            twosided_params = params_ref.clone();
            twosided_params.replace_one_bool("bool twosided", two_sided);
            shape_params_for_create = &twosided_params;
        }
        let created_shapes = Shape::create(
            &shape.base.name,
            &shape_object_to_world,
            &shape_object_to_world.inverse(),
            shape.reverse_orientation,
            shape_params_for_create,
            float_tex,
        );
        let shapes = match created_shapes {
            Ok(s) => s,
            Err(e) => {
                return Err(e);
            }
        };
        if shapes.is_empty() {
            return Ok(());
        }
        self.emit_shape_primitives(
            shape,
            rfo,
            object_to_world,
            shapes,
            materials,
            named_materials,
            default_material,
            named_media,
            out_prims,
            out_area_lights,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_shape_primitives(
        &self,
        shape: &ShapeSceneEntity,
        rfo: RenderFromObject,
        object_to_world: Transform,
        shapes: Vec<Shape>,
        materials: &[Arc<Material>],
        named_materials: &HashMap<String, Arc<Material>>,
        default_material: &Arc<Material>,
        named_media: &MediumMap,
        out_prims: &mut Vec<Arc<Primitive>>,
        out_area_lights: &mut Vec<Arc<Light>>,
    ) -> Result<(), PbrtError> {
        if shapes.is_empty() {
            return Ok(());
        }
        let animated = matches!(rfo, RenderFromObject::Animated { .. });

        let material = if shape.material_is_default
            && shape.material_index == usize::MAX
            && shape.material_name.is_none()
        {
            Some(Arc::clone(default_material))
        } else {
            self.resolve_shape_material(shape, materials, named_materials)?
        };
        let medium_interface = build_medium_interface(&shape.medium_interface, named_media);

        if animated {
            // Animated shapes: no area light (warned at parse time).
            let mut piece_prims: Vec<Arc<Primitive>> = Vec::with_capacity(shapes.len());
            for s in shapes {
                let prim = Arc::new(Primitive::new_geometric(
                    s,
                    &material,
                    &None,
                    &medium_interface,
                ));
                piece_prims.push(prim);
            }
            let animated_transform = match rfo {
                RenderFromObject::Animated {
                    from,
                    to,
                    start_time,
                    end_time,
                } => match AnimatedTransform::new(&from, start_time, &to, end_time) {
                    Some(transform) => transform,
                    None => {
                        error!("Skipping shape: animated transform decomposition failed");
                        return Ok(());
                    }
                },
                _ => unreachable!(),
            };
            let final_prim = if piece_prims.len() > 1 {
                match self.create_shape_accelerator(&piece_prims) {
                    Ok(accel) => accel,
                    Err(e) => {
                        return Err(e);
                    }
                }
            } else {
                match piece_prims.into_iter().next() {
                    Some(prim) => prim,
                    None => {
                        return Err(PbrtError::error(
                            "animated shape produced no primitives to emit",
                        ));
                    }
                }
            };
            let prim = Arc::new(Primitive::Transformed(TransformedPrimitive::new(
                &final_prim,
                &animated_transform,
            )));
            out_prims.push(prim);
            return Ok(());
        }

        // Static shapes emit leaf primitives directly. The scene-level
        // accelerator is responsible for grouping them.
        let mut piece_prims = Vec::with_capacity(shapes.len());
        for s in shapes {
            if let Some(al_idx) = shape.area_light_index {
                let al = &self.area_lights[al_idx];
                let al_params = make_absolute_path(&al.base.params, &self.seen_work_dirs);
                let al_mi = build_medium_interface(&al.medium_interface, named_media);
                let shared_shape = Arc::new(s);
                let area_light = match Light::create_area(
                    &al.base.name,
                    &object_to_world,
                    &al_mi,
                    &al_params,
                    &shared_shape,
                ) {
                    Ok(l) => {
                        out_area_lights.push(Arc::clone(&l));
                        l
                    }
                    Err(e) => {
                        return Err(e);
                    }
                };
                let prim = Arc::new(Primitive::new_geometric_shared(
                    shared_shape,
                    &material,
                    &Some(area_light),
                    &medium_interface,
                ));
                piece_prims.push(prim);
            } else {
                let prim = Arc::new(Primitive::new_geometric(
                    s,
                    &material,
                    &None,
                    &medium_interface,
                ));
                piece_prims.push(prim);
            }
        }
        out_prims.extend(piece_prims);
        Ok(())
    }

    fn create_shape_accelerator(
        &self,
        prims: &[Arc<Primitive>],
    ) -> Result<Arc<Primitive>, PbrtError> {
        create_accelerator(&self.accelerator_name, prims, &self.accelerator_params)
            .map(Arc::new)
            .map_err(|e| PbrtError::error(&format!("create_accelerator failed: {}", e.msg)))
    }

    fn resolve_shape_material(
        &self,
        shape: &ShapeSceneEntity,
        materials: &[Arc<Material>],
        named_materials: &HashMap<String, Arc<Material>>,
    ) -> Result<Option<Arc<Material>>, PbrtError> {
        if let Some(name) = shape.material_name.as_ref() {
            return Ok(named_materials.get(name).cloned());
        }
        if shape.material_index != usize::MAX {
            return Ok(materials.get(shape.material_index).cloned());
        }
        Ok(None) // interface (no surface material)
    }

    /// Rebuild the Material when shape params override material params,
    /// mirroring `SceneBuilder::get_material_for_shape`.
    fn realize_instance_use(
        &self,
        inst: &InstanceSceneEntity,
        instance_aggregates: &HashMap<String, Arc<Primitive>>,
        render_from_world: &Transform,
    ) -> Option<Arc<Primitive>> {
        let aggregate = instance_aggregates.get(&inst.name).cloned()?;
        let rfo = compose_with_render_from_world(&inst.render_from_instance, render_from_world);
        let prim = match rfo {
            RenderFromObject::Static(t) => {
                if t == Transform::identity() {
                    aggregate
                } else {
                    // SceneBuilder treats single-static-transform instance use as
                    // a regular TransformedPrimitive (no animated wrapper).
                    let Some(at) = AnimatedTransform::new(
                        &t,
                        self.transform_start_time,
                        &t,
                        self.transform_end_time,
                    ) else {
                        error!(
                            "Skipping instance \"{}\": transform decomposition failed",
                            inst.name
                        );
                        return None;
                    };
                    Arc::new(Primitive::Transformed(TransformedPrimitive::new(
                        &aggregate, &at,
                    )))
                }
            }
            RenderFromObject::Animated {
                from,
                to,
                start_time,
                end_time,
            } => {
                let Some(at) = AnimatedTransform::new(&from, start_time, &to, end_time) else {
                    error!(
                        "Skipping instance \"{}\": animated transform decomposition failed",
                        inst.name
                    );
                    return None;
                };
                Arc::new(Primitive::Transformed(TransformedPrimitive::new(
                    &aggregate, &at,
                )))
            }
        };
        Some(prim)
    }

    fn realize_lights(
        &self,
        named_media: &MediumMap,
        area_lights: &mut Vec<Arc<Light>>,
        render_from_world: &Transform,
    ) -> Result<Vec<Arc<Light>>, PbrtError> {
        let mut lights: Vec<Arc<Light>> = Vec::new();
        lights.append(area_lights);
        for l in &self.lights {
            let params = make_absolute_path(&l.base.base.params, &self.seen_work_dirs);
            let rfo = compose_with_render_from_world(&l.base.render_from_object, render_from_world);
            let transform = render_from_object_to_transform(&rfo);
            // Light entities have a single medium name; use it on both sides.
            let mi = if !l.medium.is_empty() {
                build_medium_interface(
                    &MediumInterfaceNames::new(l.medium.clone(), l.medium.clone()),
                    named_media,
                )
            } else {
                MediumInterface::new()
            };
            match Light::create(
                &l.base.base.name,
                &transform,
                &mi,
                &params,
                render_from_world,
            ) {
                Ok(light) => lights.push(light),
                Err(e) => return Err(e),
            }
        }
        Ok(lights)
    }

    fn realize_filter(&self) -> Result<Filter, PbrtError> {
        Filter::create(&self.filter_name, &self.filter_params)
    }

    fn realize_film(&self, filter: &Filter) -> Result<Arc<RwLock<Film>>, PbrtError> {
        Film::create(&self.film_name, &self.film_params, filter)
    }

    fn realize_camera(
        &self,
        film: &Arc<RwLock<Film>>,
        named_media: &MediumMap,
        camera_transform: &CameraTransform,
    ) -> Result<Camera, PbrtError> {
        // pbrt-v4 hands the camera ctor `CameraTransform::RenderFromCamera`
        // (cameras.cpp:55) rather than `worldFromCamera`, so the camera
        // emits rays directly in render space. For `World` mode this
        // equals the established `cameraToWorld` (since renderFromWorld is
        // identity), so existing camera code carries over unchanged.
        let animated = camera_transform.render_from_camera().clone();
        // Camera's medium is the outside of the current MediumInterface at
        // Camera time. We didn't capture per-Camera medium interface; use the
        // global current_outside_medium recorded at parse time (graphics_state
        // is preserved through WorldBegin).
        let cur_outside = &self
            .graphics_states
            .last()
            .map(|gs| gs.current_outside_medium.as_str())
            .unwrap_or("");
        let medium = if !cur_outside.is_empty() {
            named_media.get(*cur_outside).cloned()
        } else {
            None
        };
        let params = make_absolute_path(&self.camera_params, &self.seen_work_dirs);
        Camera::create(&self.camera_name, &params, &animated, film, &medium)
    }

    fn realize_sampler(&self, film: &Arc<RwLock<Film>>) -> Result<Arc<RwLock<Sampler>>, PbrtError> {
        let full_resolution = match film.read() {
            Ok(film) => film.full_resolution(),
            Err(_) => {
                return Err(PbrtError::error(
                    "failed to read film while creating sampler",
                ));
            }
        };
        let sampler = Sampler::create(&self.sampler_name, &self.sampler_params, full_resolution)?;
        Ok(Arc::new(RwLock::new(sampler)))
    }
}

// ===== Free helpers =========================================================

fn invert_named_map(map: &HashMap<String, usize>) -> HashMap<usize, String> {
    map.iter().map(|(n, i)| (*i, n.clone())).collect()
}

fn render_from_object_to_transform(rfo: &RenderFromObject) -> Transform {
    match rfo {
        RenderFromObject::Static(t) => *t,
        RenderFromObject::Animated { from, .. } => *from,
    }
}

/// pbrt-v4 maps every entity's object-from-world transform into render
/// space at scene-build time by pre-multiplying with
/// `renderFromWorld = Inverse(worldFromRender)`. This helper applies
/// that to a `RenderFromObject` recorded in world space.
///
/// The naming is sticky: parser-side, `RenderFromObject` (and the
/// matching CTM in `parse_target_impl`) is captured in **world**
/// coordinates — it became "render space" because the renderer uses
/// `RenderingCoordinateSystem::World`. With CameraWorld active we
/// need this extra composition step.
fn compose_with_render_from_world(
    rfo: &RenderFromObject,
    render_from_world: &Transform,
) -> RenderFromObject {
    if render_from_world.is_identity() {
        return rfo.clone();
    }
    match rfo {
        RenderFromObject::Static(t) => RenderFromObject::Static(*render_from_world * *t),
        RenderFromObject::Animated {
            from,
            to,
            start_time,
            end_time,
        } => RenderFromObject::Animated {
            from: *render_from_world * *from,
            to: *render_from_world * *to,
            start_time: *start_time,
            end_time: *end_time,
        },
    }
}

fn build_medium_interface(
    names: &MediumInterfaceNames,
    named_media: &MediumMap,
) -> MediumInterface {
    let mut mi = MediumInterface::new();
    if !names.inside_medium.is_empty() {
        if let Some(m) = named_media.get(&names.inside_medium) {
            mi.set_inside(m);
        }
    }
    if !names.outside_medium.is_empty() {
        if let Some(m) = named_media.get(&names.outside_medium) {
            mi.set_outside(m);
        }
    }
    mi
}

/// Equivalent to `SceneBuilder::make_material`. "mix" is the only
/// special case that consults `named_materials`; everything else
/// dispatches directly through `Material::create`.
fn build_material(
    name: &str,
    mp: &TextureParameterDictionary,
    named_materials: &HashMap<String, Arc<Material>>,
    integrator_name: &str,
) -> Result<Arc<Material>, PbrtError> {
    if name == "subsurface" {
        if integrator_name != "path" && integrator_name != "volpath" {
            warn!(
                "Subsurface scattering material \"{}\" used, but \"{}\" integrator doesn't support subsurface scattering. Use \"path\" or \"volpath\".",
                name, integrator_name
            );
        }
    }
    if name == "mix" {
        let (m1, m2) = resolve_mix_material_names(mp);
        if m1.is_empty() || m2.is_empty() {
            return Err(PbrtError::error(
                "Mix material is missing named-material references.",
            ));
        }
        let mat1 = named_materials.get(&m1).cloned().ok_or_else(|| {
            PbrtError::error(&format!(
                "Mix material references unknown material \"{}\".",
                m1
            ))
        })?;
        let mat2 = named_materials.get(&m2).cloned().ok_or_else(|| {
            PbrtError::error(&format!(
                "Mix material references unknown material \"{}\".",
                m2
            ))
        })?;
        match MixMaterial::create(mp, &mat1, &mat2) {
            Ok(m) => return Ok(Arc::new(Material::Mix(m))),
            Err(e) => return Err(e),
        }
    }
    Ok(Arc::new(make_material(name, mp)?))
}

fn make_material(name: &str, mp: &TextureParameterDictionary) -> Result<Material, PbrtError> {
    match Material::create(name, mp) {
        Ok(m) => Ok(m),
        Err(e) => Err(e),
    }
}

fn resolve_mix_material_names(mp: &TextureParameterDictionary) -> (String, String) {
    if let Some(names) = mp.params.get_strings_ref("materials") {
        if names.len() >= 2 {
            return (names[0].clone(), names[1].clone());
        }
    }
    (
        mp.get_one_string("namedmaterial1", ""),
        mp.get_one_string("namedmaterial2", ""),
    )
}

/// True if any of `params`' keys has the `"texture"` type prefix (e.g.,
/// `"texture reflectance"`). Used by `realize_textures` to partition textures
/// into leaf (no chain refs) vs chain (depends on other textures) for
/// parallel-vs-sequential build.
fn has_texture_ref(params: &ParameterDictionary) -> bool {
    params.get_keys().iter().any(|k| {
        let parts: Vec<&str> = k.split_ascii_whitespace().collect();
        parts.len() >= 2 && parts[0] == "texture"
    })
}
