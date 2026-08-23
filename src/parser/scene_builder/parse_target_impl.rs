//! `ParseTarget` implementation for [`SceneBuilder`].
//!
//! Each callback mirrors v4 `BasicSceneBuilder` and pushes the matching
//! `*SceneEntity` onto the builder. It explicitly does **not** realise
//! anything (no PLY reads, Material construction, or BVH building);
//! that work belongs to `SceneBuilder::build()`.
//!
//! Design notes: see `docs/scene_loader_refactor_ja.md`.

use super::scene_entity::{
    AreaLightSceneEntity, InstanceDefinitionSceneEntity, InstanceSceneEntity, LightSceneEntity,
    MaterialSceneEntity, MediumSceneEntity, SceneEntity, ShapeSceneEntity, TextureKind,
    TextureSceneEntity, TransformedSceneEntity,
};
use super::state::{ApiState, PushKind};
use super::{SceneBuilder, CURVES_SHAPE_NAME};
use crate::options::PbrtOptions;
use crate::paramdict::ParameterDictionary;
use crate::parser::parsed_parameter::into_parameter_dictionary;
use crate::parser::{parse_file, parse_target::ParseTarget, ParsedParameterVector};
use crate::util::base::Float;
use crate::util::spectrum::rgb_to_spectrum::lookup_color_space_by_name;
use crate::util::transform::transform_set::{
    ALL_TRANSFORM_BITS, END_TRANSFORM_BITS, START_TRANSFORM_BITS,
};
use crate::util::transform::{Matrix4x4, Transform};

use log::{error, info, warn};
use std::path::Path;

impl SceneBuilder {
    fn append_shape(shapes: &mut Vec<ShapeSceneEntity>, mut shape: ShapeSceneEntity) {
        if let Some(parent) = shapes.last_mut() {
            if Self::can_group_shapes(parent, &shape) {
                parent.child_params.push(shape.base.params);
                return;
            }
        }

        if shape.base.name == "curve" && shape.area_light_index.is_none() {
            shape.base.name = CURVES_SHAPE_NAME.to_owned();
            shape
                .child_params
                .push(std::mem::take(&mut shape.base.params));
        }
        shapes.push(shape);
    }

    fn can_group_shapes(parent: &ShapeSceneEntity, child: &ShapeSceneEntity) -> bool {
        if parent.base.name != CURVES_SHAPE_NAME
            || child.base.name != "curve"
            || parent.area_light_index.is_some()
            || child.area_light_index.is_some()
        {
            return false;
        }

        if parent.render_from_object != child.render_from_object
            || parent.reverse_orientation != child.reverse_orientation
            || parent.material_index != child.material_index
            || parent.material_name != child.material_name
            || parent.material_is_default != child.material_is_default
            || parent.medium_interface != child.medium_interface
            || parent.instance_name != child.instance_name
        {
            return false;
        }

        let Some(parent_params) = parent.child_params.last() else {
            return false;
        };
        Self::curve_params_share_group(parent_params, &child.base.params)
    }

    fn curve_params_share_group(a: &ParameterDictionary, b: &ParameterDictionary) -> bool {
        a.get_one_string("type", "flat") == b.get_one_string("type", "flat")
            && Self::same_curve_parameter(a, b, "alpha")
            && Self::same_curve_parameter(a, b, "shadowalpha")
    }

    fn same_curve_parameter(a: &ParameterDictionary, b: &ParameterDictionary, name: &str) -> bool {
        let a_textures = a.get_textures_ref(name);
        let b_textures = b.get_textures_ref(name);
        match (a_textures.as_ref(), b_textures.as_ref()) {
            (Some(a), Some(b)) => return a.as_slice() == b.as_slice(),
            (Some(_), None) | (None, Some(_)) => return false,
            (None, None) => {}
        }

        let a_values = a.get_floats_ref(name);
        let b_values = b.get_floats_ref(name);
        match (a_values.as_ref(), b_values.as_ref()) {
            (Some(a), Some(b)) => a.as_slice() == b.as_slice(),
            (Some(_), None) | (None, Some(_)) => false,
            (None, None) => true,
        }
    }

    fn verify_options(&self, fun: &str) {
        if self.api_state != ApiState::OptionsBlock {
            error!(
                "Options cannot be set inside world block;\n\"{}\" not allowed. Ignoring.",
                fun
            );
        }
    }

    fn verify_world(&self, fun: &str) {
        if self.api_state != ApiState::WorldBlock {
            error!(
                "Scene description must be inside world block;\n\"{}\" not allowed. Ignoring.",
                fun
            );
        }
    }

    /// Clone `params`, fold in the pending per-target `Attribute`
    /// parameters (`params` taking precedence), and tag the result with
    /// the current graphics-state color space so RGB values convert to
    /// spectra in the right space. Mirrors pbrt-v4, where a directive's
    /// own parameters win over inherited `Attribute` parameters and every
    /// dictionary inherits `graphicsState.colorSpace`.
    fn params_with_attributes(
        &self,
        params: ParameterDictionary,
        attrs: &ParameterDictionary,
    ) -> ParameterDictionary {
        let mut p = params;
        p.merge_missing(attrs);
        if !p.has_parameter("displacement") {
            p.rename_parameter("bumpmap", "displacement");
        }
        if let Some(state) = self.graphics_states.last() {
            p.set_color_space(state.color_space);
        }
        p
    }
}

impl ParseTarget for SceneBuilder {
    fn cleanup(&mut self) {
        *self = Self::new();
    }

    // ===== Transform manipulation ===========================================
    fn identity(&mut self) {
        let bits = self.top_transform_bits();
        self.top_transform_mut()
            .set_transform(&Transform::identity(), bits);
    }

    fn translate(&mut self, dx: Float, dy: Float, dz: Float) {
        let t = Transform::translate(dx, dy, dz);
        let bits = self.top_transform_bits();
        self.top_transform_mut().mul_transform(&t, bits);
    }

    fn rotate(&mut self, angle: Float, ax: Float, ay: Float, az: Float) {
        let t = Transform::rotate(angle, ax, ay, az);
        let bits = self.top_transform_bits();
        self.top_transform_mut().mul_transform(&t, bits);
    }

    fn scale(&mut self, sx: Float, sy: Float, sz: Float) {
        let t = Transform::scale(sx, sy, sz);
        let bits = self.top_transform_bits();
        self.top_transform_mut().mul_transform(&t, bits);
    }

    fn look_at(
        &mut self,
        ex: Float,
        ey: Float,
        ez: Float,
        lx: Float,
        ly: Float,
        lz: Float,
        ux: Float,
        uy: Float,
        uz: Float,
    ) {
        let t = Transform::look_at(ex, ey, ez, lx, ly, lz, ux, uy, uz);
        let bits = self.top_transform_bits();
        self.top_transform_mut().mul_transform(&t, bits);
    }

    fn concat_transform(&mut self, t: &[Float]) {
        #[rustfmt::skip]
        let m = Matrix4x4::from([
            t[0], t[4], t[8],  t[12],
            t[1], t[5], t[9],  t[13],
            t[2], t[6], t[10], t[14],
            t[3], t[7], t[11], t[15],
        ]);
        if let Some(im) = m.inverse() {
            let tr = Transform::from((m, im));
            let bits = self.top_transform_bits();
            self.top_transform_mut().mul_transform(&tr, bits);
        } else {
            error!("Singular matrix in MatrixInvert");
        }
    }

    fn transform(&mut self, t: &[Float]) {
        #[rustfmt::skip]
        let m = Matrix4x4::from([
            t[0], t[4], t[8],  t[12],
            t[1], t[5], t[9],  t[13],
            t[2], t[6], t[10], t[14],
            t[3], t[7], t[11], t[15],
        ]);
        if let Some(im) = m.inverse() {
            let tr = Transform::from((m, im));
            let bits = self.top_transform_bits();
            self.top_transform_mut().set_transform(&tr, bits);
        } else {
            error!("Singular matrix in MatrixInvert");
        }
    }

    fn color_space(&mut self, name: &str) {
        match lookup_color_space_by_name(name) {
            Some(cs) => {
                if let Some(state) = self.graphics_states.last_mut() {
                    state.color_space = cs;
                }
            }
            None => error!("Color space \"{}\" unknown. Ignoring.", name),
        }
    }

    fn option(&mut self, name: &str, value: &str) {
        self.verify_options("Option");
        if let Err(error) = PbrtOptions::apply_option(name, value) {
            self.option_errors.push(error);
        }
    }

    fn coordinate_system(&mut self, name: &str) {
        let top = *self.top_transform();
        self.named_coordinate_systems
            .insert(String::from(name), top);
    }

    fn coord_sys_transform(&mut self, name: &str) {
        if let Some(t) = self.named_coordinate_systems.get(name).copied() {
            let bits = self.top_transform_bits();
            self.top_transform_mut().set(&t, bits);
        } else {
            warn!("Couldn't find named coordinate system \"{}\"", name);
        }
    }

    fn active_transform_all(&mut self) {
        self.set_top_transform_bits(ALL_TRANSFORM_BITS);
    }

    fn active_transform_end_time(&mut self) {
        self.set_top_transform_bits(END_TRANSFORM_BITS);
    }

    fn active_transform_start_time(&mut self) {
        self.set_top_transform_bits(START_TRANSFORM_BITS);
    }

    fn transform_times(&mut self, start: Float, end: Float) {
        self.transform_start_time = start;
        self.transform_end_time = end;
    }

    // ===== Options-block setters =============================================
    fn pixel_filter(&mut self, name: &str, params: ParsedParameterVector) {
        self.verify_options("PixelFilter");
        self.filter_name = name.to_string();
        self.filter_params = into_parameter_dictionary(params);
    }

    fn film(&mut self, name: &str, params: ParsedParameterVector) {
        self.verify_options("Film");
        self.film_name = name.to_string();
        self.film_params = into_parameter_dictionary(params);
    }

    fn sampler(&mut self, name: &str, params: ParsedParameterVector) {
        self.verify_options("Sampler");
        self.sampler_name = name.to_string();
        self.sampler_params = into_parameter_dictionary(params);
    }

    fn accelerator(&mut self, name: &str, params: ParsedParameterVector) {
        self.verify_options("Accelerator");
        self.accelerator_name = name.to_string();
        self.accelerator_params = into_parameter_dictionary(params);
    }

    fn integrator(&mut self, name: &str, params: ParsedParameterVector) {
        self.verify_options("Integrator");
        self.integrator_name = name.to_string();
        self.integrator_params = into_parameter_dictionary(params);
    }

    fn camera(&mut self, name: &str, params: ParsedParameterVector) {
        self.verify_options("Camera");
        self.camera_name = name.to_string();
        self.camera_params = into_parameter_dictionary(params);
        // pbrt's `LookAt` accumulates `cameraFromWorld` into the CTM, so
        // the active transform at the `Camera` directive is actually the
        // world-to-camera direction; `realize_camera` inverts it before
        // building the AnimatedTransform.
        self.camera_to_world = *self.top_transform();
        // pbrt-v4 `BasicSceneBuilder::Camera` (scene.cpp:147) stores
        // `Inverse(cameraFromWorld)` in `namedCoordinateSystems["camera"]`,
        // so that a subsequent `CoordSysTransform "camera"` restores the
        // camera-to-world transform expected by, e.g., a distant light
        // declared in camera space.
        self.named_coordinate_systems
            .insert(String::from("camera"), self.camera_to_world.inverse());
    }

    fn make_named_medium(&mut self, name: &str, params: ParsedParameterVector) {
        let params = into_parameter_dictionary(params);
        if self.media.contains_key(name) {
            warn!("Named medium \"{}\" redefined.", name);
        }
        let params =
            self.params_with_attributes(params, &self.top_graphics_state().medium_attributes);
        let entity = MediumSceneEntity {
            base: SceneEntity::new(name, params, self.current_file_loc()),
            render_from_medium: self.render_from_object(),
        };
        self.media.insert(name.to_string(), entity);
    }

    fn medium_interface(&mut self, inside_name: &str, outside_name: &str) {
        let gs = self.top_graphics_state_mut();
        gs.current_inside_medium = inside_name.to_string();
        gs.current_outside_medium = outside_name.to_string();
    }

    // ===== Block management =================================================
    fn world_begin(&mut self) {
        self.verify_options("WorldBegin");
        self.api_state = ApiState::WorldBlock;
        // Reset the active transform to identity. Anything set inside
        // the Options block (LookAt etc.) is already captured in
        // CameraToWorld.
        let bits = ALL_TRANSFORM_BITS;
        self.top_transform_mut()
            .set_transform(&Transform::identity(), bits);
        self.set_top_transform_bits(ALL_TRANSFORM_BITS);
        self.named_coordinate_systems
            .insert(String::from("world"), *self.top_transform());
    }

    fn attribute(&mut self, target: &str, params: ParsedParameterVector) {
        let params = into_parameter_dictionary(params);
        self.verify_world("Attribute");
        let gs = self.top_graphics_state_mut();
        let dict = match target {
            "shape" => &mut gs.shape_attributes,
            "light" | "lightsource" => &mut gs.light_attributes,
            "material" => &mut gs.material_attributes,
            "medium" => &mut gs.medium_attributes,
            "texture" => &mut gs.texture_attributes,
            _ => {
                error!("Unknown Attribute target \"{}\". Ignoring.", target);
                return;
            }
        };
        // Attribute is additive: later declarations override earlier ones
        // for the same parameter, so merge with `params` taking precedence.
        let mut merged = params.clone();
        merged.merge_missing(dict);
        *dict = merged;
    }

    fn attribute_begin(&mut self) {
        self.verify_world("AttributeBegin");
        self.push_graphics_state();
        self.push_transform();
        self.push_stack.push(PushKind::Attribute);
    }

    fn attribute_end(&mut self) {
        self.verify_world("AttributeEnd");
        match self.push_stack.pop() {
            Some(PushKind::Attribute) => {
                self.pop_transform();
                self.pop_graphics_state();
            }
            Some(other) => {
                error!("AttributeEnd mismatched (popped {:?}); ignoring.", other);
            }
            None => {
                error!("AttributeEnd with empty push stack; ignoring.");
            }
        }
    }

    fn transform_begin(&mut self) {
        self.verify_world("TransformBegin");
        self.push_transform();
        self.push_stack.push(PushKind::Transform);
    }

    fn transform_end(&mut self) {
        self.verify_world("TransformEnd");
        match self.push_stack.pop() {
            Some(PushKind::Transform) => {
                self.pop_transform();
            }
            Some(other) => {
                error!("TransformEnd mismatched (popped {:?}); ignoring.", other);
            }
            None => {
                error!("TransformEnd with empty push stack; ignoring.");
            }
        }
    }

    // ===== Texture / Material / Light / Shape ===============================
    fn texture(
        &mut self,
        name: &str,
        type_name: &str,
        tex_name: &str,
        params: ParsedParameterVector,
    ) {
        let params = into_parameter_dictionary(params);
        self.verify_world("Texture");
        let params =
            self.params_with_attributes(params, &self.top_graphics_state().texture_attributes);
        let entity_base = SceneEntity::new(tex_name, params, self.current_file_loc());
        let render_from_texture = self.top_transform().to_transform();
        match type_name {
            "float" => {
                let idx = self.float_textures.len();
                self.float_textures.push(TextureSceneEntity {
                    base: entity_base,
                    texture_kind: TextureKind::Float,
                    render_from_texture,
                });
                self.named_float_textures.insert(name.to_string(), idx);
            }
            "color" | "rgb" | "spectrum" => {
                let idx = self.spectrum_textures.len();
                self.spectrum_textures.push(TextureSceneEntity {
                    base: entity_base,
                    texture_kind: TextureKind::Spectrum,
                    render_from_texture,
                });
                self.named_spectrum_textures.insert(name.to_string(), idx);
            }
            _ => {
                error!(
                    "Unknown texture type \"{}\" for texture \"{}\".",
                    type_name, name
                );
            }
        }
    }

    fn material(&mut self, name: &str, params: ParsedParameterVector) {
        let params = into_parameter_dictionary(params);
        self.verify_world("Material");
        if name.is_empty() || name == "interface" {
            // No surface material (used for volume boundaries).
            let gs = self.top_graphics_state_mut();
            gs.current_material_index = None;
            gs.current_material_name = None;
            gs.current_material_is_default = false;
        } else {
            let params =
                self.params_with_attributes(params, &self.top_graphics_state().material_attributes);
            let idx = self.materials.len();
            self.materials.push(MaterialSceneEntity {
                base: SceneEntity::new(name, params, self.current_file_loc()),
            });
            let gs = self.top_graphics_state_mut();
            gs.current_material_index = Some(idx);
            gs.current_material_name = None;
            gs.current_material_is_default = false;
        }
    }

    fn make_named_material(&mut self, name: &str, params: ParsedParameterVector) {
        let params = into_parameter_dictionary(params);
        let type_name = params.get_one_string("type", "");
        if type_name.is_empty() {
            error!("No parameter string \"type\" found in MakeNamedMaterial");
            return;
        }
        if self.named_materials.contains_key(name) {
            warn!("Named material \"{}\" redefined.", name);
        }
        let params =
            self.params_with_attributes(params, &self.top_graphics_state().material_attributes);
        let idx = self.materials.len();
        self.materials.push(MaterialSceneEntity {
            base: SceneEntity::new(&type_name, params, self.current_file_loc()),
        });
        self.named_materials.insert(name.to_string(), idx);
    }

    fn named_material(&mut self, name: &str) {
        self.verify_world("NamedMaterial");
        let gs = self.top_graphics_state_mut();
        gs.current_material_index = None;
        gs.current_material_name = Some(name.to_string());
        gs.current_material_is_default = false;
    }

    fn light_source(&mut self, name: &str, params: ParsedParameterVector) {
        let params = into_parameter_dictionary(params);
        self.verify_world("LightSource");
        let render_from_object = self.render_from_object();
        let gs = self.top_graphics_state();
        let medium = if !gs.current_inside_medium.is_empty() {
            gs.current_inside_medium.clone()
        } else {
            gs.current_outside_medium.clone()
        };
        let params =
            self.params_with_attributes(params, &self.top_graphics_state().light_attributes);
        self.lights.push(LightSceneEntity {
            base: TransformedSceneEntity {
                base: SceneEntity::new(name, params, self.current_file_loc()),
                render_from_object,
            },
            medium,
        });
    }

    fn area_light_source(&mut self, name: &str, params: ParsedParameterVector) {
        let params = into_parameter_dictionary(params);
        self.verify_world("AreaLightSource");
        // Bound to the next Shape; the actual entity push happens in
        // `shape()`.
        let params =
            self.params_with_attributes(params, &self.top_graphics_state().light_attributes);
        let gs = self.top_graphics_state_mut();
        gs.area_light_name = name.to_string();
        gs.area_light_params = params;
    }

    fn shape(&mut self, name: &str, params: ParsedParameterVector) {
        let params = into_parameter_dictionary(params);
        self.verify_world("Shape");

        let render_from_object = self.render_from_object();
        let animated = matches!(
            render_from_object,
            super::scene_entity::RenderFromObject::Animated { .. }
        );

        let (area_light_index, area_light_name_set) = {
            let gs = self.top_graphics_state();
            if !gs.area_light_name.is_empty() {
                if animated {
                    warn!("Ignoring set area light when creating animated shape");
                    (None, false)
                } else {
                    let idx = self.area_lights.len();
                    self.area_lights.push(AreaLightSceneEntity {
                        base: SceneEntity::new(
                            &gs.area_light_name,
                            gs.area_light_params.clone(),
                            self.current_file_loc(),
                        ),
                        render_from_light: render_from_object.clone(),
                        medium_interface: self.current_medium_interface(),
                    });
                    (Some(idx), true)
                }
            } else {
                (None, false)
            }
        };
        let _ = area_light_name_set;

        let params =
            self.params_with_attributes(params, &self.top_graphics_state().shape_attributes);
        let gs = self.top_graphics_state();
        let entity = ShapeSceneEntity {
            base: SceneEntity::new(name, params, self.current_file_loc()),
            child_params: Vec::new(),
            render_from_object: render_from_object.clone(),
            reverse_orientation: gs.reverse_orientation,
            material_index: gs.current_material_index.unwrap_or(usize::MAX),
            material_name: gs.current_material_name.clone(),
            material_is_default: gs.current_material_is_default,
            area_light_index,
            medium_interface: self.current_medium_interface(),
            instance_name: self.current_instance_name.clone(),
        };

        if let Some(instance_name) = self.current_instance_name.clone() {
            let loc = self.current_file_loc();
            let def = self
                .instance_definitions
                .entry(instance_name.clone())
                .or_insert_with(|| InstanceDefinitionSceneEntity {
                    name: instance_name,
                    loc,
                    shapes: Vec::new(),
                    animated_shapes: Vec::new(),
                });
            if animated {
                Self::append_shape(&mut def.animated_shapes, entity);
            } else {
                Self::append_shape(&mut def.shapes, entity);
            }
        } else if animated {
            Self::append_shape(&mut self.animated_shapes, entity);
        } else {
            Self::append_shape(&mut self.shapes, entity);
        }
    }

    fn reverse_orientation(&mut self) {
        self.verify_world("ReverseOrientation");
        let gs = self.top_graphics_state_mut();
        gs.reverse_orientation = !gs.reverse_orientation;
    }

    fn object_begin(&mut self, name: &str) {
        self.verify_world("ObjectBegin");
        if self.current_instance_name.is_some() {
            error!("ObjectBegin within ObjectBegin scope; ignoring.");
            return;
        }
        // Also push graphics state + transform (equivalent to
        // `AttributeBegin`).
        self.push_graphics_state();
        self.push_transform();
        self.push_stack.push(PushKind::Object);

        if self.instance_definitions.contains_key(name) {
            warn!("ObjectBegin trying to redefine instance \"{}\".", name);
        } else {
            self.instance_definitions.insert(
                name.to_string(),
                InstanceDefinitionSceneEntity {
                    name: name.to_string(),
                    loc: self.current_file_loc(),
                    shapes: Vec::new(),
                    animated_shapes: Vec::new(),
                },
            );
        }
        self.current_instance_name = Some(name.to_string());
    }

    fn object_end(&mut self) {
        self.verify_world("ObjectEnd");
        if self.current_instance_name.is_none() {
            error!("ObjectEnd without matching ObjectBegin; ignoring.");
            return;
        }
        self.current_instance_name = None;
        match self.push_stack.pop() {
            Some(PushKind::Object) => {
                self.pop_transform();
                self.pop_graphics_state();
            }
            Some(other) => {
                error!("ObjectEnd mismatched (popped {:?}); ignoring.", other);
            }
            None => {
                error!("ObjectEnd with empty push stack; ignoring.");
            }
        }
    }

    fn object_instance(&mut self, name: &str) {
        self.verify_world("ObjectInstance");
        if self.current_instance_name.is_some() {
            // Match pbrt-v4: ignore `ObjectInstance` inside an
            // `ObjectBegin` scope.
            return;
        }
        self.instance_uses.push(InstanceSceneEntity {
            name: name.to_string(),
            loc: self.current_file_loc(),
            render_from_instance: self.render_from_object(),
        });
    }

    fn world_end(&mut self) {
        self.verify_world("WorldEnd");
        info!(
            "SceneBuilder parse complete: \
             shapes={} animated_shapes={} materials={} named_materials={} \
             lights={} area_lights={} media={} \
             float_textures={} spectrum_textures={} \
             instance_defs={} instance_uses={}",
            self.shapes.len(),
            self.animated_shapes.len(),
            self.materials.len(),
            self.named_materials.len(),
            self.lights.len(),
            self.area_lights.len(),
            self.media.len(),
            self.float_textures.len(),
            self.spectrum_textures.len(),
            self.instance_definitions.len(),
            self.instance_uses.len(),
        );
    }

    fn parse_file(&mut self, _file_name: &str) {}
    fn parse_string(&mut self, _s: &str) {}

    fn work_dir_begin(&mut self, path: &str) {
        self.work_dirs.push(path.to_string());
        if !self.seen_work_dirs.iter().any(|d| d == path) {
            self.seen_work_dirs.push(path.to_string());
        }
    }

    fn work_dir_end(&mut self) {
        self.work_dirs.pop();
    }

    fn import(&mut self, filename: &str, _params: ParsedParameterVector) {
        let path = Path::new(filename);
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.work_dirs
                .iter()
                .rev()
                .map(Path::new)
                .map(|dir| dir.join(path))
                .find(|candidate| candidate.exists())
                .unwrap_or_else(|| path.to_path_buf())
        };
        let resolved = resolved.to_string_lossy().into_owned();
        if let Err(err) = parse_file(&resolved, self) {
            self.import_errors.push(format!(
                "Import \"{}\" failed while reading \"{}\": {}",
                filename, resolved, err
            ));
        }
    }
}
