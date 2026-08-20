use crate::interaction::*;
use crate::paramdict::*;
use crate::shapes::*;
use crate::textures::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;

use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default)]
pub struct ShapeSampleContext {
    pub p: Point3f,
    pub n: Normal3f,
    pub ns: Normal3f,
    pub time: Float,
}

impl ShapeSampleContext {
    pub fn get_p(&self) -> Point3f {
        self.p
    }

    pub fn get_n(&self) -> Normal3f {
        self.n
    }

    pub fn get_time(&self) -> Float {
        self.time
    }

    pub fn as_surface_interaction(&self) -> Option<SurfaceInteraction> {
        if self.n.length_squared() == 0.0 && self.ns.length_squared() == 0.0 {
            return None;
        }
        let mut si = SurfaceInteraction::default();
        si.p = self.p;
        si.n = self.n;
        si.time = self.time;
        si.shading.n = self.ns;
        Some(si)
    }
}

impl From<&Interaction> for ShapeSampleContext {
    fn from(value: &Interaction) -> Self {
        if let Some(si) = value.as_surface_interaction() {
            Self {
                p: si.p,
                n: si.n,
                ns: si.shading.n,
                time: si.time,
            }
        } else {
            Self {
                p: value.get_p(),
                n: value.get_n(),
                ns: value.get_n(),
                time: value.get_time(),
            }
        }
    }
}

fn wrap_alpha_masks(
    shapes: Vec<Shape>,
    params: &ParameterDictionary,
    float_textures: &HashMap<String, Arc<FloatTexture>>,
) -> Result<Vec<Shape>, PbrtError> {
    let alpha_mask_info = get_alpha_texture(params, float_textures)?;
    let shadow_alpha_mask_info = get_shadow_alpha_texture(params, float_textures)?;
    if alpha_mask_info.is_none() && shadow_alpha_mask_info.is_none() {
        return Ok(shapes);
    }

    Ok(shapes
        .into_iter()
        .map(|shape| {
            let shape = Arc::new(shape);
            Shape::AlphaMask(Box::new(AlphaMaskShape::new(
                &shape,
                &alpha_mask_info,
                &shadow_alpha_mask_info,
            )))
        })
        .collect())
}

/// Shape enum that unifies all shape types
///
/// Benefits:
/// - Better performance (no dynamic dispatch)
/// - Easier to reason about
/// - More idiomatic Rust
/// - Matches pbrt-v4's TaggedPointer pattern
// Rust pads every `Shape` allocation to the size of its largest
// variant. The bulky variants (`Sphere`/`Cylinder`/`Disk`/`Cone`/
// `Paraboloid`/`Hyperboloid` all carry a `BaseShape` with two
// `Transform` = two 4x4 matrices ≈ 270-320 bytes; `AlphaMaskShape`
// is ~48 bytes) blow the enum out so every Triangle pays for that
// padding. On kroken's ~50M-triangle scene the padding alone wastes
// ~10 GB.
//
// `Triangle` (32 bytes) is the largest inline variant, so any other
// variant ≤ 32 bytes can stay inline without changing the enum size.
// `BilinearPatch` is 24 bytes — boxing it would not shrink the enum,
// just add a heap alloc per patch on scenes that use them, so it
// stays inline. `AlphaMaskShape` IS bigger than Triangle AND hot
// per-mesh on textured-alpha scenes (watercolor, kroken), so it
// gets boxed. The truly large shapes (transform-carrying primitives)
// only ever appear in handfuls per scene, so the extra heap alloc
// each is negligible.
pub enum Shape {
    Sphere(Box<Sphere>),
    Cylinder(Box<Cylinder>),
    Disk(Box<Disk>),
    Cone(Box<Cone>),
    Paraboloid(Box<Paraboloid>),
    Hyperboloid(Box<Hyperboloid>),
    Triangle(Triangle),
    BilinearPatch(BilinearPatch),
    Curve(Curve),
    AlphaMask(Box<AlphaMaskShape>),
}

impl Shape {
    pub fn create_curves(
        render_from_object: &Transform,
        object_from_render: &Transform,
        reverse_orientation: bool,
        shape_params: &[ParameterDictionary],
        float_textures: &HashMap<String, Arc<FloatTexture>>,
    ) -> Result<Vec<Vec<Shape>>, PbrtError> {
        let curve_sets = create_curves_shape(
            render_from_object,
            object_from_render,
            reverse_orientation,
            shape_params,
        )?;
        let mut shape_sets = Vec::with_capacity(curve_sets.len());
        debug_assert_eq!(curve_sets.len(), shape_params.len());
        for (curves, params) in curve_sets.into_iter().zip(shape_params) {
            let curve_shapes = curves.into_iter().map(Shape::Curve).collect();
            shape_sets.push(wrap_alpha_masks(curve_shapes, params, float_textures)?);
        }
        Ok(shape_sets)
    }

    /// Create shapes from shape name and parameters
    ///
    /// Corresponds to pbrt-v4's Shape::Create
    ///
    /// # Arguments
    /// * `name` - Shape type name (sphere, cylinder, disk, etc.)
    /// * `render_from_object` - Object-to-world transformation
    /// * `object_from_render` - World-to-object transformation
    /// * `reverse_orientation` - Whether to reverse surface normal orientation
    /// * `params` - Shape parameters
    /// * `float_textures` - Map of available float textures for displacement mapping
    ///
    /// # Returns
    /// * `Result<Vec<Shape>, PbrtError>` - Vector of created shapes (may be multiple for meshes)
    pub fn create(
        name: &str,
        render_from_object: &Transform,
        object_from_render: &Transform,
        reverse_orientation: bool,
        params: &ParameterDictionary,
        float_textures: &HashMap<String, Arc<FloatTexture>>,
    ) -> Result<Vec<Shape>, PbrtError> {
        match name {
            "sphere" => {
                let s = Sphere::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                )?;
                return wrap_alpha_masks(vec![Shape::Sphere(Box::new(s))], params, float_textures);
            }
            "cylinder" => {
                let s = Cylinder::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                )?;
                return wrap_alpha_masks(
                    vec![Shape::Cylinder(Box::new(s))],
                    params,
                    float_textures,
                );
            }
            "disk" => {
                let s = Disk::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                )?;
                return wrap_alpha_masks(vec![Shape::Disk(Box::new(s))], params, float_textures);
            }
            "cone" => {
                let s = Cone::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                )?;
                return wrap_alpha_masks(vec![Shape::Cone(Box::new(s))], params, float_textures);
            }
            "paraboloid" => {
                let s = Paraboloid::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                )?;
                return wrap_alpha_masks(
                    vec![Shape::Paraboloid(Box::new(s))],
                    params,
                    float_textures,
                );
            }
            "hyperboloid" => {
                let s = Hyperboloid::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                )?;
                return wrap_alpha_masks(
                    vec![Shape::Hyperboloid(Box::new(s))],
                    params,
                    float_textures,
                );
            }
            "curve" => {
                let curves = Curve::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                )?;
                let shapes = curves.into_iter().map(Shape::Curve).collect();
                return wrap_alpha_masks(shapes, params, float_textures);
            }
            "trianglemesh" => {
                return TriangleMesh::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                    float_textures,
                );
            }
            "bilinearmesh" => {
                return BilinearPatchMesh::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                    float_textures,
                );
            }
            "plymesh" => {
                return PlyMesh::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                    float_textures,
                );
            }
            "heightfield" => {
                let shapes = HeightField::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                )?;
                return wrap_alpha_masks(shapes, params, float_textures);
            }
            "loopsubdiv" => {
                let shapes = LoopSubdiv::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                )?;
                return wrap_alpha_masks(shapes, params, float_textures);
            }
            "nurbs" => {
                let shapes = NURBS::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    params,
                )?;
                return wrap_alpha_masks(shapes, params, float_textures);
            }
            _ => {
                return Err(PbrtError::error(&format!("{} shape cannot create", name)));
            }
        }
    }

    /// Returns the object-space bounding box of the shape
    pub fn object_bound(&self) -> Bounds3f {
        match self {
            Shape::Sphere(s) => s.object_bound(),
            Shape::Cylinder(s) => s.object_bound(),
            Shape::Disk(s) => s.object_bound(),
            Shape::Cone(s) => s.object_bound(),
            Shape::Paraboloid(s) => s.object_bound(),
            Shape::Hyperboloid(s) => s.object_bound(),
            Shape::Triangle(s) => s.object_bound(),
            Shape::BilinearPatch(s) => s.object_bound(),
            Shape::Curve(s) => s.object_bound(),
            Shape::AlphaMask(s) => s.object_bound(),
        }
    }

    /// Returns the world-space bounding box of the shape
    pub fn world_bound(&self) -> Bounds3f {
        match self {
            Shape::Sphere(s) => s.world_bound(),
            Shape::Cylinder(s) => s.world_bound(),
            Shape::Disk(s) => s.world_bound(),
            Shape::Cone(s) => s.world_bound(),
            Shape::Paraboloid(s) => s.world_bound(),
            Shape::Hyperboloid(s) => s.world_bound(),
            Shape::Triangle(s) => s.world_bound(),
            Shape::BilinearPatch(s) => s.world_bound(),
            Shape::Curve(s) => s.world_bound(),
            Shape::AlphaMask(s) => s.world_bound(),
        }
    }

    /// pbrt-v4 `Shape::NormalBounds()` (shapes.h:1583). Returns a
    /// DirectionCone bounding every surface-normal direction the
    /// shape can emit. Used by `DiffuseAreaLight::Bounds` to feed
    /// `BVHLightSampler::Importance` so directional discrimination
    /// across emissive shapes can prune unfavourable light samples.
    pub fn normal_bounds(&self) -> DirectionCone {
        match self {
            Shape::Sphere(s) => s.normal_bounds(),
            Shape::Cylinder(s) => s.normal_bounds(),
            Shape::Disk(s) => s.normal_bounds(),
            Shape::Cone(s) => s.normal_bounds(),
            Shape::Paraboloid(s) => s.normal_bounds(),
            Shape::Hyperboloid(s) => s.normal_bounds(),
            Shape::Triangle(s) => s.normal_bounds(),
            Shape::BilinearPatch(s) => s.normal_bounds(),
            Shape::Curve(s) => s.normal_bounds(),
            Shape::AlphaMask(s) => s.normal_bounds(),
        }
    }

    pub fn has_constant_zero_alpha_mask(&self) -> bool {
        match self {
            Shape::AlphaMask(s) => s.has_constant_zero_alpha(),
            _ => false,
        }
    }

    pub fn alpha(&self, inter: &Interaction) -> Option<Float> {
        match self {
            Shape::AlphaMask(s) => s.alpha(inter),
            _ => None,
        }
    }

    /// Test for intersection with a ray
    ///
    /// # Arguments
    /// * `r` - The ray to test for intersection
    /// * `t_max` - Maximum parametric distance along the ray (matches pbrt-v4)
    ///
    /// # Returns
    /// * `Option<ShapeIntersection>` - ShapeIntersection containing the surface interaction and t_hit
    pub fn intersect(&self, r: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        match self {
            Shape::Sphere(s) => s.intersect(r, t_max),
            Shape::Cylinder(s) => s.intersect(r, t_max),
            Shape::Disk(s) => s.intersect(r, t_max),
            Shape::Cone(s) => s.intersect(r, t_max),
            Shape::Paraboloid(s) => s.intersect(r, t_max),
            Shape::Hyperboloid(s) => s.intersect(r, t_max),
            Shape::Triangle(s) => s.intersect(r, t_max),
            Shape::BilinearPatch(s) => s.intersect(r, t_max),
            Shape::Curve(s) => s.intersect(r, t_max),
            Shape::AlphaMask(s) => s.intersect(r, t_max),
        }
    }

    /// Predicate version of intersection test (faster, no SurfaceInteraction)
    ///
    /// # Arguments
    /// * `r` - The ray to test for intersection
    /// * `t_max` - Maximum parametric distance along the ray (matches pbrt-v4)
    pub fn intersect_p(&self, r: &Ray, t_max: Float) -> bool {
        match self {
            Shape::Sphere(s) => s.intersect_p(r, t_max),
            Shape::Cylinder(s) => s.intersect_p(r, t_max),
            Shape::Disk(s) => s.intersect_p(r, t_max),
            Shape::Cone(s) => s.intersect_p(r, t_max),
            Shape::Paraboloid(s) => s.intersect_p(r, t_max),
            Shape::Hyperboloid(s) => s.intersect_p(r, t_max),
            Shape::Triangle(s) => s.intersect_p(r, t_max),
            Shape::BilinearPatch(s) => s.intersect_p(r, t_max),
            Shape::Curve(s) => s.intersect_p(r, t_max),
            Shape::AlphaMask(s) => s.intersect_p(r, t_max),
        }
    }

    /// Returns the surface area of the shape
    pub fn area(&self) -> Float {
        match self {
            Shape::Sphere(s) => s.area(),
            Shape::Cylinder(s) => s.area(),
            Shape::Disk(s) => s.area(),
            Shape::Cone(s) => s.area(),
            Shape::Paraboloid(s) => s.area(),
            Shape::Hyperboloid(s) => s.area(),
            Shape::Triangle(s) => s.area(),
            Shape::BilinearPatch(s) => s.area(),
            Shape::Curve(s) => s.area(),
            Shape::AlphaMask(s) => s.area(),
        }
    }

    /// Sample a point on the surface
    pub fn sample(&self, u: &Point2f) -> Option<(Interaction, Float)> {
        match self {
            Shape::Sphere(s) => s.sample(u),
            Shape::Cylinder(s) => s.sample(u),
            Shape::Disk(s) => s.sample(u),
            Shape::Cone(s) => s.sample(u),
            Shape::Paraboloid(s) => s.sample(u),
            Shape::Hyperboloid(s) => s.sample(u),
            Shape::Triangle(s) => s.sample(u),
            Shape::BilinearPatch(s) => s.sample(u),
            Shape::Curve(s) => s.sample(u),
            Shape::AlphaMask(s) => s.sample(u),
        }
    }

    /// Probability density function for sampling
    pub fn pdf(&self, inter: &Interaction) -> Float {
        match self {
            Shape::Sphere(s) => s.pdf(inter),
            Shape::Cylinder(s) => s.pdf(inter),
            Shape::Disk(s) => s.pdf(inter),
            Shape::Cone(s) => s.pdf(inter),
            Shape::Paraboloid(s) => s.pdf(inter),
            Shape::Hyperboloid(s) => s.pdf(inter),
            Shape::Triangle(s) => s.pdf(inter),
            Shape::BilinearPatch(s) => s.pdf(inter),
            Shape::Curve(s) => s.pdf(inter),
            Shape::AlphaMask(s) => s.pdf(inter),
        }
    }

    /// Sample a point on the surface with respect to a reference point
    pub fn sample_from(
        &self,
        ctx: &ShapeSampleContext,
        u: &Point2f,
    ) -> Option<(Interaction, Float)> {
        let mut si = SurfaceInteraction::default();
        si.p = ctx.p;
        si.n = ctx.n;
        si.time = ctx.time;
        si.shading.n = ctx.ns;
        let inter = Interaction::Surface(si);
        match self {
            Shape::Sphere(s) => s.sample_from(&inter, u),
            Shape::Cylinder(s) => s.sample_from(&inter, u),
            Shape::Disk(s) => s.sample_from(&inter, u),
            Shape::Cone(s) => s.sample_from(&inter, u),
            Shape::Paraboloid(s) => s.sample_from(&inter, u),
            Shape::Hyperboloid(s) => s.sample_from(&inter, u),
            Shape::Triangle(s) => s.sample_from(&inter, u),
            Shape::BilinearPatch(s) => s.sample_from(&inter, u),
            Shape::Curve(s) => s.sample_from(&inter, u),
            Shape::AlphaMask(s) => s.sample_from(ctx, u),
        }
    }

    /// PDF for sampling with respect to a reference point
    pub fn pdf_from(&self, ctx: &ShapeSampleContext, wi: &Vector3f) -> Float {
        let mut si = SurfaceInteraction::default();
        si.p = ctx.p;
        si.n = ctx.n;
        si.time = ctx.time;
        si.shading.n = ctx.ns;
        let inter = Interaction::Surface(si);
        match self {
            Shape::Sphere(s) => s.pdf_from(&inter, wi),
            Shape::Cylinder(s) => s.pdf_from(&inter, wi),
            Shape::Disk(s) => s.pdf_from(&inter, wi),
            Shape::Cone(s) => s.pdf_from(&inter, wi),
            Shape::Paraboloid(s) => s.pdf_from(&inter, wi),
            Shape::Hyperboloid(s) => s.pdf_from(&inter, wi),
            Shape::Triangle(s) => s.pdf_from(&inter, wi),
            Shape::BilinearPatch(s) => s.pdf_from(&inter, wi),
            Shape::Curve(s) => s.pdf_from(&inter, wi),
            Shape::AlphaMask(s) => s.pdf_from(ctx, wi),
        }
    }

    /// Compute solid angle subtended by the shape from a point
    pub fn solid_angle(&self, p: &Point3f, n_samples: i32) -> Float {
        match self {
            Shape::Sphere(s) => s.solid_angle(p, n_samples),
            Shape::Cylinder(s) => s.solid_angle(p, n_samples),
            Shape::Disk(s) => s.solid_angle(p, n_samples),
            Shape::Cone(s) => s.solid_angle(p, n_samples),
            Shape::Paraboloid(s) => s.solid_angle(p, n_samples),
            Shape::Hyperboloid(s) => s.solid_angle(p, n_samples),
            Shape::Triangle(s) => s.solid_angle(p, n_samples),
            Shape::BilinearPatch(s) => s.solid_angle(p, n_samples),
            Shape::Curve(s) => s.solid_angle(p, n_samples),
            Shape::AlphaMask(s) => s.solid_angle(p, n_samples),
        }
    }
}
