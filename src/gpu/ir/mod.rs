//! Minimal semantic GPU IR used by the backend contract.
//!
//! This is intentionally not a device ABI. It contains no `wgpu` handles,
//! raw pointers, shader bindings, or CPU trait objects. Geometry, materials,
//! and textures will be added in later IR phases.

mod geometry;
mod math;
mod render;
mod texture;

pub use geometry::*;
pub use math::*;
pub use render::*;
pub use texture::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IrVersion {
    pub major: u16,
    pub minor: u16,
}

pub const CURRENT_IR_VERSION: IrVersion = IrVersion { major: 1, minor: 0 };

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Feature {
    TriangleMesh,
    StaticTransform,
    AnimatedTransform,
    FloatConstantTexture,
    FloatImageTexture,
    SpectrumConstantTexture,
    SpectrumImageTexture,
    DiffuseMaterial,
    PointLight,
    DiffuseAreaLight,
    UniformInfiniteLight,
    PerspectiveCamera,
    IndependentSampler,
    RgbFilm,
    BoxFilter,
    WavefrontVolPath,
    UniformLightSampler,
    Curve,
    BilinearPatch,
    Quadric,
    DisplacedTriangle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequiredFeature {
    pub feature: Feature,
    pub sources: Box<[SourceId]>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceCounts {
    pub transforms: u64,
    pub spectra: u64,
    pub images: u64,
    pub texture_mappings: u64,
    pub float_textures: u64,
    pub spectrum_textures: u64,
    pub materials: u64,
    pub lights: u64,
    pub geometries: u64,
    pub primitives: u64,
    pub instance_definitions: u64,
    pub instances: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SemanticMaxima {
    pub texture_graph_depth: u32,
    pub material_graph_depth: u32,
    pub instance_depth: u32,
    pub image_dimension: u32,
    pub vertices_per_geometry: u64,
    pub elements_per_geometry: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Requirements {
    pub features: Box<[RequiredFeature]>,
    pub resource_counts: ResourceCounts,
    pub maxima: SemanticMaxima,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bounds2i {
    pub min: [Index; 2],
    pub max: [Index; 2],
}

impl Bounds2i {
    pub fn pixel_count(self) -> Option<usize> {
        let width = self.max[0].checked_sub(self.min[0])?;
        let height = self.max[1].checked_sub(self.min[1])?;
        usize::try_from(width)
            .ok()?
            .checked_mul(usize::try_from(height).ok()?)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InstanceDefinition {
    pub primitives: Vec<PrimitiveId>,
    pub instances: Vec<InstanceId>,
    pub local_bounds: Bounds3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Instance {
    pub definition: InstanceDefinitionId,
    pub transform: TransformId,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffuseMaterial {
    pub reflectance: SpectrumTextureId,
    pub displacement: Option<FloatTextureId>,
    pub normal_map: Option<ImageId>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Material {
    Diffuse(DiffuseMaterial),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointLight {
    pub render_from_light: TransformId,
    pub intensity: SpectrumId,
    pub scale: Float,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffuseAreaLight {
    pub emission: SpectrumTextureId,
    pub scale: Float,
    pub two_sided: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformInfiniteLight {
    pub radiance: SpectrumId,
    pub scale: Float,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Light {
    Point(PointLight),
    DiffuseArea(DiffuseAreaLight),
    UniformInfinite(UniformInfiniteLight),
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            camera: PerspectiveCamera {
                render_from_camera: TransformId(0),
                camera_from_raster: Matrix4x4::identity(),
                lens_radius: 0.0,
                focal_distance: 1.0,
                shutter_open: 0.0,
                shutter_close: 1.0,
            },
            sampler: IndependentSampler {
                samples_per_pixel: 1,
                seed: 0,
            },
            film: RgbFilm {
                full_resolution: [1, 1],
                pixel_bounds: Bounds2i {
                    min: [0, 0],
                    max: [1, 1],
                },
                diagonal_mm: 35.0,
                output_rgb_from_xyz: Matrix3x3::identity(),
                iso: 100.0,
                max_component_value: 1e6,
            },
            filter: BoxFilter {
                radius: Vector2([0.5, 0.5]),
            },
            integrator: WavefrontVolPath {
                max_depth: 5,
                regularize: false,
            },
            light_sampler: LightSampler::Uniform,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SceneData {
    pub transforms: Vec<Transform>,
    pub spectra: Vec<SpectrumResource>,
    pub float_textures: Vec<FloatTexture>,
    pub spectrum_textures: Vec<SpectrumTexture>,
    pub texture_mappings: Vec<TextureMapping>,
    pub images: Vec<ImageResource>,
    pub geometry: Vec<Geometry>,
    pub materials: Vec<Material>,
    pub lights: Vec<Light>,
    pub primitives: Vec<Primitive>,
    pub instance_definitions: Vec<InstanceDefinition>,
    pub instances: Vec<Instance>,
    pub world_primitives: Box<[PrimitiveId]>,
    pub world_instances: Box<[InstanceId]>,
    pub render: RenderConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SceneDraft {
    pub version: IrVersion,
    pub data: SceneData,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SceneIr {
    version: IrVersion,
    data: SceneData,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneView<'a> {
    pub version: &'a IrVersion,
    pub transforms: &'a [Transform],
    pub spectra: &'a [SpectrumResource],
    pub float_textures: &'a [FloatTexture],
    pub spectrum_textures: &'a [SpectrumTexture],
    pub texture_mappings: &'a [TextureMapping],
    pub images: &'a [ImageResource],
    pub geometry: &'a [Geometry],
    pub materials: &'a [Material],
    pub lights: &'a [Light],
    pub primitives: &'a [Primitive],
    pub instance_definitions: &'a [InstanceDefinition],
    pub instances: &'a [Instance],
    pub world_primitives: &'a [PrimitiveId],
    pub world_instances: &'a [InstanceId],
    pub render: &'a RenderConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IrValidationError {
    UnsupportedMajorVersion {
        found: IrVersion,
        expected_major: u16,
    },
    InvalidPixelBounds,
    InvalidSampleCount,
    EmptyTriangleMesh {
        geometry: GeometryId,
    },
    TriangleIndexOutOfBounds {
        geometry: GeometryId,
        index: Index,
    },
    DegenerateTriangle {
        geometry: GeometryId,
        triangle: Index,
    },
    AttributeLengthMismatch {
        geometry: GeometryId,
    },
    InvalidGeometryReference {
        primitive: PrimitiveId,
        geometry: GeometryId,
    },
    InvalidTransformReference {
        primitive: PrimitiveId,
        transform: TransformId,
    },
    InvalidMaterialReference {
        primitive: PrimitiveId,
        material: MaterialId,
    },
    InvalidFloatTextureReference {
        primitive: PrimitiveId,
        texture: FloatTextureId,
    },
    InvalidMaterialFloatTextureReference {
        material: MaterialId,
        texture: FloatTextureId,
    },
    InvalidAreaLightReference {
        primitive: PrimitiveId,
        light: LightId,
    },
    InvalidSpectrumTextureReference {
        material: MaterialId,
        texture: SpectrumTextureId,
    },
    InvalidTextureSpectrumReference {
        texture: SpectrumTextureId,
        spectrum: SpectrumId,
    },
    InvalidImageReference {
        image: ImageId,
    },
    InvalidTextureMappingReference {
        mapping: TextureMappingId,
    },
    InvalidImageData {
        image: ImageId,
    },
    InvalidLightTransformReference {
        light: LightId,
        transform: TransformId,
    },
    InvalidLightSpectrumReference {
        light: LightId,
        spectrum: SpectrumId,
    },
    InvalidLightTextureReference {
        light: LightId,
        texture: SpectrumTextureId,
    },
    InvalidQuadric {
        geometry: GeometryId,
    },
    InvalidCurve {
        geometry: GeometryId,
        curve: Index,
    },
    InvalidInstanceDefinitionReference {
        instance: InstanceId,
        definition: InstanceDefinitionId,
    },
    InvalidInstanceTransformReference {
        instance: InstanceId,
        transform: TransformId,
    },
    InvalidInstancePrimitiveReference {
        definition: InstanceDefinitionId,
        primitive: PrimitiveId,
    },
    InvalidInstanceReference {
        definition: InstanceDefinitionId,
        instance: InstanceId,
    },
    InvalidWorldPrimitiveReference {
        primitive: PrimitiveId,
    },
    InvalidWorldInstanceReference {
        instance: InstanceId,
    },
    InvalidInstanceBounds {
        definition: InstanceDefinitionId,
    },
    InvalidDisplacementBase {
        geometry: GeometryId,
    },
    InvalidDisplacementTexture {
        geometry: GeometryId,
        texture: FloatTextureId,
    },
    InvalidDisplacementData {
        geometry: GeometryId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IrValidationErrors {
    issues: Box<[IrValidationError]>,
}

impl IrValidationErrors {
    pub fn issues(&self) -> &[IrValidationError] {
        &self.issues
    }
}

impl SceneDraft {
    pub fn finish(self) -> Result<SceneIr, IrValidationErrors> {
        let mut issues = Vec::new();
        if self.version.major != CURRENT_IR_VERSION.major {
            issues.push(IrValidationError::UnsupportedMajorVersion {
                found: self.version,
                expected_major: CURRENT_IR_VERSION.major,
            });
        }
        if self.data.render.film.pixel_bounds.area().is_none() {
            issues.push(IrValidationError::InvalidPixelBounds);
        }
        if self.data.render.sampler.samples_per_pixel == 0 {
            issues.push(IrValidationError::InvalidSampleCount);
        }
        for (geometry_index, geometry) in self.data.geometry.iter().enumerate() {
            let geometry_id = GeometryId(geometry_index as Index);
            match geometry {
                Geometry::TriangleMesh(mesh) => {
                    validate_triangle_mesh(geometry_id, mesh, &mut issues)
                }
                Geometry::BilinearPatchMesh(mesh) => {
                    validate_bilinear_mesh(geometry_id, mesh, &mut issues)
                }
                Geometry::CurveMesh(mesh) => validate_curve_mesh(geometry_id, mesh, &mut issues),
                Geometry::Quadric(quadric) => validate_quadric(geometry_id, quadric, &mut issues),
                Geometry::DisplacedTriangleMesh(mesh) => {
                    validate_displaced_mesh(geometry_id, mesh, &self.data, &mut issues)
                }
            }
        }
        for (primitive_index, primitive) in self.data.primitives.iter().enumerate() {
            let primitive_id = PrimitiveId(primitive_index as Index);
            if usize::try_from(primitive.geometry.0)
                .ok()
                .and_then(|index| self.data.geometry.get(index))
                .is_none()
            {
                issues.push(IrValidationError::InvalidGeometryReference {
                    primitive: primitive_id,
                    geometry: primitive.geometry,
                });
            }
            if usize::try_from(primitive.transform.0)
                .ok()
                .and_then(|index| self.data.transforms.get(index))
                .is_none()
            {
                issues.push(IrValidationError::InvalidTransformReference {
                    primitive: primitive_id,
                    transform: primitive.transform,
                });
            }
            if let Some(material) = primitive.material {
                if usize::try_from(material.0)
                    .ok()
                    .and_then(|index| self.data.materials.get(index))
                    .is_none()
                {
                    issues.push(IrValidationError::InvalidMaterialReference {
                        primitive: primitive_id,
                        material,
                    });
                }
            }
            if let Some(texture) = primitive.alpha {
                if usize::try_from(texture.0)
                    .ok()
                    .and_then(|index| self.data.float_textures.get(index))
                    .is_none()
                {
                    issues.push(IrValidationError::InvalidFloatTextureReference {
                        primitive: primitive_id,
                        texture,
                    });
                }
            }
            let area_lights = match &primitive.area_light {
                AreaLightBinding::None => &[][..],
                AreaLightBinding::Uniform(light) => std::slice::from_ref(light),
                AreaLightBinding::PerElement(lights) => lights,
            };
            for light in area_lights {
                if usize::try_from(light.0)
                    .ok()
                    .and_then(|index| self.data.lights.get(index))
                    .is_none()
                {
                    issues.push(IrValidationError::InvalidAreaLightReference {
                        primitive: primitive_id,
                        light: *light,
                    });
                }
            }
        }
        for primitive in &self.data.world_primitives {
            if usize::try_from(primitive.0)
                .ok()
                .and_then(|index| self.data.primitives.get(index))
                .is_none()
            {
                issues.push(IrValidationError::InvalidWorldPrimitiveReference {
                    primitive: *primitive,
                });
            }
        }
        for instance in &self.data.world_instances {
            if usize::try_from(instance.0)
                .ok()
                .and_then(|index| self.data.instances.get(index))
                .is_none()
            {
                issues.push(IrValidationError::InvalidWorldInstanceReference {
                    instance: *instance,
                });
            }
        }
        for (definition_index, definition) in self.data.instance_definitions.iter().enumerate() {
            let definition_id = InstanceDefinitionId(definition_index as Index);
            let valid_bounds = definition
                .local_bounds
                .min
                .0
                .iter()
                .zip(definition.local_bounds.max.0.iter())
                .all(|(min, max)| min.is_finite() && max.is_finite() && min <= max);
            if !valid_bounds {
                issues.push(IrValidationError::InvalidInstanceBounds {
                    definition: definition_id,
                });
            }
            for primitive in &definition.primitives {
                if usize::try_from(primitive.0)
                    .ok()
                    .and_then(|index| self.data.primitives.get(index))
                    .is_none()
                {
                    issues.push(IrValidationError::InvalidInstancePrimitiveReference {
                        definition: definition_id,
                        primitive: *primitive,
                    });
                }
            }
            for instance in &definition.instances {
                if usize::try_from(instance.0)
                    .ok()
                    .and_then(|index| self.data.instances.get(index))
                    .is_none()
                {
                    issues.push(IrValidationError::InvalidInstanceReference {
                        definition: definition_id,
                        instance: *instance,
                    });
                }
            }
        }
        for (instance_index, instance) in self.data.instances.iter().enumerate() {
            let instance_id = InstanceId(instance_index as Index);
            if usize::try_from(instance.definition.0)
                .ok()
                .and_then(|index| self.data.instance_definitions.get(index))
                .is_none()
            {
                issues.push(IrValidationError::InvalidInstanceDefinitionReference {
                    instance: instance_id,
                    definition: instance.definition,
                });
            }
            if usize::try_from(instance.transform.0)
                .ok()
                .and_then(|index| self.data.transforms.get(index))
                .is_none()
            {
                issues.push(IrValidationError::InvalidInstanceTransformReference {
                    instance: instance_id,
                    transform: instance.transform,
                });
            }
        }
        for (material_index, material) in self.data.materials.iter().enumerate() {
            let material_id = MaterialId(material_index as Index);
            let (texture, displacement, normal_map) = match material {
                Material::Diffuse(diffuse) => (
                    diffuse.reflectance,
                    diffuse.displacement,
                    diffuse.normal_map,
                ),
            };
            if usize::try_from(texture.0)
                .ok()
                .and_then(|index| self.data.spectrum_textures.get(index))
                .is_none()
            {
                issues.push(IrValidationError::InvalidSpectrumTextureReference {
                    material: material_id,
                    texture,
                });
            }
            if let Some(displacement) = displacement {
                if usize::try_from(displacement.0)
                    .ok()
                    .and_then(|index| self.data.float_textures.get(index))
                    .is_none()
                {
                    issues.push(IrValidationError::InvalidMaterialFloatTextureReference {
                        material: material_id,
                        texture: displacement,
                    });
                }
            }
            if let Some(normal_map) = normal_map {
                if usize::try_from(normal_map.0)
                    .ok()
                    .and_then(|index| self.data.images.get(index))
                    .is_none()
                {
                    issues.push(IrValidationError::InvalidImageReference { image: normal_map });
                }
            }
        }
        for (texture_index, texture) in self.data.spectrum_textures.iter().enumerate() {
            let texture_id = SpectrumTextureId(texture_index as Index);
            match texture {
                SpectrumTexture::Constant { value } => {
                    if usize::try_from(value.0)
                        .ok()
                        .and_then(|index| self.data.spectra.get(index))
                        .is_none()
                    {
                        issues.push(IrValidationError::InvalidTextureSpectrumReference {
                            texture: texture_id,
                            spectrum: *value,
                        });
                    }
                }
                SpectrumTexture::Image { image, mapping, .. } => {
                    validate_image_texture_refs(*image, *mapping, &self.data, &mut issues)
                }
            }
        }
        for texture in &self.data.float_textures {
            if let FloatTexture::Image { image, mapping, .. } = texture {
                validate_image_texture_refs(*image, *mapping, &self.data, &mut issues);
            }
        }
        for (light_index, light) in self.data.lights.iter().enumerate() {
            let light_id = LightId(light_index as Index);
            let (transform, spectrum, texture) = match light {
                Light::Point(point) => (Some(point.render_from_light), Some(point.intensity), None),
                Light::DiffuseArea(area) => (None, None, Some(area.emission)),
                Light::UniformInfinite(infinite) => (None, Some(infinite.radiance), None),
            };
            if let Some(transform) = transform {
                if usize::try_from(transform.0)
                    .ok()
                    .and_then(|index| self.data.transforms.get(index))
                    .is_none()
                {
                    issues.push(IrValidationError::InvalidLightTransformReference {
                        light: light_id,
                        transform,
                    });
                }
            }
            if let Some(spectrum) = spectrum {
                if usize::try_from(spectrum.0)
                    .ok()
                    .and_then(|index| self.data.spectra.get(index))
                    .is_none()
                {
                    issues.push(IrValidationError::InvalidLightSpectrumReference {
                        light: light_id,
                        spectrum,
                    });
                }
            }
            if let Some(texture) = texture {
                if usize::try_from(texture.0)
                    .ok()
                    .and_then(|index| self.data.spectrum_textures.get(index))
                    .is_none()
                {
                    issues.push(IrValidationError::InvalidLightTextureReference {
                        light: light_id,
                        texture,
                    });
                }
            }
        }
        if issues.is_empty() {
            Ok(SceneIr {
                version: self.version,
                data: self.data,
            })
        } else {
            Err(IrValidationErrors {
                issues: issues.into_boxed_slice(),
            })
        }
    }
}

fn validate_triangle_mesh(
    geometry: GeometryId,
    mesh: &TriangleMesh,
    issues: &mut Vec<IrValidationError>,
) {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        issues.push(IrValidationError::EmptyTriangleMesh { geometry });
        return;
    }
    let position_count = mesh.positions.len();
    for (triangle_index, triangle) in mesh.indices.iter().enumerate() {
        if triangle
            .iter()
            .any(|index| usize::try_from(*index).map_or(true, |index| index >= position_count))
        {
            let index = triangle
                .iter()
                .copied()
                .find(|index| usize::try_from(*index).map_or(true, |index| index >= position_count))
                .unwrap_or_default();
            issues.push(IrValidationError::TriangleIndexOutOfBounds { geometry, index });
        }
        if triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[2] == triangle[0] {
            issues.push(IrValidationError::DegenerateTriangle {
                geometry,
                triangle: triangle_index as Index,
            });
        }
    }
    let expected = position_count;
    if mesh.normals.as_ref().is_some_and(|v| v.len() != expected)
        || mesh.tangents.as_ref().is_some_and(|v| v.len() != expected)
        || mesh.uvs.as_ref().is_some_and(|v| v.len() != expected)
        || mesh
            .face_indices
            .as_ref()
            .is_some_and(|v| v.len() != mesh.indices.len())
    {
        issues.push(IrValidationError::AttributeLengthMismatch { geometry });
    }
}

fn validate_bilinear_mesh(
    geometry: GeometryId,
    mesh: &BilinearPatchMesh,
    issues: &mut Vec<IrValidationError>,
) {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        issues.push(IrValidationError::EmptyTriangleMesh { geometry });
        return;
    }
    let position_count = mesh.positions.len();
    for patch in &mesh.indices {
        if patch
            .iter()
            .any(|index| usize::try_from(*index).map_or(true, |index| index >= position_count))
        {
            let index = patch
                .iter()
                .copied()
                .find(|index| usize::try_from(*index).map_or(true, |index| index >= position_count))
                .unwrap_or_default();
            issues.push(IrValidationError::TriangleIndexOutOfBounds { geometry, index });
        }
    }
    if mesh
        .normals
        .as_ref()
        .is_some_and(|v| v.len() != position_count)
        || mesh.uvs.as_ref().is_some_and(|v| v.len() != position_count)
        || mesh
            .face_indices
            .as_ref()
            .is_some_and(|v| v.len() != mesh.indices.len())
    {
        issues.push(IrValidationError::AttributeLengthMismatch { geometry });
    }
}

fn validate_curve_mesh(
    geometry: GeometryId,
    mesh: &CurveMesh,
    issues: &mut Vec<IrValidationError>,
) {
    if mesh.curves.is_empty() {
        issues.push(IrValidationError::InvalidCurve { geometry, curve: 0 });
    }
    for (curve_index, curve) in mesh.curves.iter().enumerate() {
        let valid_points = curve
            .control_points
            .iter()
            .all(|point| point.0.iter().all(|value| value.is_finite()));
        let valid_widths = curve
            .widths
            .iter()
            .all(|width| width.is_finite() && *width >= 0.0);
        let valid_normals = curve.endpoint_normals.as_ref().map_or(true, |normals| {
            normals
                .iter()
                .all(|normal| normal.0.iter().all(|value| value.is_finite()))
        });
        let type_valid = match mesh.curve_type {
            CurveType::Ribbon => curve.endpoint_normals.is_some(),
            CurveType::Flat | CurveType::Cylinder => curve.endpoint_normals.is_none(),
        };
        if !valid_points || !valid_widths || !valid_normals || !type_valid {
            issues.push(IrValidationError::InvalidCurve {
                geometry,
                curve: curve_index as Index,
            });
        }
    }
}

fn validate_quadric(geometry: GeometryId, quadric: &Quadric, issues: &mut Vec<IrValidationError>) {
    let valid = match quadric {
        Quadric::Sphere {
            radius,
            z_min,
            z_max,
            phi_max_radians,
        }
        | Quadric::Cylinder {
            radius,
            z_min,
            z_max,
            phi_max_radians,
        } => {
            radius.is_finite()
                && *radius > 0.0
                && z_min.is_finite()
                && z_max.is_finite()
                && z_min <= z_max
                && phi_max_radians.is_finite()
                && *phi_max_radians > 0.0
                && *phi_max_radians <= 2.0 * std::f32::consts::PI
        }
        Quadric::Disk {
            height,
            radius,
            inner_radius,
            phi_max_radians,
        } => {
            height.is_finite()
                && radius.is_finite()
                && *radius > 0.0
                && inner_radius.is_finite()
                && *inner_radius >= 0.0
                && *inner_radius < *radius
                && phi_max_radians.is_finite()
                && *phi_max_radians > 0.0
                && *phi_max_radians <= 2.0 * std::f32::consts::PI
        }
    };
    if !valid {
        issues.push(IrValidationError::InvalidQuadric { geometry });
    }
}

fn validate_displaced_mesh(
    geometry: GeometryId,
    mesh: &DisplacedTriangleMesh,
    data: &SceneData,
    issues: &mut Vec<IrValidationError>,
) {
    if !matches!(
        usize::try_from(mesh.base_mesh.0)
            .ok()
            .and_then(|index| data.geometry.get(index)),
        Some(Geometry::TriangleMesh(_))
    ) {
        issues.push(IrValidationError::InvalidDisplacementBase { geometry });
    }
    if usize::try_from(mesh.displacement.0)
        .ok()
        .and_then(|index| data.float_textures.get(index))
        .is_none()
    {
        issues.push(IrValidationError::InvalidDisplacementTexture {
            geometry,
            texture: mesh.displacement,
        });
    }
    let finite = mesh.displacement_scale.is_finite()
        && mesh.displacement_offset.is_finite()
        && mesh.edge_length.is_finite()
        && mesh.edge_length > 0.0;
    let bounds_match = mesh.triangle_roots.len() == mesh.displaced_bounds_object.len();
    let roots_valid = mesh.triangle_roots.iter().all(|root| {
        usize::try_from(root.0)
            .ok()
            .and_then(|index| mesh.min_max_nodes.get(index))
            .is_some()
    });
    let nodes_valid = mesh.min_max_nodes.iter().all(|node| {
        node.parameter_bounds
            .min
            .0
            .iter()
            .zip(node.parameter_bounds.max.0.iter())
            .all(|(min, max)| min.is_finite() && max.is_finite() && min <= max)
            && node.displacement_min.is_finite()
            && node.displacement_max.is_finite()
            && node.displacement_min <= node.displacement_max
            && node.children.map_or(true, |children| {
                children.iter().all(|child| {
                    usize::try_from(child.0)
                        .ok()
                        .and_then(|index| mesh.min_max_nodes.get(index))
                        .is_some()
                })
            })
    });
    let bounds_valid = mesh.displaced_bounds_object.iter().all(|bounds| {
        bounds
            .min
            .0
            .iter()
            .zip(bounds.max.0.iter())
            .all(|(min, max)| min.is_finite() && max.is_finite() && min <= max)
    });
    let graph_valid = validate_minmax_graph(&mesh.min_max_nodes);
    if !finite || !bounds_match || !roots_valid || !nodes_valid || !bounds_valid || !graph_valid {
        issues.push(IrValidationError::InvalidDisplacementData { geometry });
    }
}

fn validate_minmax_graph(nodes: &[MinMaxNode]) -> bool {
    for parent in nodes {
        let Some(children) = parent.children else {
            continue;
        };
        for child_id in children {
            let Some(child) = usize::try_from(child_id.0)
                .ok()
                .and_then(|index| nodes.get(index))
            else {
                return false;
            };
            let parameter_contained = child
                .parameter_bounds
                .min
                .0
                .iter()
                .zip(parent.parameter_bounds.min.0.iter())
                .zip(
                    child
                        .parameter_bounds
                        .max
                        .0
                        .iter()
                        .zip(parent.parameter_bounds.max.0.iter()),
                )
                .all(|((child_min, parent_min), (child_max, parent_max))| {
                    child_min >= parent_min && child_max <= parent_max
                });
            if !parameter_contained
                || child.displacement_min < parent.displacement_min
                || child.displacement_max > parent.displacement_max
            {
                return false;
            }
        }
    }
    let mut marks = vec![0_u8; nodes.len()];
    for index in 0..nodes.len() {
        if !visit_minmax_node(index, nodes, &mut marks) {
            return false;
        }
    }
    true
}

fn visit_minmax_node(index: usize, nodes: &[MinMaxNode], marks: &mut [u8]) -> bool {
    if marks[index] == 1 {
        return false;
    }
    if marks[index] == 2 {
        return true;
    }
    marks[index] = 1;
    if let Some(children) = nodes[index].children {
        for child in children {
            let Some(child_index) = usize::try_from(child.0).ok() else {
                return false;
            };
            if child_index >= nodes.len() || !visit_minmax_node(child_index, nodes, marks) {
                return false;
            }
        }
    }
    marks[index] = 2;
    true
}

impl SceneIr {
    pub fn requirements(&self) -> Requirements {
        use std::collections::BTreeSet;

        let data = &self.data;
        let mut features = BTreeSet::new();
        for transform in &data.transforms {
            features.insert(match transform {
                Transform::Static(_) => Feature::StaticTransform,
                Transform::Animated(_) => Feature::AnimatedTransform,
            });
        }
        for geometry in &data.geometry {
            features.insert(match geometry {
                Geometry::TriangleMesh(_) => Feature::TriangleMesh,
                Geometry::BilinearPatchMesh(_) => Feature::BilinearPatch,
                Geometry::CurveMesh(_) => Feature::Curve,
                Geometry::Quadric(_) => Feature::Quadric,
                Geometry::DisplacedTriangleMesh(_) => Feature::DisplacedTriangle,
            });
        }
        for texture in &data.float_textures {
            features.insert(match texture {
                FloatTexture::Constant { .. } => Feature::FloatConstantTexture,
                FloatTexture::Image { .. } => Feature::FloatImageTexture,
            });
        }
        for texture in &data.spectrum_textures {
            features.insert(match texture {
                SpectrumTexture::Constant { .. } => Feature::SpectrumConstantTexture,
                SpectrumTexture::Image { .. } => Feature::SpectrumImageTexture,
            });
        }
        for material in &data.materials {
            if matches!(material, Material::Diffuse(_)) {
                features.insert(Feature::DiffuseMaterial);
            }
        }
        for light in &data.lights {
            features.insert(match light {
                Light::Point(_) => Feature::PointLight,
                Light::DiffuseArea(_) => Feature::DiffuseAreaLight,
                Light::UniformInfinite(_) => Feature::UniformInfiniteLight,
            });
        }
        features.extend([
            Feature::PerspectiveCamera,
            Feature::IndependentSampler,
            Feature::RgbFilm,
            Feature::BoxFilter,
            Feature::WavefrontVolPath,
            Feature::UniformLightSampler,
        ]);

        let resource_counts = ResourceCounts {
            transforms: data.transforms.len() as u64,
            spectra: data.spectra.len() as u64,
            images: data.images.len() as u64,
            texture_mappings: data.texture_mappings.len() as u64,
            float_textures: data.float_textures.len() as u64,
            spectrum_textures: data.spectrum_textures.len() as u64,
            materials: data.materials.len() as u64,
            lights: data.lights.len() as u64,
            geometries: data.geometry.len() as u64,
            primitives: data.primitives.len() as u64,
            instance_definitions: data.instance_definitions.len() as u64,
            instances: data.instances.len() as u64,
        };
        let maxima = SemanticMaxima {
            texture_graph_depth: u32::from(
                !data.float_textures.is_empty() || !data.spectrum_textures.is_empty(),
            ),
            material_graph_depth: u32::from(!data.materials.is_empty()),
            instance_depth: u32::from(!data.instances.is_empty()),
            image_dimension: data
                .images
                .iter()
                .flat_map(|image| image.resolution)
                .max()
                .unwrap_or(0),
            vertices_per_geometry: data
                .geometry
                .iter()
                .map(|geometry| match geometry {
                    Geometry::TriangleMesh(mesh) => mesh.positions.len() as u64,
                    Geometry::BilinearPatchMesh(mesh) => mesh.positions.len() as u64,
                    Geometry::CurveMesh(mesh) => (mesh.curves.len() * 4) as u64,
                    Geometry::Quadric(_) => 0,
                    Geometry::DisplacedTriangleMesh(mesh) => {
                        mesh.displaced_bounds_object.len() as u64
                    }
                })
                .max()
                .unwrap_or(0),
            elements_per_geometry: data
                .geometry
                .iter()
                .map(|geometry| match geometry {
                    Geometry::TriangleMesh(mesh) => mesh.indices.len() as u64,
                    Geometry::BilinearPatchMesh(mesh) => mesh.indices.len() as u64,
                    Geometry::CurveMesh(mesh) => mesh.curves.len() as u64,
                    Geometry::Quadric(_) => 1,
                    Geometry::DisplacedTriangleMesh(mesh) => mesh.triangle_roots.len() as u64,
                })
                .max()
                .unwrap_or(0),
        };
        Requirements {
            features: features
                .into_iter()
                .map(|feature| RequiredFeature {
                    feature,
                    sources: Box::new([]),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            resource_counts,
            maxima,
        }
    }

    pub fn view(&self) -> SceneView<'_> {
        SceneView {
            version: &self.version,
            transforms: &self.data.transforms,
            spectra: &self.data.spectra,
            float_textures: &self.data.float_textures,
            spectrum_textures: &self.data.spectrum_textures,
            texture_mappings: &self.data.texture_mappings,
            images: &self.data.images,
            geometry: &self.data.geometry,
            materials: &self.data.materials,
            lights: &self.data.lights,
            primitives: &self.data.primitives,
            instance_definitions: &self.data.instance_definitions,
            instances: &self.data.instances,
            world_primitives: &self.data.world_primitives,
            world_instances: &self.data.world_instances,
            render: &self.data.render,
        }
    }
}

fn validate_image_texture_refs(
    image: ImageId,
    mapping: TextureMappingId,
    data: &SceneData,
    issues: &mut Vec<IrValidationError>,
) {
    let Some(image_resource) = usize::try_from(image.0)
        .ok()
        .and_then(|index| data.images.get(index))
    else {
        issues.push(IrValidationError::InvalidImageReference { image });
        return;
    };
    if usize::try_from(mapping.0)
        .ok()
        .and_then(|index| data.texture_mappings.get(index))
        .is_none()
    {
        issues.push(IrValidationError::InvalidTextureMappingReference { mapping });
    }
    let (storage_len, storage_finite) = match &image_resource.storage {
        TexelStorage::U8(values) => (values.len(), true),
        TexelStorage::F16(values) => (
            values.len(),
            values
                .iter()
                .all(|value| half::f16::from_bits(*value).to_f32().is_finite()),
        ),
        TexelStorage::F32(values) => (values.len(), values.iter().all(|value| value.is_finite())),
    };
    let channels = image_resource.channels.count();
    let base_components = usize::try_from(image_resource.resolution[0])
        .ok()
        .and_then(|width| {
            usize::try_from(image_resource.resolution[1])
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(channels));
    let mip_layout_valid = !image_resource.mip_levels.is_empty()
        && image_resource
            .mip_levels
            .first()
            .is_some_and(|level| level.resolution == image_resource.resolution)
        && image_resource.mip_levels.windows(2).all(|levels| {
            levels[1].resolution
                == [
                    (levels[0].resolution[0] / 2).max(1),
                    (levels[0].resolution[1] / 2).max(1),
                ]
        })
        && image_resource
            .mip_levels
            .last()
            .is_some_and(|level| level.resolution == [1, 1])
        && image_resource.mip_levels.iter().all(|level| {
            let expected = usize::try_from(level.resolution[0])
                .ok()
                .and_then(|width| {
                    usize::try_from(level.resolution[1])
                        .ok()
                        .and_then(|height| width.checked_mul(height))
                })
                .and_then(|pixels| pixels.checked_mul(channels));
            expected == level.texel_count.try_into().ok()
                && level
                    .texel_offset
                    .checked_add(level.texel_count)
                    .is_some_and(|end| end <= storage_len as u64)
        })
        && image_resource
            .mip_levels
            .windows(2)
            .all(|levels| levels[0].texel_offset + levels[0].texel_count <= levels[1].texel_offset);
    let encoding_valid = match image_resource.color_encoding {
        ColorEncoding::Linear | ColorEncoding::Srgb => true,
        ColorEncoding::Gamma { exponent } => exponent.is_finite() && exponent > 0.0,
    };
    if base_components
        != image_resource
            .mip_levels
            .first()
            .map(|level| level.texel_count.try_into().unwrap_or(usize::MAX))
        || !mip_layout_valid
        || !encoding_valid
        || image_resource.resolution.contains(&0)
        || !storage_finite
    {
        issues.push(IrValidationError::InvalidImageData { image });
    }
}
