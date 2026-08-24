//! Minimal semantic GPU IR used by the initial backend contract.
//!
//! This is intentionally not a device ABI. It contains no `wgpu` handles,
//! raw pointers, shader bindings, or CPU trait objects. Geometry, materials,
//! and textures will be added in later IR phases.

pub type GpuFloat = f32;
pub type GpuIndex = u32;

macro_rules! typed_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub struct $name(pub GpuIndex);
    };
}

typed_id!(TransformId);
typed_id!(GeometryId);
typed_id!(PrimitiveId);
typed_id!(MaterialId);
typed_id!(SpectrumId);
typed_id!(LightId);
typed_id!(FloatTextureId);
typed_id!(SpectrumTextureId);
typed_id!(TextureMappingId);
typed_id!(ImageId);
typed_id!(InstanceDefinitionId);
typed_id!(InstanceId);
typed_id!(MinMaxNodeId);
typed_id!(SourceId);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuPoint2(pub [GpuFloat; 2]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuPoint3(pub [GpuFloat; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuVector2(pub [GpuFloat; 2]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuVector3(pub [GpuFloat; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuNormal3(pub [GpuFloat; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMatrix4x4(pub [[GpuFloat; 4]; 4]);

impl GpuMatrix4x4 {
    pub fn identity() -> Self {
        Self([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMatrix3x3(pub [[GpuFloat; 3]; 3]);

impl GpuMatrix3x3 {
    pub fn identity() -> Self {
        Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuStaticTransform {
    pub render_from_object: GpuMatrix4x4,
    pub object_from_render: GpuMatrix4x4,
    pub swaps_handedness: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuTransform {
    Static(GpuStaticTransform),
    Animated(GpuAnimatedTransform),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuAnimatedTransform {
    pub start: GpuStaticTransform,
    pub end: GpuStaticTransform,
    pub start_time: GpuFloat,
    pub end_time: GpuFloat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBounds3 {
    pub min: GpuPoint3,
    pub max: GpuPoint3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBounds2 {
    pub min: GpuPoint2,
    pub max: GpuPoint2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRange {
    pub offset: GpuIndex,
    pub count: GpuIndex,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuSpectrumResource {
    Constant {
        value: GpuFloat,
    },
    PiecewiseLinear {
        wavelengths_nm: Box<[GpuFloat]>,
        values: Box<[GpuFloat]>,
    },
    RgbAlbedo {
        coefficients: [GpuFloat; 3],
    },
    RgbUnbounded {
        coefficients: [GpuFloat; 3],
    },
    RgbIlluminant {
        coefficients: [GpuFloat; 3],
        illuminant: SpectrumId,
    },
    Blackbody {
        temperature_kelvin: GpuFloat,
        scale: GpuFloat,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuImageChannels {
    R,
    Rg,
    Rgb,
    Rgba,
}

impl GpuImageChannels {
    pub fn count(self) -> usize {
        match self {
            Self::R => 1,
            Self::Rg => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuTexelStorage {
    U8(Box<[u8]>),
    F16(Box<[u16]>),
    F32(Box<[GpuFloat]>),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuColorEncoding {
    Linear,
    Srgb,
    Gamma { exponent: GpuFloat },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMipLevel {
    pub resolution: [GpuIndex; 2],
    pub texel_offset: u64,
    pub texel_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuImageResource {
    pub resolution: [GpuIndex; 2],
    pub channels: GpuImageChannels,
    pub storage: GpuTexelStorage,
    pub mip_levels: Box<[GpuMipLevel]>,
    pub color_encoding: GpuColorEncoding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuIrVersion {
    pub major: u16,
    pub minor: u16,
}

pub const CURRENT_IR_VERSION: GpuIrVersion = GpuIrVersion { major: 1, minor: 0 };

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum GpuFeature {
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
pub struct GpuRequiredFeature {
    pub feature: GpuFeature,
    pub sources: Box<[SourceId]>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuResourceCounts {
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
pub struct GpuSemanticMaxima {
    pub texture_graph_depth: u32,
    pub material_graph_depth: u32,
    pub instance_depth: u32,
    pub image_dimension: u32,
    pub vertices_per_geometry: u64,
    pub elements_per_geometry: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuRequirements {
    pub features: Box<[GpuRequiredFeature]>,
    pub resource_counts: GpuResourceCounts,
    pub maxima: GpuSemanticMaxima,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBounds2i {
    pub min: [GpuIndex; 2],
    pub max: [GpuIndex; 2],
}

impl GpuBounds2i {
    pub fn pixel_count(self) -> Option<usize> {
        let width = self.max[0].checked_sub(self.min[0])?;
        let height = self.max[1].checked_sub(self.min[1])?;
        usize::try_from(width)
            .ok()?
            .checked_mul(usize::try_from(height).ok()?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuRenderRequestError {
    ZeroSampleCount,
    SampleRangeOverflow,
    SampleRangeExceedsSamplesPerPixel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRenderRequest {
    pub sample_start: u64,
    pub sample_count: u32,
}

impl GpuRenderRequest {
    pub fn new(
        render: &GpuRenderConfig,
        sample_start: u64,
        sample_count: u32,
    ) -> Result<Self, GpuRenderRequestError> {
        if sample_count == 0 {
            return Err(GpuRenderRequestError::ZeroSampleCount);
        }
        let sample_end = sample_start
            .checked_add(u64::from(sample_count))
            .ok_or(GpuRenderRequestError::SampleRangeOverflow)?;
        if sample_end > u64::from(render.sampler.samples_per_pixel) {
            return Err(GpuRenderRequestError::SampleRangeExceedsSamplesPerPixel);
        }
        Ok(Self {
            sample_start,
            sample_count,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuRenderOutput {
    pub pixel_bounds: GpuBounds2i,
    pub rgb: Box<[[f32; 3]]>,
    pub sample_start: u64,
    pub sample_count: u32,
}

impl GpuBounds2i {
    pub fn area(self) -> Option<u64> {
        let width = u64::from(self.max[0]).checked_sub(u64::from(self.min[0]))?;
        let height = u64::from(self.max[1]).checked_sub(u64::from(self.min[1]))?;
        (width > 0 && height > 0).then(|| width.checked_mul(height))?
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuPerspectiveCamera {
    pub render_from_camera: TransformId,
    pub camera_from_raster: GpuMatrix4x4,
    pub lens_radius: GpuFloat,
    pub focal_distance: GpuFloat,
    pub shutter_open: GpuFloat,
    pub shutter_close: GpuFloat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuIndependentSampler {
    pub samples_per_pixel: u32,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRgbFilm {
    pub full_resolution: [GpuIndex; 2],
    pub pixel_bounds: GpuBounds2i,
    pub diagonal_mm: GpuFloat,
    pub output_rgb_from_xyz: GpuMatrix3x3,
    pub iso: GpuFloat,
    pub max_component_value: GpuFloat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBoxFilter {
    pub radius: GpuVector2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuWavefrontVolPath {
    pub max_depth: u32,
    pub regularize: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuLightSampler {
    Uniform,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRenderConfig {
    pub camera: GpuPerspectiveCamera,
    pub sampler: GpuIndependentSampler,
    pub film: GpuRgbFilm,
    pub filter: GpuBoxFilter,
    pub integrator: GpuWavefrontVolPath,
    pub light_sampler: GpuLightSampler,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuTriangleMesh {
    pub positions: Vec<GpuPoint3>,
    pub indices: Vec<[GpuIndex; 3]>,
    pub normals: Option<Vec<GpuNormal3>>,
    pub tangents: Option<Vec<GpuVector3>>,
    pub uvs: Option<Vec<GpuPoint2>>,
    pub face_indices: Option<Vec<GpuIndex>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuBilinearPatchMesh {
    pub positions: Vec<GpuPoint3>,
    pub indices: Vec<[GpuIndex; 4]>,
    pub normals: Option<Vec<GpuNormal3>>,
    pub uvs: Option<Vec<GpuPoint2>>,
    pub face_indices: Option<Vec<GpuIndex>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCurveType {
    Flat,
    Cylinder,
    Ribbon,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuCurveSegment {
    pub control_points: [GpuPoint3; 4],
    pub widths: [GpuFloat; 2],
    pub endpoint_normals: Option<[GpuNormal3; 2]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuCurveMesh {
    pub curve_type: GpuCurveType,
    pub curves: Vec<GpuCurveSegment>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuQuadric {
    Sphere {
        radius: GpuFloat,
        z_min: GpuFloat,
        z_max: GpuFloat,
        phi_max_radians: GpuFloat,
    },
    Cylinder {
        radius: GpuFloat,
        z_min: GpuFloat,
        z_max: GpuFloat,
        phi_max_radians: GpuFloat,
    },
    Disk {
        height: GpuFloat,
        radius: GpuFloat,
        inner_radius: GpuFloat,
        phi_max_radians: GpuFloat,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuMinMaxNode {
    pub parameter_bounds: GpuBounds2,
    pub displacement_min: GpuFloat,
    pub displacement_max: GpuFloat,
    pub children: Option<[MinMaxNodeId; 4]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuDisplacedTriangleMesh {
    pub base_mesh: GeometryId,
    pub displacement: FloatTextureId,
    pub displacement_scale: GpuFloat,
    pub displacement_offset: GpuFloat,
    pub edge_length: GpuFloat,
    pub min_max_nodes: Box<[GpuMinMaxNode]>,
    pub triangle_roots: Box<[MinMaxNodeId]>,
    pub displaced_bounds_object: Box<[GpuBounds3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuGeometry {
    TriangleMesh(GpuTriangleMesh),
    BilinearPatchMesh(GpuBilinearPatchMesh),
    CurveMesh(GpuCurveMesh),
    Quadric(GpuQuadric),
    DisplacedTriangleMesh(GpuDisplacedTriangleMesh),
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuAreaLightBinding {
    None,
    Uniform(LightId),
    PerElement(Vec<LightId>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuPrimitive {
    pub geometry: GeometryId,
    pub transform: TransformId,
    pub material: Option<MaterialId>,
    pub alpha: Option<FloatTextureId>,
    pub shadow_alpha: Option<FloatTextureId>,
    pub area_light: GpuAreaLightBinding,
    pub reverse_orientation: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuInstanceDefinition {
    pub primitives: Vec<PrimitiveId>,
    pub instances: Vec<InstanceId>,
    pub local_bounds: GpuBounds3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuInstance {
    pub definition: InstanceDefinitionId,
    pub transform: TransformId,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuDiffuseMaterial {
    pub reflectance: SpectrumTextureId,
    pub displacement: Option<FloatTextureId>,
    pub normal_map: Option<ImageId>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuMaterial {
    Diffuse(GpuDiffuseMaterial),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuImageWrapMode {
    Black,
    Clamp,
    Repeat,
    OctahedralSphere,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuImageFilter {
    Point,
    Bilinear,
    Trilinear,
    Ewa { max_anisotropy: GpuFloat },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuTextureMapping {
    Uv {
        su: GpuFloat,
        sv: GpuFloat,
        du: GpuFloat,
        dv: GpuFloat,
    },
    Spherical {
        texture_from_render: GpuMatrix4x4,
    },
    Cylindrical {
        texture_from_render: GpuMatrix4x4,
    },
    Planar {
        texture_from_render: GpuMatrix4x4,
        vs: GpuVector3,
        vt: GpuVector3,
        ds: GpuFloat,
        dt: GpuFloat,
    },
    Transform3D {
        texture_from_render: GpuMatrix4x4,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuFloatTexture {
    Constant {
        value: GpuFloat,
    },
    Image {
        image: ImageId,
        mapping: TextureMappingId,
        scale: GpuFloat,
        invert: bool,
        swrap: GpuImageWrapMode,
        twrap: GpuImageWrapMode,
        filter: GpuImageFilter,
        channel: GpuFloatImageChannel,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuFloatImageChannel {
    Channel0,
    Alpha,
    RgbAverage,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuSpectrumTexture {
    Constant {
        value: SpectrumId,
    },
    Image {
        image: ImageId,
        mapping: TextureMappingId,
        scale: GpuFloat,
        invert: bool,
        swrap: GpuImageWrapMode,
        twrap: GpuImageWrapMode,
        filter: GpuImageFilter,
        spectrum_type: GpuSpectrumType,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuSpectrumType {
    Albedo,
    Unbounded,
    Illuminant,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuPointLight {
    pub render_from_light: TransformId,
    pub intensity: SpectrumId,
    pub scale: GpuFloat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuDiffuseAreaLight {
    pub emission: SpectrumTextureId,
    pub scale: GpuFloat,
    pub two_sided: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuUniformInfiniteLight {
    pub radiance: SpectrumId,
    pub scale: GpuFloat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuLight {
    Point(GpuPointLight),
    DiffuseArea(GpuDiffuseAreaLight),
    UniformInfinite(GpuUniformInfiniteLight),
}

impl Default for GpuRenderConfig {
    fn default() -> Self {
        Self {
            camera: GpuPerspectiveCamera {
                render_from_camera: TransformId(0),
                camera_from_raster: GpuMatrix4x4::identity(),
                lens_radius: 0.0,
                focal_distance: 1.0,
                shutter_open: 0.0,
                shutter_close: 1.0,
            },
            sampler: GpuIndependentSampler {
                samples_per_pixel: 1,
                seed: 0,
            },
            film: GpuRgbFilm {
                full_resolution: [1, 1],
                pixel_bounds: GpuBounds2i {
                    min: [0, 0],
                    max: [1, 1],
                },
                diagonal_mm: 35.0,
                output_rgb_from_xyz: GpuMatrix3x3::identity(),
                iso: 100.0,
                max_component_value: 1e6,
            },
            filter: GpuBoxFilter {
                radius: GpuVector2([0.5, 0.5]),
            },
            integrator: GpuWavefrontVolPath {
                max_depth: 5,
                regularize: false,
            },
            light_sampler: GpuLightSampler::Uniform,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuSceneData {
    pub transforms: Vec<GpuTransform>,
    pub spectra: Vec<GpuSpectrumResource>,
    pub float_textures: Vec<GpuFloatTexture>,
    pub spectrum_textures: Vec<GpuSpectrumTexture>,
    pub texture_mappings: Vec<GpuTextureMapping>,
    pub images: Vec<GpuImageResource>,
    pub geometry: Vec<GpuGeometry>,
    pub materials: Vec<GpuMaterial>,
    pub lights: Vec<GpuLight>,
    pub primitives: Vec<GpuPrimitive>,
    pub instance_definitions: Vec<GpuInstanceDefinition>,
    pub instances: Vec<GpuInstance>,
    pub world_primitives: Box<[PrimitiveId]>,
    pub world_instances: Box<[InstanceId]>,
    pub render: GpuRenderConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuSceneDraft {
    pub version: GpuIrVersion,
    pub data: GpuSceneData,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuSceneIr {
    version: GpuIrVersion,
    data: GpuSceneData,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSceneView<'a> {
    pub version: &'a GpuIrVersion,
    pub transforms: &'a [GpuTransform],
    pub spectra: &'a [GpuSpectrumResource],
    pub float_textures: &'a [GpuFloatTexture],
    pub spectrum_textures: &'a [GpuSpectrumTexture],
    pub texture_mappings: &'a [GpuTextureMapping],
    pub images: &'a [GpuImageResource],
    pub geometry: &'a [GpuGeometry],
    pub materials: &'a [GpuMaterial],
    pub lights: &'a [GpuLight],
    pub primitives: &'a [GpuPrimitive],
    pub instance_definitions: &'a [GpuInstanceDefinition],
    pub instances: &'a [GpuInstance],
    pub world_primitives: &'a [PrimitiveId],
    pub world_instances: &'a [InstanceId],
    pub render: &'a GpuRenderConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuIrValidationError {
    UnsupportedMajorVersion {
        found: GpuIrVersion,
        expected_major: u16,
    },
    InvalidPixelBounds,
    InvalidSampleCount,
    EmptyTriangleMesh {
        geometry: GeometryId,
    },
    TriangleIndexOutOfBounds {
        geometry: GeometryId,
        index: GpuIndex,
    },
    DegenerateTriangle {
        geometry: GeometryId,
        triangle: GpuIndex,
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
        curve: GpuIndex,
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
pub struct GpuIrValidationErrors {
    issues: Box<[GpuIrValidationError]>,
}

impl GpuIrValidationErrors {
    pub fn issues(&self) -> &[GpuIrValidationError] {
        &self.issues
    }
}

impl GpuSceneDraft {
    pub fn finish(self) -> Result<GpuSceneIr, GpuIrValidationErrors> {
        let mut issues = Vec::new();
        if self.version.major != CURRENT_IR_VERSION.major {
            issues.push(GpuIrValidationError::UnsupportedMajorVersion {
                found: self.version,
                expected_major: CURRENT_IR_VERSION.major,
            });
        }
        if self.data.render.film.pixel_bounds.area().is_none() {
            issues.push(GpuIrValidationError::InvalidPixelBounds);
        }
        if self.data.render.sampler.samples_per_pixel == 0 {
            issues.push(GpuIrValidationError::InvalidSampleCount);
        }
        for (geometry_index, geometry) in self.data.geometry.iter().enumerate() {
            let geometry_id = GeometryId(geometry_index as GpuIndex);
            match geometry {
                GpuGeometry::TriangleMesh(mesh) => {
                    validate_triangle_mesh(geometry_id, mesh, &mut issues)
                }
                GpuGeometry::BilinearPatchMesh(mesh) => {
                    validate_bilinear_mesh(geometry_id, mesh, &mut issues)
                }
                GpuGeometry::CurveMesh(mesh) => validate_curve_mesh(geometry_id, mesh, &mut issues),
                GpuGeometry::Quadric(quadric) => {
                    validate_quadric(geometry_id, quadric, &mut issues)
                }
                GpuGeometry::DisplacedTriangleMesh(mesh) => {
                    validate_displaced_mesh(geometry_id, mesh, &self.data, &mut issues)
                }
            }
        }
        for (primitive_index, primitive) in self.data.primitives.iter().enumerate() {
            let primitive_id = PrimitiveId(primitive_index as GpuIndex);
            if usize::try_from(primitive.geometry.0)
                .ok()
                .and_then(|index| self.data.geometry.get(index))
                .is_none()
            {
                issues.push(GpuIrValidationError::InvalidGeometryReference {
                    primitive: primitive_id,
                    geometry: primitive.geometry,
                });
            }
            if usize::try_from(primitive.transform.0)
                .ok()
                .and_then(|index| self.data.transforms.get(index))
                .is_none()
            {
                issues.push(GpuIrValidationError::InvalidTransformReference {
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
                    issues.push(GpuIrValidationError::InvalidMaterialReference {
                        primitive: primitive_id,
                        material,
                    });
                }
            }
            for texture in [primitive.alpha, primitive.shadow_alpha]
                .into_iter()
                .flatten()
            {
                if usize::try_from(texture.0)
                    .ok()
                    .and_then(|index| self.data.float_textures.get(index))
                    .is_none()
                {
                    issues.push(GpuIrValidationError::InvalidFloatTextureReference {
                        primitive: primitive_id,
                        texture,
                    });
                }
            }
            let area_lights = match &primitive.area_light {
                GpuAreaLightBinding::None => &[][..],
                GpuAreaLightBinding::Uniform(light) => std::slice::from_ref(light),
                GpuAreaLightBinding::PerElement(lights) => lights,
            };
            for light in area_lights {
                if usize::try_from(light.0)
                    .ok()
                    .and_then(|index| self.data.lights.get(index))
                    .is_none()
                {
                    issues.push(GpuIrValidationError::InvalidAreaLightReference {
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
                issues.push(GpuIrValidationError::InvalidWorldPrimitiveReference {
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
                issues.push(GpuIrValidationError::InvalidWorldInstanceReference {
                    instance: *instance,
                });
            }
        }
        for (definition_index, definition) in self.data.instance_definitions.iter().enumerate() {
            let definition_id = InstanceDefinitionId(definition_index as GpuIndex);
            let valid_bounds = definition
                .local_bounds
                .min
                .0
                .iter()
                .zip(definition.local_bounds.max.0.iter())
                .all(|(min, max)| min.is_finite() && max.is_finite() && min <= max);
            if !valid_bounds {
                issues.push(GpuIrValidationError::InvalidInstanceBounds {
                    definition: definition_id,
                });
            }
            for primitive in &definition.primitives {
                if usize::try_from(primitive.0)
                    .ok()
                    .and_then(|index| self.data.primitives.get(index))
                    .is_none()
                {
                    issues.push(GpuIrValidationError::InvalidInstancePrimitiveReference {
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
                    issues.push(GpuIrValidationError::InvalidInstanceReference {
                        definition: definition_id,
                        instance: *instance,
                    });
                }
            }
        }
        for (instance_index, instance) in self.data.instances.iter().enumerate() {
            let instance_id = InstanceId(instance_index as GpuIndex);
            if usize::try_from(instance.definition.0)
                .ok()
                .and_then(|index| self.data.instance_definitions.get(index))
                .is_none()
            {
                issues.push(GpuIrValidationError::InvalidInstanceDefinitionReference {
                    instance: instance_id,
                    definition: instance.definition,
                });
            }
            if usize::try_from(instance.transform.0)
                .ok()
                .and_then(|index| self.data.transforms.get(index))
                .is_none()
            {
                issues.push(GpuIrValidationError::InvalidInstanceTransformReference {
                    instance: instance_id,
                    transform: instance.transform,
                });
            }
        }
        for (material_index, material) in self.data.materials.iter().enumerate() {
            let material_id = MaterialId(material_index as GpuIndex);
            let (texture, displacement, normal_map) = match material {
                GpuMaterial::Diffuse(diffuse) => (
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
                issues.push(GpuIrValidationError::InvalidSpectrumTextureReference {
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
                    issues.push(GpuIrValidationError::InvalidMaterialFloatTextureReference {
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
                    issues.push(GpuIrValidationError::InvalidImageReference { image: normal_map });
                }
            }
        }
        for (texture_index, texture) in self.data.spectrum_textures.iter().enumerate() {
            let texture_id = SpectrumTextureId(texture_index as GpuIndex);
            match texture {
                GpuSpectrumTexture::Constant { value } => {
                    if usize::try_from(value.0)
                        .ok()
                        .and_then(|index| self.data.spectra.get(index))
                        .is_none()
                    {
                        issues.push(GpuIrValidationError::InvalidTextureSpectrumReference {
                            texture: texture_id,
                            spectrum: *value,
                        });
                    }
                }
                GpuSpectrumTexture::Image { image, mapping, .. } => {
                    validate_image_texture_refs(*image, *mapping, &self.data, &mut issues)
                }
            }
        }
        for texture in &self.data.float_textures {
            if let GpuFloatTexture::Image { image, mapping, .. } = texture {
                validate_image_texture_refs(*image, *mapping, &self.data, &mut issues);
            }
        }
        for (light_index, light) in self.data.lights.iter().enumerate() {
            let light_id = LightId(light_index as GpuIndex);
            let (transform, spectrum, texture) = match light {
                GpuLight::Point(point) => {
                    (Some(point.render_from_light), Some(point.intensity), None)
                }
                GpuLight::DiffuseArea(area) => (None, None, Some(area.emission)),
                GpuLight::UniformInfinite(infinite) => (None, Some(infinite.radiance), None),
            };
            if let Some(transform) = transform {
                if usize::try_from(transform.0)
                    .ok()
                    .and_then(|index| self.data.transforms.get(index))
                    .is_none()
                {
                    issues.push(GpuIrValidationError::InvalidLightTransformReference {
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
                    issues.push(GpuIrValidationError::InvalidLightSpectrumReference {
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
                    issues.push(GpuIrValidationError::InvalidLightTextureReference {
                        light: light_id,
                        texture,
                    });
                }
            }
        }
        if issues.is_empty() {
            Ok(GpuSceneIr {
                version: self.version,
                data: self.data,
            })
        } else {
            Err(GpuIrValidationErrors {
                issues: issues.into_boxed_slice(),
            })
        }
    }
}

fn validate_triangle_mesh(
    geometry: GeometryId,
    mesh: &GpuTriangleMesh,
    issues: &mut Vec<GpuIrValidationError>,
) {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        issues.push(GpuIrValidationError::EmptyTriangleMesh { geometry });
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
            issues.push(GpuIrValidationError::TriangleIndexOutOfBounds { geometry, index });
        }
        if triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[2] == triangle[0] {
            issues.push(GpuIrValidationError::DegenerateTriangle {
                geometry,
                triangle: triangle_index as GpuIndex,
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
        issues.push(GpuIrValidationError::AttributeLengthMismatch { geometry });
    }
}

fn validate_bilinear_mesh(
    geometry: GeometryId,
    mesh: &GpuBilinearPatchMesh,
    issues: &mut Vec<GpuIrValidationError>,
) {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        issues.push(GpuIrValidationError::EmptyTriangleMesh { geometry });
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
            issues.push(GpuIrValidationError::TriangleIndexOutOfBounds { geometry, index });
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
        issues.push(GpuIrValidationError::AttributeLengthMismatch { geometry });
    }
}

fn validate_curve_mesh(
    geometry: GeometryId,
    mesh: &GpuCurveMesh,
    issues: &mut Vec<GpuIrValidationError>,
) {
    if mesh.curves.is_empty() {
        issues.push(GpuIrValidationError::InvalidCurve { geometry, curve: 0 });
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
            GpuCurveType::Ribbon => curve.endpoint_normals.is_some(),
            GpuCurveType::Flat | GpuCurveType::Cylinder => curve.endpoint_normals.is_none(),
        };
        if !valid_points || !valid_widths || !valid_normals || !type_valid {
            issues.push(GpuIrValidationError::InvalidCurve {
                geometry,
                curve: curve_index as GpuIndex,
            });
        }
    }
}

fn validate_quadric(
    geometry: GeometryId,
    quadric: &GpuQuadric,
    issues: &mut Vec<GpuIrValidationError>,
) {
    let valid = match quadric {
        GpuQuadric::Sphere {
            radius,
            z_min,
            z_max,
            phi_max_radians,
        }
        | GpuQuadric::Cylinder {
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
        GpuQuadric::Disk {
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
        issues.push(GpuIrValidationError::InvalidQuadric { geometry });
    }
}

fn validate_displaced_mesh(
    geometry: GeometryId,
    mesh: &GpuDisplacedTriangleMesh,
    data: &GpuSceneData,
    issues: &mut Vec<GpuIrValidationError>,
) {
    if !matches!(
        usize::try_from(mesh.base_mesh.0)
            .ok()
            .and_then(|index| data.geometry.get(index)),
        Some(GpuGeometry::TriangleMesh(_))
    ) {
        issues.push(GpuIrValidationError::InvalidDisplacementBase { geometry });
    }
    if usize::try_from(mesh.displacement.0)
        .ok()
        .and_then(|index| data.float_textures.get(index))
        .is_none()
    {
        issues.push(GpuIrValidationError::InvalidDisplacementTexture {
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
        issues.push(GpuIrValidationError::InvalidDisplacementData { geometry });
    }
}

fn validate_minmax_graph(nodes: &[GpuMinMaxNode]) -> bool {
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

fn visit_minmax_node(index: usize, nodes: &[GpuMinMaxNode], marks: &mut [u8]) -> bool {
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

impl GpuSceneIr {
    pub fn requirements(&self) -> GpuRequirements {
        use std::collections::BTreeSet;

        let data = &self.data;
        let mut features = BTreeSet::new();
        for transform in &data.transforms {
            features.insert(match transform {
                GpuTransform::Static(_) => GpuFeature::StaticTransform,
                GpuTransform::Animated(_) => GpuFeature::AnimatedTransform,
            });
        }
        for geometry in &data.geometry {
            features.insert(match geometry {
                GpuGeometry::TriangleMesh(_) => GpuFeature::TriangleMesh,
                GpuGeometry::BilinearPatchMesh(_) => GpuFeature::BilinearPatch,
                GpuGeometry::CurveMesh(_) => GpuFeature::Curve,
                GpuGeometry::Quadric(_) => GpuFeature::Quadric,
                GpuGeometry::DisplacedTriangleMesh(_) => GpuFeature::DisplacedTriangle,
            });
        }
        for texture in &data.float_textures {
            features.insert(match texture {
                GpuFloatTexture::Constant { .. } => GpuFeature::FloatConstantTexture,
                GpuFloatTexture::Image { .. } => GpuFeature::FloatImageTexture,
            });
        }
        for texture in &data.spectrum_textures {
            features.insert(match texture {
                GpuSpectrumTexture::Constant { .. } => GpuFeature::SpectrumConstantTexture,
                GpuSpectrumTexture::Image { .. } => GpuFeature::SpectrumImageTexture,
            });
        }
        for material in &data.materials {
            if matches!(material, GpuMaterial::Diffuse(_)) {
                features.insert(GpuFeature::DiffuseMaterial);
            }
        }
        for light in &data.lights {
            features.insert(match light {
                GpuLight::Point(_) => GpuFeature::PointLight,
                GpuLight::DiffuseArea(_) => GpuFeature::DiffuseAreaLight,
                GpuLight::UniformInfinite(_) => GpuFeature::UniformInfiniteLight,
            });
        }
        features.extend([
            GpuFeature::PerspectiveCamera,
            GpuFeature::IndependentSampler,
            GpuFeature::RgbFilm,
            GpuFeature::BoxFilter,
            GpuFeature::WavefrontVolPath,
            GpuFeature::UniformLightSampler,
        ]);

        let resource_counts = GpuResourceCounts {
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
        let maxima = GpuSemanticMaxima {
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
                    GpuGeometry::TriangleMesh(mesh) => mesh.positions.len() as u64,
                    GpuGeometry::BilinearPatchMesh(mesh) => mesh.positions.len() as u64,
                    GpuGeometry::CurveMesh(mesh) => (mesh.curves.len() * 4) as u64,
                    GpuGeometry::Quadric(_) => 0,
                    GpuGeometry::DisplacedTriangleMesh(mesh) => {
                        mesh.displaced_bounds_object.len() as u64
                    }
                })
                .max()
                .unwrap_or(0),
            elements_per_geometry: data
                .geometry
                .iter()
                .map(|geometry| match geometry {
                    GpuGeometry::TriangleMesh(mesh) => mesh.indices.len() as u64,
                    GpuGeometry::BilinearPatchMesh(mesh) => mesh.indices.len() as u64,
                    GpuGeometry::CurveMesh(mesh) => mesh.curves.len() as u64,
                    GpuGeometry::Quadric(_) => 1,
                    GpuGeometry::DisplacedTriangleMesh(mesh) => mesh.triangle_roots.len() as u64,
                })
                .max()
                .unwrap_or(0),
        };
        GpuRequirements {
            features: features
                .into_iter()
                .map(|feature| GpuRequiredFeature {
                    feature,
                    sources: Box::new([]),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            resource_counts,
            maxima,
        }
    }

    pub fn view(&self) -> GpuSceneView<'_> {
        GpuSceneView {
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
    data: &GpuSceneData,
    issues: &mut Vec<GpuIrValidationError>,
) {
    let Some(image_resource) = usize::try_from(image.0)
        .ok()
        .and_then(|index| data.images.get(index))
    else {
        issues.push(GpuIrValidationError::InvalidImageReference { image });
        return;
    };
    if usize::try_from(mapping.0)
        .ok()
        .and_then(|index| data.texture_mappings.get(index))
        .is_none()
    {
        issues.push(GpuIrValidationError::InvalidTextureMappingReference { mapping });
    }
    let (storage_len, storage_finite) = match &image_resource.storage {
        GpuTexelStorage::U8(values) => (values.len(), true),
        GpuTexelStorage::F16(values) => (
            values.len(),
            values
                .iter()
                .all(|value| half::f16::from_bits(*value).to_f32().is_finite()),
        ),
        GpuTexelStorage::F32(values) => {
            (values.len(), values.iter().all(|value| value.is_finite()))
        }
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
        GpuColorEncoding::Linear | GpuColorEncoding::Srgb => true,
        GpuColorEncoding::Gamma { exponent } => exponent.is_finite() && exponent > 0.0,
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
        issues.push(GpuIrValidationError::InvalidImageData { image });
    }
}
