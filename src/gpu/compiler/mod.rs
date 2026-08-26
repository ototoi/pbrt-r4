//! Host-side construction boundary for GPU IR.

use super::ir::{
    AnimatedTransform, BilinearPatchMesh, Bounds2, Bounds2i, Bounds3, BoxFilter, CurveMesh,
    CurveSegment, CurveType, DiffuseAreaLight, DiffuseMaterial, DisplacedTriangleMesh, Feature,
    Float, FloatTexture, Geometry, GeometryId, ImageChannels, ImageFilter, ImageResource,
    ImageWrapMode, IndependentSampler, Index, Instance, InstanceDefinition, InstanceDefinitionId,
    InstanceId, IrValidationErrors, Light, LightSampler, Material, MaterialId, Matrix3x3,
    Matrix4x4, MinMaxNodeId, Normal3, PerspectiveCamera, Point2, Point3, PointLight, Primitive,
    Quadric, RenderConfig, RgbFilm, SceneData, SceneDraft, SceneIr, SceneView, SourceId,
    SpectrumId, SpectrumResource, SpectrumTexture, StaticTransform, TextureMapping, Transform,
    TransformId, TriangleMesh, UniformInfiniteLight, Vector2, Vector3, WavefrontVolPath,
    CURRENT_IR_VERSION,
};
use crate::paramdict::ParameterDictionary;
use crate::parser::scene_builder::path_resolver::make_absolute_path;
use crate::parser::scene_builder::{
    LightSceneEntity, RenderFromObject, SceneBuilder, ShapeSceneEntity,
};
use crate::util::imageio::read_image::RawImageData;
use crate::util::imageio::{read_raw_image_with_encoding, ColorEncoding};
use crate::util::mesh::TriQuadMesh;
use crate::util::transform::{Matrix4x4 as CpuMatrix4x4, Transform as CpuTransform};
use std::path::Path;
use std::sync::Arc;

mod diagnostics;
mod light;
mod source_map;

pub use diagnostics::{GpuCompileError, GpuSourceLocation};
pub use source_map::{GpuResourceKind, GpuSourceEntry, GpuSourceMap};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuSceneBuildError {
    Compile(GpuCompileError),
    Validation(IrValidationErrors),
}

impl From<GpuCompileError> for GpuSceneBuildError {
    fn from(error: GpuCompileError) -> Self {
        Self::Compile(error)
    }
}

impl From<IrValidationErrors> for GpuSceneBuildError {
    fn from(errors: IrValidationErrors) -> Self {
        Self::Validation(errors)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuCompiledScene {
    ir: Arc<SceneIr>,
    source_map: Arc<GpuSourceMap>,
    requirements: Arc<super::ir::Requirements>,
}

impl GpuCompiledScene {
    pub fn new(ir: SceneIr, source_map: GpuSourceMap) -> Self {
        let requirements = ir.requirements();
        Self {
            ir: Arc::new(ir),
            source_map: Arc::new(source_map),
            requirements: Arc::new(requirements),
        }
    }

    fn with_source_map(
        ir: SceneIr,
        source_map: GpuSourceMap,
        requirements: super::ir::Requirements,
    ) -> Self {
        Self {
            ir: Arc::new(ir),
            source_map: Arc::new(source_map),
            requirements: Arc::new(requirements),
        }
    }

    pub fn scene(&self) -> &SceneIr {
        &self.ir
    }

    pub fn view(&self) -> SceneView<'_> {
        self.ir.view()
    }

    pub fn source_map(&self) -> &GpuSourceMap {
        &self.source_map
    }

    pub fn requirements(&self) -> super::ir::Requirements {
        (*self.requirements).clone()
    }
}

impl SceneBuilder {
    /// Compiles the currently supported subset of scene entities into the
    /// backend-independent GPU IR. Unsupported semantics are reported rather
    /// than silently omitted or delegated to the CPU renderer.
    pub fn build_gpu_ir(&self) -> Result<GpuCompiledScene, GpuSceneBuildError> {
        let mut transforms = Vec::new();
        let mut geometry = Vec::new();
        let mut spectra = vec![SpectrumResource::Constant { value: 0.5 }];
        let mut spectrum_textures = vec![SpectrumTexture::Constant {
            value: SpectrumId(0),
        }];
        let mut float_textures = Vec::new();
        let mut images = Vec::new();
        let mut texture_mappings = Vec::new();
        let (float_texture_ids, spectrum_texture_ids) = compile_textures(
            self,
            &mut spectra,
            &mut float_textures,
            &mut spectrum_textures,
            &mut images,
            &mut texture_mappings,
        )?;
        let mut materials = vec![Material::Diffuse(DiffuseMaterial {
            reflectance: super::ir::SpectrumTextureId(0),
            displacement: None,
            normal_map: None,
        })];
        let mut lights = Vec::new();
        let mut primitives = Vec::new();
        let mut instance_definitions = Vec::new();
        let mut instances = Vec::new();
        let mut world_primitives = Vec::new();
        let mut world_instances = Vec::new();

        for light in &self.lights {
            light::compile_light(self, light, &mut transforms, &mut spectra, &mut lights)?;
        }

        for shape in &self.shapes {
            let primitive = compile_shape(
                self,
                shape,
                &mut transforms,
                &mut geometry,
                &mut spectra,
                &mut spectrum_textures,
                &float_textures,
                &float_texture_ids,
                &spectrum_texture_ids,
                &mut images,
                &mut texture_mappings,
                &mut materials,
                &mut primitives,
                &mut lights,
            )?;
            world_primitives.push(primitive);
        }
        for shape in &self.animated_shapes {
            let primitive = compile_shape(
                self,
                shape,
                &mut transforms,
                &mut geometry,
                &mut spectra,
                &mut spectrum_textures,
                &float_textures,
                &float_texture_ids,
                &spectrum_texture_ids,
                &mut images,
                &mut texture_mappings,
                &mut materials,
                &mut primitives,
                &mut lights,
            )?;
            world_primitives.push(primitive);
        }

        let mut definitions: Vec<_> = self.instance_definitions.iter().collect();
        definitions.sort_by(|(left, _), (right, _)| left.cmp(right));
        for (name, definition) in definitions {
            let definition_id = InstanceDefinitionId(instance_definitions.len() as Index);
            let mut definition_primitives = Vec::new();
            for shape in definition.shapes.iter().chain(&definition.animated_shapes) {
                let primitive = compile_shape(
                    self,
                    shape,
                    &mut transforms,
                    &mut geometry,
                    &mut spectra,
                    &mut spectrum_textures,
                    &float_textures,
                    &float_texture_ids,
                    &spectrum_texture_ids,
                    &mut images,
                    &mut texture_mappings,
                    &mut materials,
                    &mut primitives,
                    &mut lights,
                )?;
                definition_primitives.push(primitive);
            }
            let local_bounds =
                bounds_for_primitives(&definition_primitives, &primitives, &geometry)?;
            instance_definitions.push(InstanceDefinition {
                primitives: definition_primitives,
                instances: Vec::new(),
                local_bounds,
            });
            for instance in self
                .instance_uses
                .iter()
                .filter(|instance| instance.name == *name)
            {
                let transform_id = compile_instance_transform(
                    &instance.render_from_instance,
                    &mut transforms,
                    &GpuSourceLocation {
                        filename: instance.loc.filename.clone(),
                        line: instance.loc.line,
                        column: instance.loc.column,
                    },
                )?;
                let instance_id = InstanceId(instances.len() as Index);
                instances.push(Instance {
                    definition: definition_id,
                    transform: transform_id,
                });
                world_instances.push(instance_id);
            }
        }

        let render = render_config(self, &mut transforms)?;
        let draft = SceneDraft {
            version: CURRENT_IR_VERSION,
            data: SceneData {
                transforms,
                spectra,
                float_textures,
                spectrum_textures,
                texture_mappings,
                images,
                geometry,
                materials,
                lights,
                primitives,
                instance_definitions,
                instances,
                world_primitives: world_primitives.into_boxed_slice(),
                world_instances: world_instances.into_boxed_slice(),
                render,
            },
        };
        let ir = draft.finish()?;
        let source_map = source_map(self, &ir);
        let mut requirements = ir.requirements();
        attach_requirement_sources(&mut requirements, ir.view(), &source_map);
        Ok(GpuCompiledScene::with_source_map(
            ir,
            source_map,
            requirements,
        ))
    }
}

struct CompiledImageTexture {
    image: super::ir::ImageId,
    mapping: super::ir::TextureMappingId,
    scale: Float,
    invert: bool,
    swrap: ImageWrapMode,
    twrap: ImageWrapMode,
    filter: ImageFilter,
    channel: super::ir::FloatImageChannel,
}

fn compile_image_texture(
    builder: &SceneBuilder,
    texture: &crate::parser::scene_builder::TextureSceneEntity,
    images: &mut Vec<ImageResource>,
    texture_mappings: &mut Vec<TextureMapping>,
) -> Result<CompiledImageTexture, GpuCompileError> {
    let source = texture_source_location(texture);
    let params = make_absolute_path(&texture.base.params, &builder.seen_work_dirs);
    let filename = params.get_one_filename("filename", "");
    if filename.is_empty() {
        return Err(invalid_parameter(
            "filename",
            "imagemap requires a non-empty filename",
            &source,
        ));
    }
    let encoding_name = params.get_one_string("encoding", "");
    let encoding = if encoding_name.is_empty() {
        if Path::new(&filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
        {
            ColorEncoding::parse("sRGB")
        } else {
            ColorEncoding::parse("linear")
        }
    } else {
        ColorEncoding::parse(&encoding_name)
    }
    .map_err(|error| GpuCompileError::InvalidParameter {
        parameter: "encoding",
        detail: error.msg,
        source: source.clone(),
    })?;
    let raw = read_raw_image_with_encoding(&filename, encoding).map_err(|error| {
        GpuCompileError::InvalidParameter {
            parameter: "filename",
            detail: error.msg,
            source: source.clone(),
        }
    })?;
    let width = u32::try_from(raw.resolution.x)
        .map_err(|_| invalid_parameter("filename", "image width must be positive", &source))?;
    let height = u32::try_from(raw.resolution.y)
        .map_err(|_| invalid_parameter("filename", "image height must be positive", &source))?;
    if width == 0 || height == 0 || raw.channels == 0 {
        return Err(invalid_parameter(
            "filename",
            "image resolution and channel count must be positive",
            &source,
        ));
    }
    let image = super::ir::ImageId(images.len() as Index);
    let pixels = raw.data_f32();
    let channels = image_channels(raw.channels, &source)?;
    let channel = float_image_channel(raw.channels, &pixels);
    let (storage, mip_levels, color_encoding) =
        build_image_storage(&raw, width, height, channels, &source)?;
    images.push(ImageResource {
        resolution: [width, height],
        channels,
        storage,
        mip_levels,
        color_encoding,
    });
    let mapping_name = params.get_one_string("mapping", "uv");
    let texture_from_render = matrix(texture.render_from_texture.m, &source)?;
    let mapping = match mapping_name.as_str() {
        "uv" => TextureMapping::Uv {
            su: finite_parameter(&params, "uscale", 1.0, &source)?,
            sv: finite_parameter(&params, "vscale", 1.0, &source)?,
            du: finite_parameter(&params, "udelta", 0.0, &source)?,
            dv: finite_parameter(&params, "vdelta", 0.0, &source)?,
        },
        "spherical" => TextureMapping::Spherical {
            texture_from_render,
        },
        "cylindrical" => TextureMapping::Cylindrical {
            texture_from_render,
        },
        "planar" => TextureMapping::Planar {
            texture_from_render,
            vs: vector3_parameter(&params, "v1", [1.0, 0.0, 0.0], &source)?,
            vt: vector3_parameter(&params, "v2", [0.0, 1.0, 0.0], &source)?,
            ds: finite_parameter(&params, "udelta", 0.0, &source)?,
            dt: finite_parameter(&params, "vdelta", 0.0, &source)?,
        },
        "3d" | "transform3d" => TextureMapping::Transform3D {
            texture_from_render,
        },
        _ => {
            return Err(invalid_parameter(
                "mapping",
                "unknown texture mapping",
                &source,
            ))
        }
    };
    let mapping_id = super::ir::TextureMappingId(texture_mappings.len() as Index);
    texture_mappings.push(mapping);
    let filter_name = params.get_one_string("filter", "bilinear");
    let filter = match filter_name.as_str() {
        "point" => ImageFilter::Point,
        "bilinear" => ImageFilter::Bilinear,
        "trilinear" => ImageFilter::Trilinear,
        "ewa" | "EWA" => {
            let max_anisotropy = finite_parameter(&params, "maxanisotropy", 8.0, &source)?;
            if max_anisotropy < 1.0 {
                return Err(invalid_parameter(
                    "maxanisotropy",
                    "EWA max anisotropy must be at least one",
                    &source,
                ));
            }
            ImageFilter::Ewa { max_anisotropy }
        }
        _ => return Err(invalid_parameter("filter", "unknown image filter", &source)),
    };
    let wrap = params.get_one_string("wrap", "repeat");
    let swrap_name = params.get_one_string("swrap", &wrap);
    let twrap_name = params.get_one_string("twrap", &wrap);
    let parse_wrap = |name: &str| match name {
        "black" => Ok(ImageWrapMode::Black),
        "clamp" => Ok(ImageWrapMode::Clamp),
        "repeat" => Ok(ImageWrapMode::Repeat),
        "octahedralsphere" => Ok(ImageWrapMode::OctahedralSphere),
        _ => Err(invalid_parameter(
            "wrap",
            "unknown image wrap mode",
            &source,
        )),
    };
    Ok(CompiledImageTexture {
        image,
        mapping: mapping_id,
        scale: finite_parameter(&params, "scale", 1.0, &source)?,
        invert: params.get_one_bool("invert", false),
        swrap: parse_wrap(&swrap_name)?,
        twrap: parse_wrap(&twrap_name)?,
        filter,
        channel,
    })
}

fn float_image_channel(channels: usize, pixels: &[Float]) -> super::ir::FloatImageChannel {
    match channels {
        1 | 2 => super::ir::FloatImageChannel::Channel0,
        4 if pixels.chunks_exact(4).any(|pixel| pixel[3] != 1.0) => {
            super::ir::FloatImageChannel::Alpha
        }
        _ => super::ir::FloatImageChannel::RgbAverage,
    }
}

fn image_channels(
    channels: usize,
    source: &GpuSourceLocation,
) -> Result<ImageChannels, GpuCompileError> {
    match channels {
        1 => Ok(ImageChannels::R),
        2 => Ok(ImageChannels::Rg),
        3 => Ok(ImageChannels::Rgb),
        4 => Ok(ImageChannels::Rgba),
        _ => Err(invalid_parameter(
            "filename",
            "GPU image supports only one to four channels",
            source,
        )),
    }
}

fn build_mip_storage(
    base: &[Float],
    width: u32,
    height: u32,
    channels: usize,
) -> (Vec<Float>, Box<[super::ir::MipLevel]>) {
    let mut storage = Vec::from(base);
    let mut levels = Vec::new();
    let mut level_width = width;
    let mut level_height = height;
    let mut level = base.to_vec();
    loop {
        let offset = storage.len() - level.len();
        levels.push(super::ir::MipLevel {
            resolution: [level_width, level_height],
            texel_offset: offset as u64,
            texel_count: level.len() as u64,
        });
        if level_width == 1 && level_height == 1 {
            break;
        }
        let next_width = (level_width / 2).max(1);
        let next_height = (level_height / 2).max(1);
        let mut next = vec![0.0; next_width as usize * next_height as usize * channels];
        for y in 0..next_height as usize {
            for x in 0..next_width as usize {
                for channel in 0..channels {
                    let mut sum = 0.0;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let sx = (2 * x + dx).min(level_width as usize - 1);
                            let sy = (2 * y + dy).min(level_height as usize - 1);
                            sum += level[(sy * level_width as usize + sx) * channels + channel];
                        }
                    }
                    next[(y * next_width as usize + x) * channels + channel] = sum / 4.0;
                }
            }
        }
        storage.extend_from_slice(&next);
        level = next;
        level_width = next_width;
        level_height = next_height;
    }
    (storage, levels.into_boxed_slice())
}

fn build_image_storage(
    raw: &crate::util::imageio::read_image::RawImage,
    width: u32,
    height: u32,
    channels: ImageChannels,
    source: &GpuSourceLocation,
) -> Result<
    (
        super::ir::TexelStorage,
        Box<[super::ir::MipLevel]>,
        super::ir::ColorEncoding,
    ),
    GpuCompileError,
> {
    let channel_count = channels.count();
    let linear_pixels = raw_linear_pixels(raw, source)?;
    let (linear_storage, mip_levels) =
        build_mip_storage(&linear_pixels, width, height, channel_count);
    let encoding = raw_encoding(raw);
    let storage = encode_mip_storage(&linear_storage, &raw.data, encoding, channel_count);
    Ok((storage, mip_levels, gpu_color_encoding(encoding)))
}

fn encode_mip_storage(
    linear_storage: &[Float],
    source_data: &RawImageData,
    encoding: ColorEncoding,
    channel_count: usize,
) -> super::ir::TexelStorage {
    match source_data {
        RawImageData::F32(_) => {
            super::ir::TexelStorage::F32(linear_storage.to_vec().into_boxed_slice())
        }
        RawImageData::F16(_) => super::ir::TexelStorage::F16(
            linear_storage
                .iter()
                .map(|value| half::f16::from_f32(*value).to_bits())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
        RawImageData::U8 { .. } => super::ir::TexelStorage::U8(
            linear_storage
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let channel = index % channel_count;
                    let encoded = if is_alpha_channel(channel, channel_count) {
                        *value
                    } else {
                        encoding.from_linear(*value)
                    };
                    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ),
    }
}

fn raw_linear_pixels(
    raw: &crate::util::imageio::read_image::RawImage,
    source: &GpuSourceLocation,
) -> Result<Vec<Float>, GpuCompileError> {
    let count = usize::try_from(raw.resolution.x)
        .ok()
        .and_then(|width| {
            usize::try_from(raw.resolution.y)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(raw.channels))
        .ok_or_else(|| invalid_parameter("filename", "image size is too large", source))?;
    let mut values = Vec::with_capacity(count);
    match &raw.data {
        RawImageData::F32(data) => values.extend_from_slice(data),
        RawImageData::F16(data) => values.extend(data.iter().map(|value| value.to_f32() as Float)),
        RawImageData::U8 { data, encoding } => {
            for (index, value) in data.iter().enumerate() {
                let normalized = *value as Float / 255.0;
                let channel = index % raw.channels;
                values.push(if is_alpha_channel(channel, raw.channels) {
                    normalized
                } else {
                    encoding.to_linear(normalized)
                });
            }
        }
    }
    if values.len() != count || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_parameter(
            "filename",
            "image texels must be finite and match the image dimensions",
            source,
        ));
    }
    Ok(values)
}

fn is_alpha_channel(channel: usize, channel_count: usize) -> bool {
    (channel_count == 2 || channel_count == 4) && channel + 1 == channel_count
}

fn raw_encoding(raw: &crate::util::imageio::read_image::RawImage) -> ColorEncoding {
    match &raw.data {
        RawImageData::U8 { encoding, .. } => *encoding,
        RawImageData::F16(_) | RawImageData::F32(_) => ColorEncoding::Linear,
    }
}

fn gpu_color_encoding(encoding: ColorEncoding) -> super::ir::ColorEncoding {
    match encoding {
        ColorEncoding::Linear => super::ir::ColorEncoding::Linear,
        ColorEncoding::SRgb => super::ir::ColorEncoding::Srgb,
        ColorEncoding::Gamma(exponent) => super::ir::ColorEncoding::Gamma { exponent },
    }
}

fn compile_textures(
    builder: &SceneBuilder,
    spectra: &mut Vec<SpectrumResource>,
    float_textures: &mut Vec<FloatTexture>,
    spectrum_textures: &mut Vec<SpectrumTexture>,
    images: &mut Vec<ImageResource>,
    texture_mappings: &mut Vec<TextureMapping>,
) -> Result<
    (
        Vec<Option<super::ir::FloatTextureId>>,
        Vec<Option<super::ir::SpectrumTextureId>>,
    ),
    GpuCompileError,
> {
    let mut float_ids = vec![None; builder.float_textures.len()];
    for (index, texture) in builder.float_textures.iter().enumerate() {
        let source = texture_source_location(texture);
        if texture.base.name == "imagemap" {
            let info = compile_image_texture(builder, texture, images, texture_mappings)?;
            let id = super::ir::FloatTextureId(float_textures.len() as Index);
            float_textures.push(FloatTexture::Image {
                image: info.image,
                mapping: info.mapping,
                scale: info.scale,
                invert: info.invert,
                swrap: info.swrap,
                twrap: info.twrap,
                filter: info.filter,
                channel: info.channel,
            });
            float_ids[index] = Some(id);
            continue;
        }
        if texture.base.name != "constant" {
            return Err(GpuCompileError::UnsupportedSceneFeature {
                feature: "non-constant float texture",
                source,
            });
        }
        let values = texture.base.params.get_floats("value");
        let value = match values.as_slice() {
            [value] => to_gpu_float(*value, &source)?,
            _ => {
                return Err(invalid_parameter(
                    "value",
                    "constant float texture requires exactly one value",
                    &source,
                ))
            }
        };
        let id = super::ir::FloatTextureId(float_textures.len() as Index);
        float_textures.push(FloatTexture::Constant { value });
        float_ids[index] = Some(id);
    }

    let mut spectrum_ids = vec![None; builder.spectrum_textures.len()];
    for (index, texture) in builder.spectrum_textures.iter().enumerate() {
        let source = texture_source_location(texture);
        if texture.base.name == "imagemap" {
            let info = compile_image_texture(builder, texture, images, texture_mappings)?;
            let id = super::ir::SpectrumTextureId(spectrum_textures.len() as Index);
            spectrum_textures.push(SpectrumTexture::Image {
                image: info.image,
                mapping: info.mapping,
                scale: info.scale,
                invert: info.invert,
                swrap: info.swrap,
                twrap: info.twrap,
                filter: info.filter,
                spectrum_type: super::ir::SpectrumType::Albedo,
            });
            spectrum_ids[index] = Some(id);
            continue;
        }
        if texture.base.name != "constant" {
            return Err(GpuCompileError::UnsupportedSceneFeature {
                feature: "non-constant spectrum texture",
                source,
            });
        }
        let values = texture.base.params.get_points("value");
        let coefficients = match values.as_slice() {
            [r, g, b] => [*r, *g, *b],
            _ => {
                return Err(invalid_parameter(
                    "value",
                    "constant spectrum texture requires exactly three RGB values",
                    &source,
                ))
            }
        };
        let spectrum = SpectrumId(spectra.len() as Index);
        spectra.push(SpectrumResource::RgbAlbedo {
            coefficients: [
                to_gpu_float(coefficients[0], &source)?,
                to_gpu_float(coefficients[1], &source)?,
                to_gpu_float(coefficients[2], &source)?,
            ],
        });
        let id = super::ir::SpectrumTextureId(spectrum_textures.len() as Index);
        spectrum_textures.push(SpectrumTexture::Constant { value: spectrum });
        spectrum_ids[index] = Some(id);
    }
    Ok((float_ids, spectrum_ids))
}

fn compile_shape(
    builder: &SceneBuilder,
    shape: &ShapeSceneEntity,
    transforms: &mut Vec<Transform>,
    geometry: &mut Vec<Geometry>,
    spectra: &mut Vec<SpectrumResource>,
    spectrum_textures: &mut Vec<SpectrumTexture>,
    float_textures: &[FloatTexture],
    float_texture_ids: &[Option<super::ir::FloatTextureId>],
    spectrum_texture_ids: &[Option<super::ir::SpectrumTextureId>],
    images: &mut Vec<ImageResource>,
    texture_mappings: &mut Vec<TextureMapping>,
    materials: &mut Vec<Material>,
    primitives: &mut Vec<Primitive>,
    lights: &mut Vec<Light>,
) -> Result<super::ir::PrimitiveId, GpuCompileError> {
    let source = source_location(shape);
    if !matches!(
        shape.base.name.as_str(),
        "trianglemesh"
            | "plymesh"
            | "bilinearmesh"
            | "curve"
            | "curves"
            | "sphere"
            | "cylinder"
            | "disk"
    ) {
        return Err(GpuCompileError::UnsupportedShape {
            name: shape.base.name.clone(),
            source,
        });
    }
    if !shape.child_params.is_empty() && shape.base.name != "curve" && shape.base.name != "curves" {
        return Err(unsupported_feature(shape, "grouped child shapes"));
    }
    if !shape.medium_interface.is_empty()
        || !shape.material_is_default
            && shape.material_name.is_none()
            && shape.material_index == usize::MAX
    {
        return Err(unsupported_feature(
            shape,
            "material, area light, or medium binding",
        ));
    }
    if shape.base.params.get_textures_ref("displacement").is_some() && shape.base.name != "plymesh"
    {
        return Err(unsupported_feature(
            shape,
            "shape displacement on non-PLY geometry",
        ));
    }
    let transform_id = TransformId(transforms.len() as Index);
    transforms.push(compile_transform(&shape.render_from_object, &source)?);
    let material = compile_material(
        builder,
        shape,
        spectra,
        spectrum_textures,
        float_texture_ids,
        spectrum_texture_ids,
        images,
        materials,
    )?;
    let area_light = shape
        .area_light_index
        .map(|index| {
            let area = builder.area_lights.get(index).ok_or_else(|| {
                invalid_parameter(
                    "AreaLightSource",
                    "area light index is out of range",
                    &source,
                )
            })?;
            light::compile_area_light(
                builder,
                area,
                spectra,
                spectrum_textures,
                images,
                texture_mappings,
                lights,
                &source,
            )
        })
        .transpose()?;
    let alpha = material_texture_id(
        builder,
        shape
            .base
            .params
            .get_textures_ref("alpha")
            .as_deref()
            .map(|names| &**names),
        float_texture_ids,
        "alpha",
        &source,
    )?;
    let geometry_id = if shape.base.name == "plymesh" {
        let base_mesh = ply_mesh(builder, &shape.base.params, &source)?;
        if shape.base.params.get_textures_ref("displacement").is_some() {
            let base_id = GeometryId(geometry.len() as Index);
            geometry.push(Geometry::TriangleMesh(base_mesh.clone()));
            let displaced_id = GeometryId(geometry.len() as Index);
            let displaced = displaced_ply_mesh(
                builder,
                &shape.base.params,
                base_id,
                &base_mesh,
                float_textures,
                float_texture_ids,
                images,
                &source,
            )?;
            geometry.push(Geometry::DisplacedTriangleMesh(displaced));
            displaced_id
        } else {
            let id = GeometryId(geometry.len() as Index);
            geometry.push(Geometry::TriangleMesh(base_mesh));
            id
        }
    } else {
        let id = GeometryId(geometry.len() as Index);
        let shape_geometry = match shape.base.name.as_str() {
            "trianglemesh" => Geometry::TriangleMesh(triangle_mesh(&shape.base.params, &source)?),
            "bilinearmesh" => {
                Geometry::BilinearPatchMesh(bilinear_mesh(&shape.base.params, &source)?)
            }
            "curve" | "curves" => Geometry::CurveMesh(curve_mesh(
                &shape.base.params,
                &shape.child_params,
                &source,
            )?),
            "sphere" => Geometry::Quadric(quadric(&shape.base.params, "sphere", &source)?),
            "cylinder" => Geometry::Quadric(quadric(&shape.base.params, "cylinder", &source)?),
            "disk" => Geometry::Quadric(quadric(&shape.base.params, "disk", &source)?),
            _ => unreachable!("shape name was checked above"),
        };
        geometry.push(shape_geometry);
        id
    };
    let primitive_id = super::ir::PrimitiveId(primitives.len() as Index);
    primitives.push(Primitive {
        geometry: geometry_id,
        transform: transform_id,
        material: Some(material),
        alpha,
        area_light: area_light.map_or(
            super::ir::AreaLightBinding::None,
            super::ir::AreaLightBinding::Uniform,
        ),
        reverse_orientation: shape.reverse_orientation,
    });
    Ok(primitive_id)
}

fn ply_mesh(
    builder: &SceneBuilder,
    params: &ParameterDictionary,
    source: &GpuSourceLocation,
) -> Result<TriangleMesh, GpuCompileError> {
    let params = make_absolute_path(params, &builder.seen_work_dirs);
    let filename = params.get_one_string("filename", "");
    if filename.is_empty() {
        return Err(invalid_parameter(
            "filename",
            "plymesh requires a non-empty filename",
            source,
        ));
    }
    let mesh =
        TriQuadMesh::read_ply(&filename).map_err(|error| GpuCompileError::InvalidParameter {
            parameter: "filename",
            detail: error.msg,
            source: source.clone(),
        })?;
    if !mesh.quad_indices.is_empty() {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "PLY quad faces in GPU geometry",
            source: source.clone(),
        });
    }
    if mesh.tri_indices.is_empty() {
        return Err(invalid_parameter(
            "filename",
            "PLY mesh must contain triangle faces",
            source,
        ));
    }
    let positions = mesh
        .p
        .iter()
        .map(|point| {
            Ok(Point3([
                to_gpu_float(point[0], source)?,
                to_gpu_float(point[1], source)?,
                to_gpu_float(point[2], source)?,
            ]))
        })
        .collect::<Result<Vec<_>, GpuCompileError>>()?;
    let indices = mesh
        .tri_indices
        .chunks_exact(3)
        .map(|triangle| [triangle[0], triangle[1], triangle[2]])
        .collect::<Vec<_>>();
    let normals = if mesh.n.len() == mesh.p.len() {
        Some(
            mesh.n
                .iter()
                .map(|normal| {
                    Ok(Normal3([
                        to_gpu_float(normal[0], source)?,
                        to_gpu_float(normal[1], source)?,
                        to_gpu_float(normal[2], source)?,
                    ]))
                })
                .collect::<Result<Vec<_>, GpuCompileError>>()?,
        )
    } else {
        None
    };
    let uvs = if mesh.uv.len() == mesh.p.len() {
        Some(
            mesh.uv
                .iter()
                .map(|uv| {
                    Ok(Point2([
                        to_gpu_float(uv[0], source)?,
                        to_gpu_float(uv[1], source)?,
                    ]))
                })
                .collect::<Result<Vec<_>, GpuCompileError>>()?,
        )
    } else {
        None
    };
    let face_indices = if mesh.face_indices.len() == indices.len() {
        Some(
            mesh.face_indices
                .iter()
                .map(|index| {
                    Index::try_from(*index)
                        .map_err(|_| invalid_parameter("face_indices", "negative index", source))
                })
                .collect::<Result<Vec<_>, GpuCompileError>>()?,
        )
    } else {
        None
    };
    Ok(TriangleMesh {
        positions,
        indices,
        normals,
        tangents: None,
        uvs,
        face_indices,
    })
}

fn displaced_ply_mesh(
    builder: &SceneBuilder,
    params: &ParameterDictionary,
    base_mesh_id: GeometryId,
    base_mesh: &TriangleMesh,
    float_textures: &[FloatTexture],
    float_texture_ids: &[Option<super::ir::FloatTextureId>],
    images: &[ImageResource],
    source: &GpuSourceLocation,
) -> Result<DisplacedTriangleMesh, GpuCompileError> {
    let texture_name = params
        .get_textures_ref("displacement")
        .and_then(|names| names.first().cloned())
        .ok_or_else(|| invalid_parameter("displacement", "texture reference is empty", source))?;
    let texture_index = builder
        .named_float_textures
        .get(&texture_name)
        .copied()
        .ok_or_else(|| invalid_parameter("displacement", "texture is not declared", source))?;
    let displacement = float_texture_ids
        .get(texture_index)
        .copied()
        .flatten()
        .ok_or_else(|| invalid_parameter("displacement", "texture is not supported", source))?;
    let (displacement_min, displacement_max) =
        float_texture_bounds(displacement, float_textures, images, source)?;
    let uvs = base_mesh.uvs.as_ref().ok_or_else(|| {
        invalid_parameter(
            "displacement",
            "shape displacement requires vertex UVs",
            source,
        )
    })?;
    let mut min_max_nodes = Vec::with_capacity(base_mesh.indices.len());
    let mut triangle_roots = Vec::with_capacity(base_mesh.indices.len());
    for triangle in &base_mesh.indices {
        let uv0 = uvs[triangle[0] as usize].0;
        let uv1 = uvs[triangle[1] as usize].0;
        let uv2 = uvs[triangle[2] as usize].0;
        let parameter_bounds = Bounds2 {
            min: Point2([
                uv0[0].min(uv1[0]).min(uv2[0]),
                uv0[1].min(uv1[1]).min(uv2[1]),
            ]),
            max: Point2([
                uv0[0].max(uv1[0]).max(uv2[0]),
                uv0[1].max(uv1[1]).max(uv2[1]),
            ]),
        };
        if parameter_bounds
            .min
            .0
            .iter()
            .chain(parameter_bounds.max.0.iter())
            .any(|value| !value.is_finite())
        {
            return Err(invalid_parameter(
                "displacement",
                "displacement UVs must be finite",
                source,
            ));
        }
        let root = MinMaxNodeId(min_max_nodes.len() as Index);
        min_max_nodes.push(super::ir::MinMaxNode {
            parameter_bounds,
            displacement_min,
            displacement_max,
            children: None,
        });
        triangle_roots.push(root);
    }
    let edge_length = params.get_one_float("edgelength", 1.0);
    if !edge_length.is_finite() || edge_length <= 0.0 {
        return Err(invalid_parameter(
            "edgelength",
            "displacement edge length must be positive",
            source,
        ));
    }
    let extent = displacement_min.abs().max(displacement_max.abs());
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for point in &base_mesh.positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(point.0[axis] - extent);
            max[axis] = max[axis].max(point.0[axis] + extent);
        }
    }
    Ok(DisplacedTriangleMesh {
        base_mesh: base_mesh_id,
        displacement,
        displacement_scale: 1.0,
        displacement_offset: 0.0,
        edge_length,
        min_max_nodes: min_max_nodes.into_boxed_slice(),
        triangle_roots: triangle_roots.into_boxed_slice(),
        displaced_bounds_object: vec![
            Bounds3 {
                min: Point3(min),
                max: Point3(max),
            };
            base_mesh.indices.len()
        ]
        .into_boxed_slice(),
    })
}

fn float_texture_bounds(
    texture_id: super::ir::FloatTextureId,
    float_textures: &[FloatTexture],
    images: &[ImageResource],
    source: &GpuSourceLocation,
) -> Result<(Float, Float), GpuCompileError> {
    let texture = float_textures
        .get(texture_id.0 as usize)
        .ok_or_else(|| invalid_parameter("displacement", "texture ID is invalid", source))?;
    let (mut min, mut max) = match texture {
        FloatTexture::Constant { value } => (*value, *value),
        FloatTexture::Image {
            image,
            channel,
            scale,
            invert,
            ..
        } => {
            let image = images
                .get(image.0 as usize)
                .ok_or_else(|| invalid_parameter("displacement", "image ID is invalid", source))?;
            let values = image_float_channel_values(image, *channel, source)?;
            let (mut min, mut max) =
                values
                    .into_iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
                        let value = if *invert { 1.0 - value } else { value };
                        (min.min(value), max.max(value))
                    });
            if *scale >= 0.0 {
                min *= *scale;
                max *= *scale;
            } else {
                let old_min = min;
                min = max * *scale;
                max = old_min * *scale;
            }
            (min, max)
        }
    };
    if !min.is_finite() || !max.is_finite() {
        return Err(invalid_parameter(
            "displacement",
            "displacement range must be finite",
            source,
        ));
    }
    if min > max {
        std::mem::swap(&mut min, &mut max);
    }
    Ok((min, max))
}

fn image_float_channel_values(
    image: &ImageResource,
    channel: super::ir::FloatImageChannel,
    source: &GpuSourceLocation,
) -> Result<Vec<Float>, GpuCompileError> {
    let channel_count = image.channels.count();
    let selected = match channel {
        super::ir::FloatImageChannel::Channel0 => 0,
        super::ir::FloatImageChannel::Alpha => {
            if channel_count != 2 && channel_count != 4 {
                return Err(invalid_parameter(
                    "displacement",
                    "alpha channel is unavailable",
                    source,
                ));
            }
            channel_count - 1
        }
        super::ir::FloatImageChannel::RgbAverage => {
            if channel_count < 3 {
                return Err(invalid_parameter(
                    "displacement",
                    "RGB channels are unavailable",
                    source,
                ));
            }
            0
        }
    };
    let mut values = Vec::new();
    let mut push_pixel = |pixel: &[Float]| {
        values.push(
            if matches!(channel, super::ir::FloatImageChannel::RgbAverage) {
                (pixel[0] + pixel[1] + pixel[2]) / 3.0
            } else {
                pixel[selected]
            },
        );
    };
    match &image.storage {
        super::ir::TexelStorage::F32(data) => {
            for pixel in data.chunks_exact(channel_count) {
                push_pixel(pixel);
            }
        }
        super::ir::TexelStorage::F16(data) => {
            for pixel in data.chunks_exact(channel_count) {
                let values_f32 = pixel
                    .iter()
                    .map(|value| half::f16::from_bits(*value).to_f32() as Float)
                    .collect::<Vec<_>>();
                push_pixel(&values_f32);
            }
        }
        super::ir::TexelStorage::U8(data) => {
            for pixel in data.chunks_exact(channel_count) {
                let values_f32 = pixel
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        let normalized = *value as Float / 255.0;
                        if is_alpha_channel(index, channel_count) {
                            normalized
                        } else {
                            match image.color_encoding {
                                super::ir::ColorEncoding::Linear => normalized,
                                super::ir::ColorEncoding::Srgb => {
                                    ColorEncoding::SRgb.to_linear(normalized)
                                }
                                super::ir::ColorEncoding::Gamma { exponent } => {
                                    normalized.powf(exponent)
                                }
                            }
                        }
                    })
                    .collect::<Vec<_>>();
                push_pixel(&values_f32);
            }
        }
    }
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_parameter(
            "displacement",
            "displacement image has no finite texels",
            source,
        ));
    }
    Ok(values)
}

fn curve_mesh(
    params: &ParameterDictionary,
    child_params: &[ParameterDictionary],
    source: &GpuSourceLocation,
) -> Result<CurveMesh, GpuCompileError> {
    let grouped = params.get_points("P").is_empty();
    let first_params = if grouped {
        child_params
            .first()
            .ok_or_else(|| invalid_parameter("P", "curves requires at least one curve", source))?
    } else {
        params
    };
    let curve_type = match first_params.get_one_string("type", "flat").as_str() {
        "flat" => CurveType::Flat,
        "cylinder" => CurveType::Cylinder,
        "ribbon" => CurveType::Ribbon,
        _ => return Err(invalid_parameter("type", "unknown curve type", source)),
    };
    let mut curves = Vec::with_capacity(child_params.len() + 1);
    if grouped {
        for params in child_params {
            curves.push(curve_segment(params, curve_type, source)?);
        }
    } else {
        curves.push(curve_segment(params, curve_type, source)?);
        for params in child_params {
            curves.push(curve_segment(params, curve_type, source)?);
        }
    }
    Ok(CurveMesh { curve_type, curves })
}

fn curve_segment(
    params: &ParameterDictionary,
    curve_type: CurveType,
    source: &GpuSourceLocation,
) -> Result<CurveSegment, GpuCompileError> {
    let points = params.get_points("P");
    if points.len() != 12 {
        return Err(invalid_parameter(
            "P",
            "curve requires exactly four control points",
            source,
        ));
    }
    let mut control_points = [Point3([0.0; 3]); 4];
    for (point, values) in control_points.iter_mut().zip(points.chunks_exact(3)) {
        *point = Point3([
            to_gpu_float(values[0], source)?,
            to_gpu_float(values[1], source)?,
            to_gpu_float(values[2], source)?,
        ]);
    }
    let widths = params.get_floats("width");
    if widths.len() != 2 {
        return Err(invalid_parameter(
            "width",
            "curve requires two endpoint widths",
            source,
        ));
    }
    let widths = [
        to_gpu_float(widths[0], source)?,
        to_gpu_float(widths[1], source)?,
    ];
    if widths.iter().any(|width| *width < 0.0) {
        return Err(invalid_parameter(
            "width",
            "curve widths must be non-negative",
            source,
        ));
    }
    let raw_normals = params.get_points("N");
    let endpoint_normals = match curve_type {
        CurveType::Ribbon => {
            if raw_normals.len() != 6 {
                return Err(invalid_parameter(
                    "N",
                    "ribbon curve requires two endpoint normals",
                    source,
                ));
            }
            Some([
                Normal3([
                    to_gpu_float(raw_normals[0], source)?,
                    to_gpu_float(raw_normals[1], source)?,
                    to_gpu_float(raw_normals[2], source)?,
                ]),
                Normal3([
                    to_gpu_float(raw_normals[3], source)?,
                    to_gpu_float(raw_normals[4], source)?,
                    to_gpu_float(raw_normals[5], source)?,
                ]),
            ])
        }
        CurveType::Flat | CurveType::Cylinder => {
            if !raw_normals.is_empty() {
                return Err(invalid_parameter(
                    "N",
                    "flat and cylinder curves do not use endpoint normals",
                    source,
                ));
            }
            None
        }
    };
    Ok(CurveSegment {
        control_points,
        widths,
        endpoint_normals,
    })
}

fn compile_instance_transform(
    render_from_instance: &RenderFromObject,
    transforms: &mut Vec<Transform>,
    source: &GpuSourceLocation,
) -> Result<TransformId, GpuCompileError> {
    let transform = compile_transform(render_from_instance, source)?;
    let id = TransformId(transforms.len() as Index);
    transforms.push(transform);
    Ok(id)
}

fn compile_transform(
    render_from_object: &RenderFromObject,
    source: &GpuSourceLocation,
) -> Result<Transform, GpuCompileError> {
    Ok(match render_from_object {
        RenderFromObject::Static(transform) => {
            Transform::Static(static_transform(transform, source)?)
        }
        RenderFromObject::Animated {
            from,
            to,
            start_time,
            end_time,
        } => Transform::Animated(AnimatedTransform {
            start: static_transform(from, source)?,
            end: static_transform(to, source)?,
            start_time: to_gpu_float(*start_time, source)?,
            end_time: to_gpu_float(*end_time, source)?,
        }),
    })
}

fn bounds_for_primitives(
    primitive_ids: &[super::ir::PrimitiveId],
    primitives: &[Primitive],
    geometry: &[Geometry],
) -> Result<Bounds3, GpuCompileError> {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for primitive_id in primitive_ids {
        let primitive = primitives.get(primitive_id.0 as usize).ok_or_else(|| {
            GpuCompileError::InvalidParameter {
                parameter: "ObjectBegin",
                detail: "compiled primitive reference is out of bounds".to_owned(),
                source: empty_source(),
            }
        })?;
        let shape_geometry = geometry.get(primitive.geometry.0 as usize).ok_or_else(|| {
            GpuCompileError::InvalidParameter {
                parameter: "Shape",
                detail: "compiled geometry reference is out of bounds".to_owned(),
                source: empty_source(),
            }
        })?;
        match shape_geometry {
            Geometry::TriangleMesh(mesh) => {
                for point in &mesh.positions {
                    extend_bounds(&mut min, &mut max, point.0);
                }
            }
            Geometry::BilinearPatchMesh(mesh) => {
                for point in &mesh.positions {
                    extend_bounds(&mut min, &mut max, point.0);
                }
            }
            Geometry::CurveMesh(mesh) => {
                for curve in &mesh.curves {
                    let radius = curve.widths.iter().copied().fold(0.0, f32::max) * 0.5;
                    for point in curve.control_points {
                        extend_bounds_with_radius(&mut min, &mut max, point.0, radius);
                    }
                }
            }
            Geometry::Quadric(quadric) => {
                let (radius, z_min, z_max) = match quadric {
                    Quadric::Sphere {
                        radius,
                        z_min,
                        z_max,
                        ..
                    }
                    | Quadric::Cylinder {
                        radius,
                        z_min,
                        z_max,
                        ..
                    } => (*radius, *z_min, *z_max),
                    Quadric::Disk { radius, height, .. } => (*radius, *height, *height),
                };
                extend_bounds(&mut min, &mut max, [-radius, -radius, z_min]);
                extend_bounds(&mut min, &mut max, [radius, radius, z_max]);
            }
            Geometry::DisplacedTriangleMesh(mesh) => {
                for bound in &mesh.displaced_bounds_object {
                    extend_bounds(&mut min, &mut max, bound.min.0);
                    extend_bounds(&mut min, &mut max, bound.max.0);
                }
            }
        }
    }
    if min.iter().zip(max.iter()).all(|(min, max)| min <= max) {
        Ok(Bounds3 {
            min: Point3(min),
            max: Point3(max),
        })
    } else {
        Err(GpuCompileError::InvalidParameter {
            parameter: "ObjectBegin",
            detail: "instance definition must contain at least one supported shape".to_owned(),
            source: empty_source(),
        })
    }
}

fn extend_bounds(min: &mut [f32; 3], max: &mut [f32; 3], point: [f32; 3]) {
    extend_bounds_with_radius(min, max, point, 0.0);
}

fn extend_bounds_with_radius(min: &mut [f32; 3], max: &mut [f32; 3], point: [f32; 3], radius: f32) {
    for axis in 0..3 {
        min[axis] = min[axis].min(point[axis] - radius);
        max[axis] = max[axis].max(point[axis] + radius);
    }
}

fn bilinear_mesh(
    params: &ParameterDictionary,
    source: &GpuSourceLocation,
) -> Result<BilinearPatchMesh, GpuCompileError> {
    let raw_positions = params.get_points("P");
    if raw_positions.is_empty() || raw_positions.len() % 3 != 0 {
        return Err(invalid_parameter(
            "P",
            "position count must be a non-zero multiple of 3",
            source,
        ));
    }
    let positions = raw_positions
        .chunks_exact(3)
        .map(|point| {
            Ok(Point3([
                to_gpu_float(point[0], source)?,
                to_gpu_float(point[1], source)?,
                to_gpu_float(point[2], source)?,
            ]))
        })
        .collect::<Result<Vec<_>, GpuCompileError>>()?;
    let raw_indices = params.get_ints("indices");
    if raw_indices.is_empty() || raw_indices.len() % 4 != 0 {
        return Err(invalid_parameter(
            "indices",
            "index count must be a non-zero multiple of 4",
            source,
        ));
    }
    let indices = raw_indices
        .chunks_exact(4)
        .map(|patch| {
            let mut result = [0; 4];
            for (dst, index) in result.iter_mut().zip(patch) {
                *dst = Index::try_from(*index)
                    .map_err(|_| invalid_parameter("indices", "negative index", source))?;
            }
            Ok(result)
        })
        .collect::<Result<Vec<[Index; 4]>, GpuCompileError>>()?;
    Ok(BilinearPatchMesh {
        positions,
        indices,
        normals: optional_vec3(params.get_points("N"), "N", source)?,
        uvs: optional_vec2(params.get_points("uv"), "uv", source)?,
        face_indices: None,
    })
}

fn quadric(
    params: &ParameterDictionary,
    name: &'static str,
    source: &GpuSourceLocation,
) -> Result<Quadric, GpuCompileError> {
    let radius = finite_parameter(params, "radius", 1.0, source)?;
    let z_min = finite_parameter(params, "zmin", -1.0, source)?;
    let z_max = finite_parameter(params, "zmax", 1.0, source)?;
    let phi_max_radians =
        finite_parameter(params, "phimax", 360.0, source)? * std::f32::consts::PI / 180.0;
    match name {
        "sphere" => Ok(Quadric::Sphere {
            radius,
            z_min,
            z_max,
            phi_max_radians,
        }),
        "cylinder" => Ok(Quadric::Cylinder {
            radius,
            z_min,
            z_max,
            phi_max_radians,
        }),
        "disk" => Ok(Quadric::Disk {
            height: finite_parameter(params, "height", 0.0, source)?,
            radius,
            inner_radius: finite_parameter(params, "innerradius", 0.0, source)?,
            phi_max_radians,
        }),
        _ => Err(invalid_parameter("shape", "unknown quadric", source)),
    }
}

fn finite_parameter(
    params: &ParameterDictionary,
    name: &'static str,
    default: Float,
    source: &GpuSourceLocation,
) -> Result<Float, GpuCompileError> {
    let value = params.get_one_float(name, default) as f32;
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| invalid_parameter(name, "value must be finite", source))
}

fn vector3_parameter(
    params: &ParameterDictionary,
    name: &'static str,
    default: [Float; 3],
    source: &GpuSourceLocation,
) -> Result<Vector3, GpuCompileError> {
    let values = params.get_points(name);
    let values = if values.is_empty() {
        default
    } else if values.len() == 3 {
        [values[0], values[1], values[2]]
    } else {
        return Err(invalid_parameter(
            name,
            "vector parameter requires exactly three values",
            source,
        ));
    };
    Ok(Vector3([
        to_gpu_float(values[0], source)?,
        to_gpu_float(values[1], source)?,
        to_gpu_float(values[2], source)?,
    ]))
}

fn compile_rgb_spectrum(
    params: &ParameterDictionary,
    name: &'static str,
    default: [Float; 3],
    source: &GpuSourceLocation,
    spectra: &mut Vec<SpectrumResource>,
) -> Result<SpectrumId, GpuCompileError> {
    if params.get_textures_ref(name).is_some()
        || params.get_spectrums_ref(name).is_some()
        || params.get_sampled_spectra_ref(name).is_some()
    {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "textured or sampled light spectrum",
            source: source.clone(),
        });
    }
    let values = params.get_points(name);
    let coefficients = match values.as_slice() {
        [] => default,
        [r, g, b] => [*r, *g, *b],
        _ => {
            return Err(invalid_parameter(
                name,
                "spectrum must contain exactly three RGB values",
                source,
            ))
        }
    };
    let id = SpectrumId(spectra.len() as Index);
    spectra.push(SpectrumResource::RgbUnbounded {
        coefficients: [
            to_gpu_float(coefficients[0], source)?,
            to_gpu_float(coefficients[1], source)?,
            to_gpu_float(coefficients[2], source)?,
        ],
    });
    Ok(id)
}

fn compile_material(
    builder: &SceneBuilder,
    shape: &ShapeSceneEntity,
    spectra: &mut Vec<SpectrumResource>,
    spectrum_textures: &mut Vec<SpectrumTexture>,
    float_texture_ids: &[Option<super::ir::FloatTextureId>],
    spectrum_texture_ids: &[Option<super::ir::SpectrumTextureId>],
    images: &mut Vec<ImageResource>,
    materials: &mut Vec<Material>,
) -> Result<MaterialId, GpuCompileError> {
    if shape.material_is_default
        && shape.material_index == usize::MAX
        && shape.material_name.is_none()
    {
        return Ok(MaterialId(0));
    }
    let material_index = if let Some(index) = shape.material_name.as_ref() {
        builder.named_materials.get(index).copied().ok_or_else(|| {
            invalid_parameter(
                "material",
                "unknown named material",
                &source_location(shape),
            )
        })?
    } else {
        shape.material_index
    };
    let material = builder.materials.get(material_index).ok_or_else(|| {
        invalid_parameter(
            "material",
            "material index is out of range",
            &source_location(shape),
        )
    })?;
    if material.base.name != "diffuse" {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "non-diffuse material",
            source: GpuSourceLocation {
                filename: material.base.loc.filename.clone(),
                line: material.base.loc.line,
                column: material.base.loc.column,
            },
        });
    }
    let displacement = material_texture_id(
        builder,
        material
            .base
            .params
            .get_textures_ref("displacement")
            .as_deref()
            .map(|names| &**names),
        float_texture_ids,
        "displacement",
        &source_location(shape),
    )?;
    let normal_map = compile_normal_map_image(builder, material, images, &source_location(shape))?;
    let reflectance = material.base.params.get_points("reflectance");
    if let Some(texture_names) = material.base.params.get_textures_ref("reflectance") {
        let texture_name = texture_names.first().ok_or_else(|| {
            invalid_parameter(
                "reflectance",
                "texture reference is empty",
                &source_location(shape),
            )
        })?;
        let texture_index = builder
            .named_spectrum_textures
            .get(texture_name)
            .copied()
            .ok_or_else(|| {
                invalid_parameter(
                    "reflectance",
                    "referenced spectrum texture is not declared",
                    &source_location(shape),
                )
            })?;
        let texture_id = spectrum_texture_ids
            .get(texture_index)
            .and_then(|id| *id)
            .ok_or_else(|| {
                invalid_parameter(
                    "reflectance",
                    "referenced spectrum texture is not supported",
                    &source_location(shape),
                )
            })?;
        let id = MaterialId(materials.len() as u32);
        materials.push(Material::Diffuse(DiffuseMaterial {
            reflectance: texture_id,
            displacement,
            normal_map,
        }));
        return Ok(id);
    }
    let coefficients = match reflectance.as_slice() {
        [] => [0.5, 0.5, 0.5],
        [r, g, b] => [*r, *g, *b],
        _ => {
            return Err(invalid_parameter(
                "reflectance",
                "diffuse reflectance must contain exactly three RGB values",
                &source_location(shape),
            ))
        }
    };
    let spectrum = SpectrumId(spectra.len() as u32);
    spectra.push(SpectrumResource::RgbAlbedo {
        coefficients: [
            to_gpu_float(coefficients[0], &source_location(shape))?,
            to_gpu_float(coefficients[1], &source_location(shape))?,
            to_gpu_float(coefficients[2], &source_location(shape))?,
        ],
    });
    let spectrum_texture = super::ir::SpectrumTextureId(spectrum_textures.len() as Index);
    spectrum_textures.push(SpectrumTexture::Constant { value: spectrum });
    let id = MaterialId(materials.len() as u32);
    materials.push(Material::Diffuse(DiffuseMaterial {
        reflectance: spectrum_texture,
        displacement,
        normal_map,
    }));
    Ok(id)
}

fn material_texture_id(
    builder: &SceneBuilder,
    texture_names: Option<&[String]>,
    texture_ids: &[Option<super::ir::FloatTextureId>],
    parameter: &'static str,
    source: &GpuSourceLocation,
) -> Result<Option<super::ir::FloatTextureId>, GpuCompileError> {
    let Some(texture_name) = texture_names.and_then(|names| names.first()) else {
        return Ok(None);
    };
    let texture_index = builder
        .named_float_textures
        .get(texture_name.as_str())
        .copied()
        .ok_or_else(|| invalid_parameter(parameter, "texture is not declared", source))?;
    texture_ids
        .get(texture_index)
        .copied()
        .flatten()
        .map(Some)
        .ok_or_else(|| invalid_parameter(parameter, "texture is not supported", source))
}

fn compile_normal_map_image(
    builder: &SceneBuilder,
    material: &crate::parser::scene_builder::MaterialSceneEntity,
    images: &mut Vec<ImageResource>,
    source: &GpuSourceLocation,
) -> Result<Option<super::ir::ImageId>, GpuCompileError> {
    let params = make_absolute_path(&material.base.params, &builder.seen_work_dirs);
    let filename = params.get_one_filename("normalmap", "");
    if filename.is_empty() {
        return Ok(None);
    }
    let encoding = ColorEncoding::parse("linear")
        .map_err(|error| invalid_parameter("normalmap", &error.msg, source))?;
    let raw = read_raw_image_with_encoding(&filename, encoding)
        .map_err(|error| invalid_parameter("normalmap", &error.msg, source))?;
    if raw.channels < 3 || raw.resolution.x == 0 || raw.resolution.y == 0 {
        return Err(invalid_parameter(
            "normalmap",
            "normal map must contain at least RGB channels",
            source,
        ));
    }
    let source_pixels = raw_linear_pixels(&raw, source)?;
    let mut pixels = Vec::with_capacity(source_pixels.len() / raw.channels * 3);
    for pixel in source_pixels.chunks_exact(raw.channels) {
        pixels.extend_from_slice(&pixel[..3]);
    }
    let width = u32::try_from(raw.resolution.x)
        .map_err(|_| invalid_parameter("normalmap", "image width is too large", source))?;
    let height = u32::try_from(raw.resolution.y)
        .map_err(|_| invalid_parameter("normalmap", "image height is too large", source))?;
    let image = super::ir::ImageId(images.len() as Index);
    let (mip_values, mip_levels) = build_mip_storage(&pixels, width, height, 3);
    images.push(ImageResource {
        resolution: [width, height],
        channels: ImageChannels::Rgb,
        storage: encode_mip_storage(&mip_values, &raw.data, ColorEncoding::Linear, 3),
        mip_levels,
        color_encoding: super::ir::ColorEncoding::Linear,
    });
    Ok(Some(image))
}

fn triangle_mesh(
    params: &ParameterDictionary,
    source: &GpuSourceLocation,
) -> Result<TriangleMesh, GpuCompileError> {
    let positions = params.get_points("P");
    if positions.is_empty() {
        return Err(GpuCompileError::MissingParameter {
            parameter: "P",
            source: source.clone(),
        });
    }
    if positions.len() % 3 != 0 {
        return Err(invalid_parameter(
            "P",
            "position count is not divisible by 3",
            source,
        ));
    }
    let positions = positions
        .chunks_exact(3)
        .map(|p| {
            Ok(Point3([
                to_gpu_float(p[0], source)?,
                to_gpu_float(p[1], source)?,
                to_gpu_float(p[2], source)?,
            ]))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let raw_indices = params.get_ints("indices");
    if raw_indices.is_empty() {
        return Err(GpuCompileError::MissingParameter {
            parameter: "indices",
            source: source.clone(),
        });
    }
    if raw_indices.len() % 3 != 0 {
        return Err(invalid_parameter(
            "indices",
            "index count is not divisible by 3",
            source,
        ));
    }
    let mut indices = Vec::with_capacity(raw_indices.len() / 3);
    for triangle in raw_indices.chunks_exact(3) {
        let mut converted = [0; 3];
        for (dst, index) in converted.iter_mut().zip(triangle) {
            *dst = Index::try_from(*index)
                .map_err(|_| invalid_parameter("indices", "negative index", source))?;
        }
        indices.push(converted);
    }

    let normals = optional_vec3(params.get_points("N"), "N", source)?;
    let tangents = optional_vec3(params.get_points("S"), "S", source)?
        .map(|values| values.into_iter().map(|normal| Vector3(normal.0)).collect());
    let uvs = optional_vec2(params.get_points("uv"), "uv", source)?;
    Ok(TriangleMesh {
        positions,
        indices,
        normals,
        tangents,
        uvs,
        face_indices: None,
    })
}

fn optional_vec2(
    values: Vec<Float>,
    parameter: &'static str,
    source: &GpuSourceLocation,
) -> Result<Option<Vec<Point2>>, GpuCompileError> {
    if values.is_empty() {
        return Ok(None);
    }
    if values.len() % 2 != 0 {
        return Err(invalid_parameter(
            parameter,
            "value count is not divisible by 2",
            source,
        ));
    }
    values
        .chunks_exact(2)
        .map(|value| {
            Ok(Point2([
                to_gpu_float(value[0], source)?,
                to_gpu_float(value[1], source)?,
            ]))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn optional_vec3(
    values: Vec<Float>,
    parameter: &'static str,
    source: &GpuSourceLocation,
) -> Result<Option<Vec<Normal3>>, GpuCompileError> {
    if values.is_empty() {
        return Ok(None);
    }
    if values.len() % 3 != 0 {
        return Err(invalid_parameter(
            parameter,
            "value count is not divisible by 3",
            source,
        ));
    }
    values
        .chunks_exact(3)
        .map(|value| {
            Ok(Normal3([
                to_gpu_float(value[0], source)?,
                to_gpu_float(value[1], source)?,
                to_gpu_float(value[2], source)?,
            ]))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn static_transform(
    transform: &CpuTransform,
    source: &GpuSourceLocation,
) -> Result<StaticTransform, GpuCompileError> {
    Ok(StaticTransform {
        render_from_object: matrix(transform.m, source)?,
        object_from_render: matrix(transform.minv, source)?,
        swaps_handedness: transform.swaps_handedness(),
    })
}

fn matrix(matrix: CpuMatrix4x4, source: &GpuSourceLocation) -> Result<Matrix4x4, GpuCompileError> {
    let mut result = [[0.0; 4]; 4];
    for (row, values) in result.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = to_gpu_float(matrix.m[row * 4 + column], source)?;
        }
    }
    Ok(Matrix4x4(result))
}

fn render_config(
    builder: &SceneBuilder,
    transforms: &mut Vec<Transform>,
) -> Result<RenderConfig, GpuCompileError> {
    let source = empty_source();
    if builder.camera_name != "perspective" {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "non-perspective camera",
            source: source.clone(),
        });
    }
    if builder.sampler_name != "independent" {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "non-independent sampler",
            source: source.clone(),
        });
    }
    if builder.film_name != "rgb" {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "non-rgb film",
            source: source.clone(),
        });
    }
    if builder.filter_name != "box" {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "non-box filter",
            source: source.clone(),
        });
    }
    if builder.integrator_name != "volpath" {
        return Err(GpuCompileError::UnsupportedSceneFeature {
            feature: "non-volpath integrator",
            source,
        });
    }
    let xresolution = builder.film_params.get_one_int("xresolution", 1280);
    let yresolution = builder.film_params.get_one_int("yresolution", 720);
    if xresolution <= 0 || yresolution <= 0 {
        return Err(invalid_parameter(
            "xresolution/yresolution",
            "resolution must be positive",
            &empty_source(),
        ));
    }

    let mut min = [0_i32, 0_i32];
    let mut max = [xresolution, yresolution];
    if let Some(pixelbounds) = builder.film_params.get_ints_ref("pixelbounds") {
        if pixelbounds.len() != 4 {
            return Err(invalid_parameter(
                "pixelbounds",
                "value count must be 4",
                &empty_source(),
            ));
        }
        min = [pixelbounds[0], pixelbounds[2]];
        max = [pixelbounds[1], pixelbounds[3]];
        min[0] = min[0].max(0);
        min[1] = min[1].max(0);
        max[0] = max[0].min(xresolution);
        max[1] = max[1].min(yresolution);
    }
    if let Some(cropwindow) = builder.film_params.get_floats_ref("cropwindow") {
        if cropwindow.len() != 4 {
            return Err(invalid_parameter(
                "cropwindow",
                "value count must be 4",
                &empty_source(),
            ));
        }
        let x0 = cropwindow[0].min(cropwindow[1]).clamp(0.0, 1.0);
        let y0 = cropwindow[2].min(cropwindow[3]).clamp(0.0, 1.0);
        let x1 = cropwindow[0].max(cropwindow[1]).clamp(0.0, 1.0);
        let y1 = cropwindow[2].max(cropwindow[3]).clamp(0.0, 1.0);
        min = [
            (xresolution as Float * x0).ceil() as i32,
            (yresolution as Float * y0).ceil() as i32,
        ];
        max = [
            (xresolution as Float * x1).ceil() as i32,
            (yresolution as Float * y1).ceil() as i32,
        ];
    }
    if max[0] <= min[0] || max[1] <= min[1] {
        return Err(invalid_parameter(
            "pixelbounds/cropwindow",
            "pixel bounds must have positive area",
            &empty_source(),
        ));
    }
    let sample_count = builder.sampler_params.get_one_int("pixelsamples", 4);
    if sample_count <= 0 {
        return Err(invalid_parameter(
            "pixelsamples",
            "sample count must be positive",
            &empty_source(),
        ));
    }
    let pixel_bounds = Bounds2i {
        min: [
            u32::try_from(min[0]).map_err(|_| {
                invalid_parameter(
                    "pixelbounds",
                    "minimum must be non-negative",
                    &empty_source(),
                )
            })?,
            u32::try_from(min[1]).map_err(|_| {
                invalid_parameter(
                    "pixelbounds",
                    "minimum must be non-negative",
                    &empty_source(),
                )
            })?,
        ],
        max: [
            u32::try_from(max[0]).map_err(|_| {
                invalid_parameter(
                    "pixelbounds",
                    "maximum must be non-negative",
                    &empty_source(),
                )
            })?,
            u32::try_from(max[1]).map_err(|_| {
                invalid_parameter(
                    "pixelbounds",
                    "maximum must be non-negative",
                    &empty_source(),
                )
            })?,
        ],
    };
    let samples_per_pixel = u32::try_from(sample_count).map_err(|_| {
        invalid_parameter(
            "pixelsamples",
            "sample count does not fit u32",
            &empty_source(),
        )
    })?;

    let frame = builder.camera_params.get_one_float(
        "frameaspectratio",
        xresolution as Float / yresolution as Float,
    );
    let screen = if let Some(values) = builder.camera_params.get_floats_ref("screenwindow") {
        if values.len() != 4 {
            return Err(invalid_parameter(
                "screenwindow",
                "value count must be 4",
                &empty_source(),
            ));
        }
        [values[0], values[1], values[2], values[3]]
    } else if frame > 1.0 {
        [-frame, frame, -1.0, 1.0]
    } else {
        [-1.0, 1.0, -1.0 / frame, 1.0 / frame]
    };
    let mut fov = builder.camera_params.get_one_float("fov", 90.0);
    let halffov = builder.camera_params.get_one_float("halffov", -1.0);
    if halffov > 0.0 {
        fov = 2.0 * halffov;
    }
    let camera_to_screen = CpuTransform::perspective(fov, 1e-2, 1000.0);
    let screen_to_raster = CpuTransform::scale(xresolution as Float, yresolution as Float, 1.0)
        * CpuTransform::scale(
            1.0 / (screen[1] - screen[0]),
            1.0 / (screen[2] - screen[3]),
            1.0,
        )
        * CpuTransform::translate(-screen[0], -screen[3], 0.0);
    let raster_to_camera = camera_to_screen.inverse() * screen_to_raster.inverse();
    let camera_transform_id = TransformId(transforms.len() as Index);
    transforms.push(Transform::Static(static_transform(
        &builder.camera_to_world[0],
        &empty_source(),
    )?));

    let lens_radius = finite_parameter(&builder.camera_params, "lensradius", 0.0, &empty_source())?;
    let focal_distance = finite_parameter(
        &builder.camera_params,
        "focaldistance",
        1e6,
        &empty_source(),
    )?;
    let shutter_open =
        finite_parameter(&builder.camera_params, "shutteropen", 0.0, &empty_source())?;
    let shutter_close =
        finite_parameter(&builder.camera_params, "shutterclose", 1.0, &empty_source())?;
    if focal_distance <= 0.0 || shutter_open > shutter_close {
        return Err(invalid_parameter(
            "camera",
            "focal distance must be positive and shutter interval ordered",
            &empty_source(),
        ));
    }
    let diagonal_mm = finite_parameter(&builder.film_params, "diagonal", 35.0, &empty_source())?;
    let iso = finite_parameter(&builder.film_params, "iso", 100.0, &empty_source())?;
    let max_component_value = finite_parameter(
        &builder.film_params,
        "maxcomponentvalue",
        1e6,
        &empty_source(),
    )?;
    let xradius = finite_parameter(&builder.filter_params, "xradius", 0.5, &empty_source())?;
    let yradius = finite_parameter(&builder.filter_params, "yradius", 0.5, &empty_source())?;
    if diagonal_mm <= 0.0
        || iso <= 0.0
        || max_component_value <= 0.0
        || xradius <= 0.0
        || yradius <= 0.0
    {
        return Err(invalid_parameter(
            "film/filter",
            "film and filter parameters must be positive",
            &empty_source(),
        ));
    }
    let max_depth = builder.integrator_params.get_one_int("maxdepth", 5);
    if max_depth <= 0 {
        return Err(invalid_parameter(
            "maxdepth",
            "maximum depth must be positive",
            &empty_source(),
        ));
    }
    Ok(RenderConfig {
        camera: PerspectiveCamera {
            render_from_camera: camera_transform_id,
            camera_from_raster: matrix(raster_to_camera.m, &empty_source())?,
            lens_radius,
            focal_distance,
            shutter_open,
            shutter_close,
        },
        sampler: IndependentSampler {
            samples_per_pixel,
            seed: builder.sampler_params.get_one_int("seed", 0) as u64,
        },
        film: RgbFilm {
            full_resolution: [xresolution as u32, yresolution as u32],
            pixel_bounds,
            diagonal_mm,
            output_rgb_from_xyz: Matrix3x3([
                [3.240479, -1.537150, -0.498535],
                [-0.969256, 1.875991, 0.041556],
                [0.055648, -0.204043, 1.057311],
            ]),
            iso,
            max_component_value,
        },
        filter: BoxFilter {
            radius: Vector2([xradius, yradius]),
        },
        integrator: WavefrontVolPath {
            max_depth: u32::try_from(max_depth).map_err(|_| {
                invalid_parameter(
                    "maxdepth",
                    "maximum depth does not fit u32",
                    &empty_source(),
                )
            })?,
            regularize: builder.integrator_params.get_one_bool("regularize", false),
        },
        light_sampler: LightSampler::Uniform,
    })
}

fn empty_source() -> GpuSourceLocation {
    GpuSourceLocation {
        filename: String::new(),
        line: 0,
        column: 0,
    }
}

#[derive(Default)]
struct GpuSourceGroups {
    shapes: Vec<SourceId>,
    float_textures: Vec<SourceId>,
    spectrum_textures: Vec<SourceId>,
    materials: Vec<SourceId>,
    lights: Vec<SourceId>,
    instance_definitions: Vec<SourceId>,
    instances: Vec<SourceId>,
}

fn source_map(builder: &SceneBuilder, ir: &SceneIr) -> GpuSourceMap {
    let mut locations = Vec::new();
    let mut groups = GpuSourceGroups::default();
    for shape in &builder.shapes {
        add_source(&mut locations, &mut groups.shapes, source_location(shape));
    }
    for shape in &builder.animated_shapes {
        add_source(&mut locations, &mut groups.shapes, source_location(shape));
    }
    let mut definitions: Vec<_> = builder.instance_definitions.iter().collect();
    definitions.sort_by(|(left, _), (right, _)| left.cmp(right));
    for (_, definition) in &definitions {
        for shape in &definition.shapes {
            add_source(&mut locations, &mut groups.shapes, source_location(shape));
        }
        for shape in &definition.animated_shapes {
            add_source(&mut locations, &mut groups.shapes, source_location(shape));
        }
        add_source(
            &mut locations,
            &mut groups.instance_definitions,
            GpuSourceLocation {
                filename: definition.loc.filename.clone(),
                line: definition.loc.line,
                column: definition.loc.column,
            },
        );
    }
    for (name, _) in definitions {
        for instance in builder
            .instance_uses
            .iter()
            .filter(|instance| instance.name == *name)
        {
            add_source(
                &mut locations,
                &mut groups.instances,
                GpuSourceLocation {
                    filename: instance.loc.filename.clone(),
                    line: instance.loc.line,
                    column: instance.loc.column,
                },
            );
        }
    }
    for texture in &builder.float_textures {
        add_source(
            &mut locations,
            &mut groups.float_textures,
            texture_source_location(texture),
        );
    }
    for texture in &builder.spectrum_textures {
        add_source(
            &mut locations,
            &mut groups.spectrum_textures,
            texture_source_location(texture),
        );
    }
    for material in &builder.materials {
        groups.materials.push(SourceId(locations.len() as Index));
        locations.push(GpuSourceLocation {
            filename: material.base.loc.filename.clone(),
            line: material.base.loc.line,
            column: material.base.loc.column,
        });
    }
    for light in &builder.lights {
        add_source(
            &mut locations,
            &mut groups.lights,
            light_source_location(light),
        );
    }
    let mut resources = Vec::new();
    for (index, primitive) in ir.view().primitives.iter().enumerate() {
        let Some(&source) = groups.shapes.get(index) else {
            continue;
        };
        add_resource(
            &mut resources,
            GpuResourceKind::Primitive,
            index as Index,
            source,
        );
        add_resource(
            &mut resources,
            GpuResourceKind::Geometry,
            primitive.geometry.0,
            source,
        );
        if let Some(super::ir::Geometry::DisplacedTriangleMesh(mesh)) =
            ir.view().geometry.get(primitive.geometry.0 as usize)
        {
            add_resource(
                &mut resources,
                GpuResourceKind::Geometry,
                mesh.base_mesh.0,
                source,
            );
        }
        add_resource(
            &mut resources,
            GpuResourceKind::Transform,
            primitive.transform.0,
            source,
        );
        if let Some(material) = primitive.material {
            add_resource(
                &mut resources,
                GpuResourceKind::Material,
                material.0,
                source,
            );
            if let Some(super::ir::Material::Diffuse(material)) =
                ir.view().materials.get(material.0 as usize)
            {
                add_spectrum_texture_resources(
                    &mut resources,
                    ir.view(),
                    material.reflectance,
                    source,
                );
                if let Some(texture) = material.displacement {
                    add_float_texture_resources(&mut resources, ir.view(), texture, source);
                }
                if let Some(image) = material.normal_map {
                    add_resource(&mut resources, GpuResourceKind::Image, image.0, source);
                }
            }
        }
        if let Some(texture) = primitive.alpha {
            add_float_texture_resources(&mut resources, ir.view(), texture, source);
        }
        match &primitive.area_light {
            super::ir::AreaLightBinding::None => {}
            super::ir::AreaLightBinding::Uniform(light) => {
                add_resource(&mut resources, GpuResourceKind::Light, light.0, source);
                if let Some(super::ir::Light::DiffuseArea(light)) =
                    ir.view().lights.get(light.0 as usize)
                {
                    add_spectrum_texture_resources(
                        &mut resources,
                        ir.view(),
                        light.emission,
                        source,
                    );
                }
            }
            super::ir::AreaLightBinding::PerElement(lights) => {
                for light in lights {
                    add_resource(&mut resources, GpuResourceKind::Light, light.0, source);
                }
            }
        }
    }
    for (index, texture) in ir.view().float_textures.iter().enumerate() {
        let Some(&source) = groups.float_textures.get(index) else {
            continue;
        };
        add_resource(
            &mut resources,
            GpuResourceKind::FloatTexture,
            index as Index,
            source,
        );
        if let super::ir::FloatTexture::Image { image, mapping, .. } = texture {
            add_resource(&mut resources, GpuResourceKind::Image, image.0, source);
            add_resource(
                &mut resources,
                GpuResourceKind::TextureMapping,
                mapping.0,
                source,
            );
        }
    }
    for (index, texture) in ir.view().spectrum_textures.iter().enumerate().skip(1) {
        let Some(&source) = groups.spectrum_textures.get(index - 1) else {
            continue;
        };
        add_resource(
            &mut resources,
            GpuResourceKind::SpectrumTexture,
            index as Index,
            source,
        );
        if let super::ir::SpectrumTexture::Image { image, mapping, .. } = texture {
            add_resource(&mut resources, GpuResourceKind::Image, image.0, source);
            add_resource(
                &mut resources,
                GpuResourceKind::TextureMapping,
                mapping.0,
                source,
            );
        }
    }
    for (index, light) in ir.view().lights.iter().enumerate() {
        if let Some(&source) = groups.lights.get(index) {
            add_resource(
                &mut resources,
                GpuResourceKind::Light,
                index as Index,
                source,
            );
            if let super::ir::Light::Point(point) = light {
                add_resource(
                    &mut resources,
                    GpuResourceKind::Transform,
                    point.render_from_light.0,
                    source,
                );
            }
        }
    }
    for (index, source) in groups.instance_definitions.iter().copied().enumerate() {
        add_resource(
            &mut resources,
            GpuResourceKind::InstanceDefinition,
            index as Index,
            source,
        );
    }
    for (index, instance) in ir.view().instances.iter().enumerate() {
        let Some(&source) = groups.instances.get(index) else {
            continue;
        };
        add_resource(
            &mut resources,
            GpuResourceKind::Instance,
            index as Index,
            source,
        );
        add_resource(
            &mut resources,
            GpuResourceKind::Transform,
            instance.transform.0,
            source,
        );
    }
    resources.sort_by_key(|entry| (entry.kind, entry.index));
    GpuSourceMap {
        locations: locations.into_boxed_slice(),
        resources: resources.into_boxed_slice(),
    }
}

fn add_source(
    locations: &mut Vec<GpuSourceLocation>,
    group: &mut Vec<SourceId>,
    location: GpuSourceLocation,
) {
    let id = SourceId(locations.len() as Index);
    locations.push(location);
    group.push(id);
}

fn add_resource(
    resources: &mut Vec<GpuSourceEntry>,
    kind: GpuResourceKind,
    index: Index,
    source: SourceId,
) {
    if resources
        .iter()
        .any(|entry| entry.kind == kind && entry.index == index)
    {
        return;
    }
    resources.push(GpuSourceEntry {
        kind,
        index,
        source,
    });
}

fn add_float_texture_resources(
    resources: &mut Vec<GpuSourceEntry>,
    view: super::ir::SceneView<'_>,
    texture: super::ir::FloatTextureId,
    source: SourceId,
) {
    add_resource(resources, GpuResourceKind::FloatTexture, texture.0, source);
    if let Some(super::ir::FloatTexture::Image { image, mapping, .. }) =
        view.float_textures.get(texture.0 as usize)
    {
        add_resource(resources, GpuResourceKind::Image, image.0, source);
        add_resource(
            resources,
            GpuResourceKind::TextureMapping,
            mapping.0,
            source,
        );
    }
}

fn add_spectrum_texture_resources(
    resources: &mut Vec<GpuSourceEntry>,
    view: super::ir::SceneView<'_>,
    texture: super::ir::SpectrumTextureId,
    source: SourceId,
) {
    add_resource(
        resources,
        GpuResourceKind::SpectrumTexture,
        texture.0,
        source,
    );
    match view.spectrum_textures.get(texture.0 as usize) {
        Some(super::ir::SpectrumTexture::Constant { value }) => {
            add_resource(resources, GpuResourceKind::Spectrum, value.0, source);
        }
        Some(super::ir::SpectrumTexture::Image { image, mapping, .. }) => {
            add_resource(resources, GpuResourceKind::Image, image.0, source);
            add_resource(
                resources,
                GpuResourceKind::TextureMapping,
                mapping.0,
                source,
            );
        }
        None => {}
    }
}

fn attach_requirement_sources(
    requirements: &mut super::ir::Requirements,
    view: super::ir::SceneView<'_>,
    source_map: &GpuSourceMap,
) {
    for required in requirements.features.iter_mut() {
        let mut sources = source_map
            .resources
            .iter()
            .filter_map(|entry| {
                requirement_resource_matches(required.feature, entry, view).then_some(entry.source)
            })
            .collect::<Vec<_>>();
        sources.sort_unstable();
        sources.dedup();
        required.sources = sources.into_boxed_slice();
    }
}

fn requirement_resource_matches(
    feature: Feature,
    entry: &GpuSourceEntry,
    view: super::ir::SceneView<'_>,
) -> bool {
    match feature {
        Feature::TriangleMesh => {
            entry.kind == GpuResourceKind::Geometry
                && matches!(
                    view.geometry.get(entry.index as usize),
                    Some(super::ir::Geometry::TriangleMesh(_))
                )
        }
        Feature::BilinearPatch => {
            entry.kind == GpuResourceKind::Geometry
                && matches!(
                    view.geometry.get(entry.index as usize),
                    Some(super::ir::Geometry::BilinearPatchMesh(_))
                )
        }
        Feature::Curve => {
            entry.kind == GpuResourceKind::Geometry
                && matches!(
                    view.geometry.get(entry.index as usize),
                    Some(super::ir::Geometry::CurveMesh(_))
                )
        }
        Feature::Quadric => {
            entry.kind == GpuResourceKind::Geometry
                && matches!(
                    view.geometry.get(entry.index as usize),
                    Some(super::ir::Geometry::Quadric(_))
                )
        }
        Feature::DisplacedTriangle => {
            entry.kind == GpuResourceKind::Geometry
                && matches!(
                    view.geometry.get(entry.index as usize),
                    Some(super::ir::Geometry::DisplacedTriangleMesh(_))
                )
        }
        Feature::FloatConstantTexture => {
            entry.kind == GpuResourceKind::FloatTexture
                && matches!(
                    view.float_textures.get(entry.index as usize),
                    Some(super::ir::FloatTexture::Constant { .. })
                )
        }
        Feature::FloatImageTexture => {
            entry.kind == GpuResourceKind::FloatTexture
                && matches!(
                    view.float_textures.get(entry.index as usize),
                    Some(super::ir::FloatTexture::Image { .. })
                )
        }
        Feature::SpectrumConstantTexture => {
            entry.kind == GpuResourceKind::SpectrumTexture
                && matches!(
                    view.spectrum_textures.get(entry.index as usize),
                    Some(super::ir::SpectrumTexture::Constant { .. })
                )
        }
        Feature::SpectrumImageTexture => {
            entry.kind == GpuResourceKind::SpectrumTexture
                && matches!(
                    view.spectrum_textures.get(entry.index as usize),
                    Some(super::ir::SpectrumTexture::Image { .. })
                )
        }
        Feature::DiffuseMaterial => {
            entry.kind == GpuResourceKind::Material
                && matches!(
                    view.materials.get(entry.index as usize),
                    Some(super::ir::Material::Diffuse(_))
                )
        }
        Feature::PointLight => {
            entry.kind == GpuResourceKind::Light
                && matches!(
                    view.lights.get(entry.index as usize),
                    Some(super::ir::Light::Point(_))
                )
        }
        Feature::DiffuseAreaLight => {
            entry.kind == GpuResourceKind::Light
                && matches!(
                    view.lights.get(entry.index as usize),
                    Some(super::ir::Light::DiffuseArea(_))
                )
        }
        Feature::UniformInfiniteLight => {
            entry.kind == GpuResourceKind::Light
                && matches!(
                    view.lights.get(entry.index as usize),
                    Some(super::ir::Light::UniformInfinite(_))
                )
        }
        Feature::StaticTransform => {
            entry.kind == GpuResourceKind::Transform
                && matches!(
                    view.transforms.get(entry.index as usize),
                    Some(super::ir::Transform::Static(_))
                )
        }
        Feature::AnimatedTransform => {
            entry.kind == GpuResourceKind::Transform
                && matches!(
                    view.transforms.get(entry.index as usize),
                    Some(super::ir::Transform::Animated(_))
                )
        }
        Feature::PerspectiveCamera
        | Feature::IndependentSampler
        | Feature::RgbFilm
        | Feature::BoxFilter
        | Feature::WavefrontVolPath
        | Feature::UniformLightSampler => false,
    }
}

fn to_gpu_float(value: Float, source: &GpuSourceLocation) -> Result<Float, GpuCompileError> {
    let value = value as f32;
    value.is_finite().then_some(value).ok_or_else(|| {
        invalid_parameter(
            "numeric value",
            "value cannot be represented as finite f32",
            source,
        )
    })
}

fn source_location(shape: &ShapeSceneEntity) -> GpuSourceLocation {
    GpuSourceLocation {
        filename: shape.base.loc.filename.clone(),
        line: shape.base.loc.line,
        column: shape.base.loc.column,
    }
}

fn light_source_location(light: &LightSceneEntity) -> GpuSourceLocation {
    GpuSourceLocation {
        filename: light.base.base.loc.filename.clone(),
        line: light.base.base.loc.line,
        column: light.base.base.loc.column,
    }
}

fn texture_source_location(
    texture: &crate::parser::scene_builder::TextureSceneEntity,
) -> GpuSourceLocation {
    GpuSourceLocation {
        filename: texture.base.loc.filename.clone(),
        line: texture.base.loc.line,
        column: texture.base.loc.column,
    }
}

fn unsupported_feature(shape: &ShapeSceneEntity, feature: &'static str) -> GpuCompileError {
    GpuCompileError::UnsupportedSceneFeature {
        feature,
        source: source_location(shape),
    }
}

fn invalid_parameter(
    parameter: &'static str,
    detail: &str,
    source: &GpuSourceLocation,
) -> GpuCompileError {
    GpuCompileError::InvalidParameter {
        parameter,
        detail: detail.to_owned(),
        source: source.clone(),
    }
}
