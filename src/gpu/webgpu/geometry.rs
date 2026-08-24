use super::super::ir::{
    GeometryId, GpuAreaLightBinding, GpuColorEncoding, GpuFloatImageChannel, GpuFloatTexture,
    GpuGeometry, GpuImageChannels, GpuImageFilter, GpuImageResource, GpuImageWrapMode, GpuLight,
    GpuMaterial, GpuSceneView, GpuSpectrumResource, GpuSpectrumTexture, GpuSpectrumType,
    GpuTexelStorage, GpuTextureMapping, GpuTransform, PrimitiveId,
};
use super::device::DeviceContext;
use super::error::{BackendError, PlanError};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BlasBuildEntry, BlasGeometries, BlasGeometrySizeDescriptors,
    BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor, BufferUsages, CreateBlasDescriptor,
    CreateTlasDescriptor, IndexFormat, Tlas, TlasInstance, VertexFormat,
};

const MAX_TLAS_CUSTOM_DATA: u32 = 0x00ff_ffff;

pub fn tlas_transform(matrix: [[f32; 4]; 4]) -> [f32; 12] {
    [
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[0][3],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[1][3],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
        matrix[2][3],
    ]
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexPlan {
    pub position: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImageMipPlan {
    pub resolution: [u32; 2],
    pub texel_offset: u32,
    pub texel_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImagePlan {
    pub resolution: [u32; 2],
    pub channels: GpuImageChannels,
    pub mip_levels: Vec<ImageMipPlan>,
    pub texels: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumTexturePlan {
    pub image: u32,
    pub mapping: GpuTextureMapping,
    pub scale: f32,
    pub invert: bool,
    pub swrap: GpuImageWrapMode,
    pub twrap: GpuImageWrapMode,
    pub filter: GpuImageFilter,
    pub spectrum_type: GpuSpectrumType,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FloatTexturePlan {
    Constant(f32),
    Image {
        image: u32,
        mapping: GpuTextureMapping,
        scale: f32,
        invert: bool,
        swrap: GpuImageWrapMode,
        twrap: GpuImageWrapMode,
        filter: GpuImageFilter,
        channel: GpuFloatImageChannel,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum MaterialReflectancePlan {
    Constant([f32; 4]),
    SpectrumTexture(u32),
}

impl MaterialPlan {
    fn requires_uv(&self) -> bool {
        matches!(
            self.reflectance,
            MaterialReflectancePlan::SpectrumTexture(_)
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlasPlan {
    pub geometry: GeometryId,
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TlasInstancePlan {
    pub blas: u32,
    pub transform: [f32; 12],
    pub custom_data: u32,
    pub mask: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrimitivePlan {
    pub first_vertex: u32,
    pub first_index: u32,
    pub triangle_count: u32,
    pub material: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialPlan {
    pub reflectance: MaterialReflectancePlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransformPlan {
    pub render_from_object: [[f32; 4]; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct LightPlan {
    pub position: [f32; 4],
    pub intensity: [f32; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenePlan {
    pub vertices: Vec<VertexPlan>,
    pub indices: Vec<u32>,
    pub blases: Vec<BlasPlan>,
    pub tlas_instances: Vec<TlasInstancePlan>,
    pub primitives: Vec<PrimitivePlan>,
    pub materials: Vec<MaterialPlan>,
    pub transforms: Vec<TransformPlan>,
    pub lights: Vec<LightPlan>,
    pub images: Vec<ImagePlan>,
    pub float_textures: Vec<FloatTexturePlan>,
    pub spectrum_textures: Vec<SpectrumTexturePlan>,
}

impl ScenePlan {
    pub fn validate_custom_data(custom_data: u32) -> Result<(), PlanError> {
        if custom_data > MAX_TLAS_CUSTOM_DATA {
            return Err(PlanError::LimitExceeded {
                resource: "tlas_instance_custom_data",
                value: custom_data,
                maximum: MAX_TLAS_CUSTOM_DATA,
            });
        }
        Ok(())
    }

    pub fn from_scene(scene: GpuSceneView<'_>) -> Result<Self, BackendError> {
        if !scene.instance_definitions.is_empty()
            || !scene.instances.is_empty()
            || !scene.world_instances.is_empty()
        {
            return Err(BackendError::Plan(PlanError::UnsupportedInstances));
        }
        if scene.world_primitives.is_empty() {
            return Err(BackendError::Plan(PlanError::EmptyScene));
        }

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut blases = Vec::new();
        let mut geometry_to_blas = vec![None; scene.geometry.len()];
        let mut tlas_instances = Vec::with_capacity(scene.world_primitives.len());
        let mut primitives = Vec::with_capacity(scene.world_primitives.len());
        let mut materials = Vec::with_capacity(scene.world_primitives.len());
        let mut transforms = Vec::with_capacity(scene.world_primitives.len());
        let mut images = Vec::new();
        let mut float_textures = Vec::with_capacity(scene.float_textures.len());
        let mut spectrum_textures = Vec::new();
        let mut image_to_plan = vec![None; scene.images.len()];

        for texture in scene.float_textures {
            float_textures.push(lower_float_texture(
                scene,
                *texture,
                &mut images,
                &mut image_to_plan,
            )?);
        }

        for primitive_id in scene.world_primitives {
            let primitive =
                scene
                    .primitives
                    .get(primitive_id.0 as usize)
                    .ok_or(BackendError::Plan(PlanError::InvalidReference {
                        resource: "primitive",
                        index: primitive_id.0,
                    }))?;
            let geometry =
                scene
                    .geometry
                    .get(primitive.geometry.0 as usize)
                    .ok_or(BackendError::Plan(PlanError::InvalidReference {
                        resource: "geometry",
                        index: primitive.geometry.0,
                    }))?;
            let triangle_mesh = match geometry {
                GpuGeometry::TriangleMesh(mesh) => mesh,
                _ => {
                    return Err(BackendError::Plan(PlanError::UnsupportedGeometry {
                        geometry: primitive.geometry,
                    }))
                }
            };
            if triangle_mesh.positions.is_empty() || triangle_mesh.indices.is_empty() {
                return Err(BackendError::Plan(PlanError::EmptyGeometry {
                    geometry: primitive.geometry,
                }));
            }
            if primitive.alpha.is_some() || primitive.shadow_alpha.is_some() {
                return Err(BackendError::Plan(PlanError::UnsupportedAlpha {
                    primitive: primitive_id.0,
                }));
            }
            if !matches!(primitive.area_light, GpuAreaLightBinding::None) {
                return Err(BackendError::Plan(PlanError::UnsupportedAreaLight {
                    primitive: primitive_id.0,
                }));
            }
            if primitive.reverse_orientation {
                return Err(BackendError::Plan(
                    PlanError::UnsupportedReverseOrientation {
                        primitive: primitive_id.0,
                    },
                ));
            }
            let transform =
                scene
                    .transforms
                    .get(primitive.transform.0 as usize)
                    .ok_or(BackendError::Plan(PlanError::InvalidReference {
                        resource: "transform",
                        index: primitive.transform.0,
                    }))?;
            let transform = match transform {
                GpuTransform::Static(transform) => transform.render_from_object.0,
                GpuTransform::Animated(_) => {
                    return Err(BackendError::Plan(PlanError::UnsupportedTransform {
                        transform: primitive.transform,
                    }))
                }
            };
            let material = lower_material(
                scene,
                primitive.material,
                *primitive_id,
                &mut images,
                &mut spectrum_textures,
                &mut image_to_plan,
            )?;

            if material.requires_uv()
                && (triangle_mesh.uvs.is_none() || triangle_mesh.face_indices.is_some())
            {
                return Err(BackendError::Plan(PlanError::UnsupportedTexture {
                    texture: primitive.material.map_or(0, |material| material.0),
                }));
            }

            let blas = match geometry_to_blas[primitive.geometry.0 as usize] {
                Some(index) => index,
                None => {
                    let first_vertex = checked_len(vertices.len(), "vertex buffer")?;
                    let first_index = checked_len(indices.len(), "index buffer")?;
                    if triangle_mesh.indices.iter().flatten().any(|index| {
                        usize::try_from(*index)
                            .map(|index| index >= triangle_mesh.positions.len())
                            .unwrap_or(true)
                    }) {
                        return Err(BackendError::Plan(PlanError::InvalidReference {
                            resource: "triangle mesh vertex index",
                            index: primitive.geometry.0,
                        }));
                    }
                    let uvs = triangle_mesh.uvs.as_deref();
                    if let Some(uvs) = uvs {
                        if uvs.len() != triangle_mesh.positions.len() {
                            return Err(BackendError::Plan(PlanError::InvalidReference {
                                resource: "triangle mesh UV",
                                index: uvs.len() as u32,
                            }));
                        }
                    }
                    vertices.extend(triangle_mesh.positions.iter().enumerate().map(
                        |(index, position)| {
                            VertexPlan {
                                position: position.0,
                                uv: uvs
                                    .and_then(|uvs| uvs.get(index).map(|uv| uv.0))
                                    .unwrap_or([0.0, 0.0]),
                            }
                        },
                    ));
                    indices.extend(
                        triangle_mesh
                            .indices
                            .iter()
                            .flat_map(|triangle| triangle.iter().copied()),
                    );
                    let vertex_count = checked_len(triangle_mesh.positions.len(), "vertex count")?;
                    let index_count =
                        checked_len(
                            triangle_mesh.indices.len().checked_mul(3).ok_or(
                                BackendError::Plan(PlanError::LimitExceeded {
                                    resource: "index count",
                                    value: u32::MAX,
                                    maximum: u32::MAX,
                                }),
                            )?,
                            "index count",
                        )?;
                    let index = checked_len(blases.len(), "BLAS count")? as usize;
                    blases.push(BlasPlan {
                        geometry: primitive.geometry,
                        first_vertex,
                        vertex_count,
                        first_index,
                        index_count,
                    });
                    geometry_to_blas[primitive.geometry.0 as usize] = Some(index);
                    index
                }
            };

            let custom_data = checked_len(tlas_instances.len(), "TLAS custom data")?;
            Self::validate_custom_data(custom_data).map_err(BackendError::Plan)?;
            primitives.push(PrimitivePlan {
                first_vertex: blases[blas].first_vertex,
                first_index: blases[blas].first_index,
                triangle_count: blases[blas].index_count / 3,
                material: custom_data,
            });
            materials.push(material);
            transforms.push(TransformPlan {
                render_from_object: [transform[0], transform[1], transform[2], transform[3]],
            });
            tlas_instances.push(TlasInstancePlan {
                blas: blas as u32,
                transform: tlas_transform(transform),
                custom_data,
                mask: u8::MAX,
            });
        }

        let lights = lower_lights(scene)?;
        Ok(Self {
            vertices,
            indices,
            blases,
            tlas_instances,
            primitives,
            materials,
            transforms,
            lights,
            images,
            float_textures,
            spectrum_textures,
        })
    }
}

fn lower_material(
    scene: GpuSceneView<'_>,
    material_id: Option<super::super::ir::MaterialId>,
    primitive: PrimitiveId,
    images: &mut Vec<ImagePlan>,
    spectrum_textures: &mut Vec<SpectrumTexturePlan>,
    image_to_plan: &mut [Option<u32>],
) -> Result<MaterialPlan, BackendError> {
    let material_id = material_id.ok_or(BackendError::Plan(PlanError::UnsupportedMaterial {
        primitive: primitive.0,
    }))?;
    let material = scene
        .materials
        .get(material_id.0 as usize)
        .ok_or(BackendError::Plan(PlanError::InvalidReference {
            resource: "material",
            index: material_id.0,
        }))?;
    let GpuMaterial::Diffuse(diffuse) = material;
    if diffuse.displacement.is_some() || diffuse.normal_map.is_some() {
        return Err(BackendError::Plan(PlanError::UnsupportedMaterial {
            primitive: primitive.0,
        }));
    }
    let texture = scene
        .spectrum_textures
        .get(diffuse.reflectance.0 as usize)
        .ok_or(BackendError::Plan(PlanError::InvalidReference {
            resource: "spectrum texture",
            index: diffuse.reflectance.0,
        }))?;
    let reflectance = match texture {
        GpuSpectrumTexture::Constant { value } => {
            MaterialReflectancePlan::Constant(match spectrum_rgb(scene, *value, primitive.0)? {
                [red, green, blue] => [red, green, blue, 1.0],
            })
        }
        GpuSpectrumTexture::Image {
            image,
            mapping,
            scale,
            invert,
            swrap,
            twrap,
            filter,
            spectrum_type,
        } => {
            let texture_index = lower_spectrum_texture(
                scene,
                *image,
                *mapping,
                *scale,
                *invert,
                *swrap,
                *twrap,
                *filter,
                *spectrum_type,
                images,
                spectrum_textures,
                image_to_plan,
                primitive,
            )?;
            MaterialReflectancePlan::SpectrumTexture(texture_index)
        }
    };
    Ok(MaterialPlan { reflectance })
}

#[allow(clippy::too_many_arguments)]
fn lower_spectrum_texture(
    scene: GpuSceneView<'_>,
    image: super::super::ir::ImageId,
    mapping: super::super::ir::TextureMappingId,
    scale: f32,
    invert: bool,
    swrap: GpuImageWrapMode,
    twrap: GpuImageWrapMode,
    filter: GpuImageFilter,
    spectrum_type: GpuSpectrumType,
    images: &mut Vec<ImagePlan>,
    spectrum_textures: &mut Vec<SpectrumTexturePlan>,
    image_to_plan: &mut [Option<u32>],
    primitive: PrimitiveId,
) -> Result<u32, BackendError> {
    let image_resource = scene
        .images
        .get(image.0 as usize)
        .ok_or(BackendError::Plan(PlanError::InvalidReference {
            resource: "image",
            index: image.0,
        }))?;
    let mapping = scene
        .texture_mappings
        .get(mapping.0 as usize)
        .ok_or(BackendError::Plan(PlanError::InvalidReference {
            resource: "texture mapping",
            index: mapping.0,
        }))?;
    if !matches!(mapping, GpuTextureMapping::Uv { .. })
        || !matches!(filter, GpuImageFilter::Point | GpuImageFilter::Bilinear)
        || !supported_wrap(swrap)
        || !supported_wrap(twrap)
    {
        return Err(BackendError::Plan(PlanError::UnsupportedTexture {
            texture: primitive.0,
        }));
    }
    let image_index =
        if let Some(index) = image_to_plan.get(image.0 as usize).and_then(|index| *index) {
            index
        } else {
            let index = u32::try_from(images.len()).map_err(|_| {
                BackendError::Plan(PlanError::LimitExceeded {
                    resource: "image table",
                    value: u32::MAX,
                    maximum: u32::MAX,
                })
            })?;
            images.push(lower_image(image_resource)?);
            // This map is keyed by the image ID. A spectrum texture can share an image
            // with another spectrum texture while retaining different lookup parameters.
            if let Some(slot) = image_to_plan.get_mut(image.0 as usize) {
                *slot = Some(index);
            }
            index
        };
    let texture_index = u32::try_from(spectrum_textures.len()).map_err(|_| {
        BackendError::Plan(PlanError::LimitExceeded {
            resource: "spectrum texture table",
            value: u32::MAX,
            maximum: u32::MAX,
        })
    })?;
    spectrum_textures.push(SpectrumTexturePlan {
        image: image_index,
        mapping: *mapping,
        scale,
        invert,
        swrap,
        twrap,
        filter,
        spectrum_type,
    });
    Ok(texture_index)
}

fn supported_wrap(wrap: GpuImageWrapMode) -> bool {
    matches!(
        wrap,
        GpuImageWrapMode::Black | GpuImageWrapMode::Clamp | GpuImageWrapMode::Repeat
    )
}

fn lower_float_texture(
    scene: GpuSceneView<'_>,
    texture: GpuFloatTexture,
    images: &mut Vec<ImagePlan>,
    image_to_plan: &mut [Option<u32>],
) -> Result<FloatTexturePlan, BackendError> {
    match texture {
        GpuFloatTexture::Constant { value } => Ok(FloatTexturePlan::Constant(value)),
        GpuFloatTexture::Image {
            image,
            mapping,
            scale,
            invert,
            swrap,
            twrap,
            filter,
            channel,
        } => {
            let mapping =
                scene
                    .texture_mappings
                    .get(mapping.0 as usize)
                    .ok_or(BackendError::Plan(PlanError::InvalidReference {
                        resource: "texture mapping",
                        index: mapping.0,
                    }))?;
            let image_resource = scene
                .images
                .get(image.0 as usize)
                .ok_or(BackendError::Plan(PlanError::InvalidReference {
                    resource: "image",
                    index: image.0,
                }))?;
            if !matches!(mapping, GpuTextureMapping::Uv { .. })
                || !matches!(filter, GpuImageFilter::Point | GpuImageFilter::Bilinear)
                || !supported_wrap(swrap)
                || !supported_wrap(twrap)
            {
                return Err(BackendError::Plan(PlanError::UnsupportedTexture {
                    texture: image.0,
                }));
            }
            if matches!(channel, GpuFloatImageChannel::Alpha)
                && !matches!(
                    image_resource.channels,
                    GpuImageChannels::Rg | GpuImageChannels::Rgba
                )
            {
                return Err(BackendError::Plan(PlanError::UnsupportedTexture {
                    texture: image.0,
                }));
            }
            let image =
                if let Some(index) = image_to_plan.get(image.0 as usize).and_then(|index| *index) {
                    index
                } else {
                    let index = checked_len(images.len(), "image table")?;
                    images.push(lower_image(image_resource)?);
                    if let Some(slot) = image_to_plan.get_mut(image.0 as usize) {
                        *slot = Some(index);
                    }
                    index
                };
            Ok(FloatTexturePlan::Image {
                image,
                mapping: *mapping,
                scale,
                invert,
                swrap,
                twrap,
                filter,
                channel,
            })
        }
    }
}

fn lower_image(image: &GpuImageResource) -> Result<ImagePlan, BackendError> {
    let channel_count = image.channels.count();
    let texels = image_scalar_values(image, channel_count)?;
    let mip_levels = image
        .mip_levels
        .iter()
        .map(|level| {
            let offset = u32::try_from(level.texel_offset).map_err(|_| {
                BackendError::Plan(PlanError::LimitExceeded {
                    resource: "image mip texel offset",
                    value: u32::MAX,
                    maximum: u32::MAX,
                })
            })?;
            let count = u32::try_from(level.texel_count).map_err(|_| {
                BackendError::Plan(PlanError::LimitExceeded {
                    resource: "image mip texel count",
                    value: u32::MAX,
                    maximum: u32::MAX,
                })
            })?;
            let end = usize::try_from(level.texel_offset)
                .ok()
                .and_then(|offset| {
                    usize::try_from(level.texel_count)
                        .ok()
                        .and_then(|count| offset.checked_add(count))
                })
                .ok_or(BackendError::Plan(PlanError::InvalidReference {
                    resource: "image mip texels",
                    index: offset,
                }))?;
            if end > texels.len() {
                return Err(BackendError::Plan(PlanError::InvalidReference {
                    resource: "image mip texels",
                    index: end as u32,
                }));
            }
            Ok(ImageMipPlan {
                resolution: level.resolution,
                texel_offset: offset,
                texel_count: count,
            })
        })
        .collect::<Result<Vec<_>, BackendError>>()?;
    Ok(ImagePlan {
        resolution: image.resolution,
        channels: image.channels,
        mip_levels,
        texels,
    })
}

fn image_scalar_values(
    image: &GpuImageResource,
    channel_count: usize,
) -> Result<Vec<f32>, BackendError> {
    let mut values = Vec::new();
    match &image.storage {
        GpuTexelStorage::F32(data) => values.extend(data.iter().copied()),
        GpuTexelStorage::F16(data) => values.extend(
            data.iter()
                .map(|value| half::f16::from_bits(*value).to_f32()),
        ),
        GpuTexelStorage::U8(data) => {
            values.extend(data.iter().enumerate().map(|(index, value)| {
                let normalized = f32::from(*value) / 255.0;
                let channel = index % channel_count;
                if channel + 1 == channel_count && (channel_count == 2 || channel_count == 4) {
                    normalized
                } else {
                    match image.color_encoding {
                        GpuColorEncoding::Linear => normalized,
                        GpuColorEncoding::Srgb => {
                            crate::util::imageio::ColorEncoding::SRgb.to_linear(normalized)
                        }
                        GpuColorEncoding::Gamma { exponent } => normalized.powf(exponent),
                    }
                }
            }));
        }
    }
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(BackendError::Plan(PlanError::InvalidReference {
            resource: "image texel data",
            index: values.len() as u32,
        }));
    }
    Ok(values)
}

fn spectrum_rgb(
    scene: GpuSceneView<'_>,
    spectrum_id: super::super::ir::SpectrumId,
    resource_id: u32,
) -> Result<[f32; 3], BackendError> {
    match scene
        .spectra
        .get(spectrum_id.0 as usize)
        .ok_or(BackendError::Plan(PlanError::InvalidReference {
            resource: "spectrum",
            index: spectrum_id.0,
        }))? {
        GpuSpectrumResource::Constant { value } => Ok([*value, *value, *value]),
        GpuSpectrumResource::RgbAlbedo { coefficients }
        | GpuSpectrumResource::RgbUnbounded { coefficients } => {
            Ok([coefficients[0], coefficients[1], coefficients[2]])
        }
        _ => {
            return Err(BackendError::Plan(PlanError::UnsupportedMaterial {
                primitive: resource_id,
            }))
        }
    }
}

fn lower_lights(scene: GpuSceneView<'_>) -> Result<Vec<LightPlan>, BackendError> {
    let mut lights = Vec::with_capacity(scene.lights.len().max(1));
    for (index, light) in scene.lights.iter().enumerate() {
        let GpuLight::Point(point) = light else {
            return Err(BackendError::Plan(PlanError::UnsupportedLight {
                light: index as u32,
            }));
        };
        let transform = scene
            .transforms
            .get(point.render_from_light.0 as usize)
            .ok_or(BackendError::Plan(PlanError::InvalidReference {
                resource: "light transform",
                index: point.render_from_light.0,
            }))?;
        let GpuTransform::Static(transform) = transform else {
            return Err(BackendError::Plan(PlanError::UnsupportedTransform {
                transform: point.render_from_light,
            }));
        };
        let matrix = transform.render_from_object.0;
        let position = [matrix[0][3], matrix[1][3], matrix[2][3], 1.0];
        let rgb = spectrum_rgb(scene, point.intensity, index as u32)?;
        lights.push(LightPlan {
            position,
            intensity: [
                rgb[0] * point.scale,
                rgb[1] * point.scale,
                rgb[2] * point.scale,
                1.0,
            ],
        });
    }
    if lights.is_empty() {
        lights.push(LightPlan {
            position: [0.0; 4],
            intensity: [0.0; 4],
        });
    }
    Ok(lights)
}

fn checked_len(length: usize, resource: &'static str) -> Result<u32, BackendError> {
    u32::try_from(length).map_err(|_| {
        BackendError::Plan(PlanError::LimitExceeded {
            resource,
            value: u32::MAX,
            maximum: u32::MAX,
        })
    })
}

pub fn vertex_bytes(plan: &ScenePlan) -> Vec<u8> {
    plan.vertices
        .iter()
        .flat_map(|vertex| {
            vertex
                .position
                .into_iter()
                .chain(std::iter::once(0.0))
                .chain(vertex.uv)
                .chain([0.0, 0.0])
                .flat_map(|value| value.to_le_bytes())
        })
        .collect()
}

pub fn index_bytes(plan: &ScenePlan) -> Vec<u8> {
    plan.indices
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect()
}

pub fn primitive_bytes(plan: &ScenePlan) -> Vec<u8> {
    plan.primitives
        .iter()
        .flat_map(|primitive| {
            [
                primitive.first_vertex,
                primitive.first_index,
                primitive.material,
                0,
            ]
            .into_iter()
            .flat_map(u32::to_le_bytes)
        })
        .collect()
}

pub fn material_bytes(plan: &ScenePlan) -> Vec<u8> {
    plan.materials
        .iter()
        .flat_map(|material| {
            let (reflectance, texture, flags): ([f32; 4], u32, u32) = match material.reflectance {
                MaterialReflectancePlan::Constant(value) => (value, 0, 0),
                MaterialReflectancePlan::SpectrumTexture(texture) => ([0.0; 4], texture, 1),
            };
            reflectance
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .chain(texture.to_le_bytes())
                .chain(flags.to_le_bytes())
                .chain(0u32.to_le_bytes())
                .chain(0u32.to_le_bytes())
        })
        .collect()
}

/// Serializes the image-texture ABI. All offsets are u32 word offsets relative
/// to this buffer. The descriptor tables retain every mip level even though
/// the initial shader samples level zero.
pub fn texture_bytes(plan: &ScenePlan) -> Vec<u8> {
    let image_offset = 8u32;
    let mip_count: u32 = plan
        .images
        .iter()
        .map(|image| image.mip_levels.len() as u32)
        .sum();
    let mip_offset = image_offset + plan.images.len() as u32 * 8;
    let float_offset = mip_offset + mip_count * 4;
    let spectrum_offset = float_offset + plan.float_textures.len() as u32 * 8;
    let scalar_offset = spectrum_offset + plan.spectrum_textures.len() as u32 * 8;
    let mut words = vec![
        image_offset,
        plan.images.len() as u32,
        mip_offset,
        mip_count,
        float_offset,
        plan.float_textures.len() as u32,
        spectrum_offset,
        plan.spectrum_textures.len() as u32,
    ];
    let mut image_data_offset = scalar_offset;
    let mut next_mip_offset = mip_offset;
    for image in &plan.images {
        let level = image.mip_levels.first();
        let resolution = level.map_or(image.resolution, |level| level.resolution);
        words.extend([
            resolution[0],
            resolution[1],
            image.channels.count() as u32,
            next_mip_offset,
            image.mip_levels.len() as u32,
            image_data_offset,
            image.texels.len() as u32,
            0u32,
        ]);
        for level in &image.mip_levels {
            words.extend([
                level.resolution[0],
                level.resolution[1],
                image_data_offset + level.texel_offset,
                level.texel_count,
            ]);
        }
        next_mip_offset += image.mip_levels.len() as u32 * 4;
        image_data_offset += image.texels.len() as u32;
    }
    for texture in &plan.float_textures {
        match texture {
            FloatTexturePlan::Constant(value) => {
                words.extend([0, 0, 0, 0, 0, 0, 1u32 << 7, value.to_bits()])
            }
            FloatTexturePlan::Image {
                image,
                mapping,
                scale,
                invert,
                swrap,
                twrap,
                filter,
                channel,
            } => {
                let mapping = uv_mapping_words(*mapping);
                let flags = u32::from(*invert)
                    | wrap_bits(*swrap, 1)
                    | wrap_bits(*twrap, 3)
                    | (u32::from(matches!(filter, GpuImageFilter::Bilinear)) << 5)
                    | (float_channel_bits(*channel) << 8);
                words.extend([
                    *image,
                    mapping[0],
                    mapping[1],
                    mapping[2],
                    mapping[3],
                    scale.to_bits(),
                    flags,
                    0,
                ]);
            }
        }
    }
    for texture in &plan.spectrum_textures {
        let mapping = uv_mapping_words(texture.mapping);
        let flags = u32::from(texture.invert)
            | wrap_bits(texture.swrap, 1)
            | wrap_bits(texture.twrap, 3)
            | (u32::from(matches!(texture.filter, GpuImageFilter::Bilinear)) << 5)
            | (spectrum_bits(texture.spectrum_type) << 6);
        words.extend([
            texture.image,
            mapping[0],
            mapping[1],
            mapping[2],
            mapping[3],
            texture.scale.to_bits(),
            flags,
            0,
        ]);
    }
    for image in &plan.images {
        words.extend(image.texels.iter().copied().map(f32::to_bits));
    }
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn uv_mapping_words(mapping: GpuTextureMapping) -> [u32; 4] {
    match mapping {
        GpuTextureMapping::Uv { su, sv, du, dv } => {
            [su.to_bits(), sv.to_bits(), du.to_bits(), dv.to_bits()]
        }
        _ => [1.0f32.to_bits(), 1.0f32.to_bits(), 0, 0],
    }
}

fn float_channel_bits(channel: GpuFloatImageChannel) -> u32 {
    match channel {
        GpuFloatImageChannel::Channel0 => 0,
        GpuFloatImageChannel::Alpha => 1,
        GpuFloatImageChannel::RgbAverage => 2,
    }
}

fn wrap_bits(wrap: GpuImageWrapMode, shift: u32) -> u32 {
    let value = match wrap {
        GpuImageWrapMode::Black => 0,
        GpuImageWrapMode::Clamp => 1,
        GpuImageWrapMode::Repeat => 2,
        GpuImageWrapMode::OctahedralSphere => 3,
    };
    value << shift
}

fn spectrum_bits(spectrum: GpuSpectrumType) -> u32 {
    match spectrum {
        GpuSpectrumType::Albedo => 0,
        GpuSpectrumType::Unbounded => 1,
        GpuSpectrumType::Illuminant => 2,
    }
}

pub fn transform_bytes(plan: &ScenePlan) -> Vec<u8> {
    plan.transforms
        .iter()
        .flat_map(|transform| {
            transform
                .render_from_object
                .into_iter()
                .flatten()
                .flat_map(f32::to_le_bytes)
        })
        .collect()
}

pub fn light_bytes(plan: &ScenePlan) -> Vec<u8> {
    plan.lights
        .iter()
        .flat_map(|light| {
            light
                .position
                .into_iter()
                .chain(light.intensity)
                .flat_map(f32::to_le_bytes)
        })
        .collect()
}

#[allow(dead_code)]
pub struct HardwareAcceleration {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub primitive_buffer: wgpu::Buffer,
    pub material_buffer: wgpu::Buffer,
    pub transform_buffer: wgpu::Buffer,
    pub light_buffer: wgpu::Buffer,
    pub texture_buffer: wgpu::Buffer,
    pub blases: Vec<wgpu::Blas>,
    pub tlas: Tlas,
}

impl HardwareAcceleration {
    pub fn create(context: &DeviceContext, plan: &ScenePlan) -> Result<Self, BackendError> {
        let vertex_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("pbrt-r4 WebGPU vertex buffer"),
            contents: &vertex_bytes(plan),
            usage: BufferUsages::BLAS_INPUT | BufferUsages::STORAGE,
        });
        let index_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("pbrt-r4 WebGPU index buffer"),
            contents: &index_bytes(plan),
            usage: BufferUsages::BLAS_INPUT | BufferUsages::STORAGE,
        });
        let primitive_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("pbrt-r4 WebGPU primitive table"),
            contents: &primitive_bytes(plan),
            usage: BufferUsages::STORAGE,
        });
        let material_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("pbrt-r4 WebGPU material table"),
            contents: &material_bytes(plan),
            usage: BufferUsages::STORAGE,
        });
        let transform_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("pbrt-r4 WebGPU transform table"),
            contents: &transform_bytes(plan),
            usage: BufferUsages::STORAGE,
        });
        let light_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("pbrt-r4 WebGPU light table"),
            contents: &light_bytes(plan),
            usage: BufferUsages::STORAGE,
        });
        let texture_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("pbrt-r4 WebGPU texture buffer"),
            contents: &texture_bytes(plan),
            usage: BufferUsages::STORAGE,
        });

        let sizes: Vec<_> = plan
            .blases
            .iter()
            .map(|blas| BlasTriangleGeometrySizeDescriptor {
                vertex_format: VertexFormat::Float32x3,
                vertex_count: blas.vertex_count,
                index_format: Some(IndexFormat::Uint32),
                index_count: Some(blas.index_count),
                flags: AccelerationStructureGeometryFlags::OPAQUE,
            })
            .collect();
        let blases: Vec<_> = sizes
            .iter()
            .map(|size| {
                context.device.create_blas(
                    &CreateBlasDescriptor {
                        label: Some("pbrt-r4 WebGPU BLAS"),
                        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                        update_mode: AccelerationStructureUpdateMode::Build,
                    },
                    BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![size.clone()],
                    },
                )
            })
            .collect();

        let mut tlas = context.device.create_tlas(&CreateTlasDescriptor {
            label: Some("pbrt-r4 WebGPU TLAS"),
            max_instances: u32::try_from(plan.tlas_instances.len()).map_err(|_| {
                BackendError::Plan(PlanError::LimitExceeded {
                    resource: "TLAS instance count",
                    value: u32::MAX,
                    maximum: u32::MAX,
                })
            })?,
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        });
        for (index, instance) in plan.tlas_instances.iter().enumerate() {
            let blas = blases
                .get(instance.blas as usize)
                .ok_or(BackendError::Plan(PlanError::InvalidReference {
                    resource: "BLAS",
                    index: instance.blas,
                }))?;
            tlas[index] = Some(TlasInstance::new(
                blas,
                instance.transform,
                instance.custom_data,
                instance.mask,
            ));
        }

        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pbrt-r4 WebGPU acceleration build"),
            });
        let entries: Vec<_> = plan
            .blases
            .iter()
            .zip(sizes.iter())
            .zip(blases.iter())
            .map(|((blas, size), acceleration)| BlasBuildEntry {
                blas: acceleration,
                geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                    size,
                    vertex_buffer: &vertex_buffer,
                    first_vertex: blas.first_vertex,
                    vertex_stride: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    index_buffer: Some(&index_buffer),
                    first_index: Some(blas.first_index),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            })
            .collect();
        encoder.build_acceleration_structures(entries.iter(), Some(&tlas));
        context.queue.submit(Some(encoder.finish()));

        Ok(Self {
            vertex_buffer,
            index_buffer,
            primitive_buffer,
            material_buffer,
            transform_buffer,
            light_buffer,
            texture_buffer,
            blases,
            tlas,
        })
    }
}
