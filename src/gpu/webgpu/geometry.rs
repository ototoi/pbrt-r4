use super::super::ir::{
    GeometryId, GpuAreaLightBinding, GpuColorEncoding, GpuFloatImageChannel, GpuFloatTexture,
    GpuGeometry, GpuImageChannels, GpuImageFilter, GpuImageResource, GpuImageWrapMode, GpuLight,
    GpuMaterial, GpuSceneView, GpuSpectrumResource, GpuSpectrumTexture, GpuSpectrumType,
    GpuTexelStorage, GpuTextureMapping, GpuTransform, InstanceId, LightId, PrimitiveId,
};
use super::device::DeviceContext;
use super::error::{BackendError, PlanError};
use crate::util::transform::Matrix4x4;
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
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
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
        ) || self.normal_map.is_some()
            || self.displacement.is_some()
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
    pub alpha: Option<u32>,
    pub reverse_orientation: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialPlan {
    pub reflectance: MaterialReflectancePlan,
    pub normal_map: Option<u32>,
    pub displacement: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransformPlan {
    pub render_from_object: [[f32; 4]; 4],
    pub normal_from_object: [[f32; 4]; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct LightPlan {
    pub position: [f32; 4],
    pub intensity: [f32; 4],
    pub kind: u32,
    pub primitive: Option<u32>,
    pub triangle: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AreaLightOccurrence {
    light: LightId,
    primitive: u32,
    triangle: u32,
    constant_zero_alpha: bool,
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

fn primitive_transform(
    scene: GpuSceneView<'_>,
    primitive_id: PrimitiveId,
) -> Result<[[f32; 4]; 4], BackendError> {
    let primitive = scene
        .primitives
        .get(primitive_id.0 as usize)
        .ok_or(BackendError::Plan(PlanError::InvalidReference {
            resource: "primitive",
            index: primitive_id.0,
        }))?;
    let transform =
        scene
            .transforms
            .get(primitive.transform.0 as usize)
            .ok_or(BackendError::Plan(PlanError::InvalidReference {
                resource: "transform",
                index: primitive.transform.0,
            }))?;
    match transform {
        GpuTransform::Static(transform) => Ok(transform.render_from_object.0),
        GpuTransform::Animated(_) => Err(BackendError::Plan(PlanError::UnsupportedTransform {
            transform: primitive.transform,
        })),
    }
}

fn collect_instance_primitives(
    scene: GpuSceneView<'_>,
    instance_id: InstanceId,
    parent: [[f32; 4]; 4],
    output: &mut Vec<(PrimitiveId, [[f32; 4]; 4])>,
    stack: &mut Vec<InstanceId>,
) -> Result<(), BackendError> {
    if stack.contains(&instance_id) {
        return Err(BackendError::Plan(PlanError::InstanceCycle {
            instance: instance_id.0,
        }));
    }
    if stack.len() >= 64 {
        return Err(BackendError::Plan(PlanError::LimitExceeded {
            resource: "instance_depth",
            value: stack.len() as u32 + 1,
            maximum: 64,
        }));
    }
    let instance = scene
        .instances
        .get(instance_id.0 as usize)
        .ok_or(BackendError::Plan(PlanError::InvalidReference {
            resource: "instance",
            index: instance_id.0,
        }))?;
    let instance_transform =
        scene
            .transforms
            .get(instance.transform.0 as usize)
            .ok_or(BackendError::Plan(PlanError::InvalidReference {
                resource: "instance transform",
                index: instance.transform.0,
            }))?;
    let GpuTransform::Static(instance_transform) = instance_transform else {
        return Err(BackendError::Plan(PlanError::UnsupportedTransform {
            transform: instance.transform,
        }));
    };
    let instance_to_render = matrix_mul(parent, instance_transform.render_from_object.0);
    let definition = scene
        .instance_definitions
        .get(instance.definition.0 as usize)
        .ok_or(BackendError::Plan(PlanError::InvalidReference {
            resource: "instance definition",
            index: instance.definition.0,
        }))?;
    stack.push(instance_id);
    for primitive_id in &definition.primitives {
        let primitive_to_instance = primitive_transform(scene, *primitive_id)?;
        output.push((
            *primitive_id,
            matrix_mul(instance_to_render, primitive_to_instance),
        ));
    }
    for nested in &definition.instances {
        collect_instance_primitives(scene, *nested, instance_to_render, output, stack)?;
    }
    stack.pop();
    Ok(())
}

fn matrix_mul(left: [[f32; 4]; 4], right: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            result[row][column] = (0..4)
                .map(|index| left[row][index] * right[index][column])
                .sum();
        }
    }
    result
}

fn normal_transform(
    render_from_object: [[f32; 4]; 4],
    primitive: u32,
) -> Result<[[f32; 4]; 4], BackendError> {
    let matrix = Matrix4x4::from([
        render_from_object[0][0],
        render_from_object[0][1],
        render_from_object[0][2],
        render_from_object[0][3],
        render_from_object[1][0],
        render_from_object[1][1],
        render_from_object[1][2],
        render_from_object[1][3],
        render_from_object[2][0],
        render_from_object[2][1],
        render_from_object[2][2],
        render_from_object[2][3],
        render_from_object[3][0],
        render_from_object[3][1],
        render_from_object[3][2],
        render_from_object[3][3],
    ]);
    let inverse = matrix
        .inverse()
        .ok_or(BackendError::Plan(PlanError::InvalidTransform {
            primitive,
        }))?;
    let normal = inverse.transpose();
    Ok([
        [normal.m[0], normal.m[1], normal.m[2], normal.m[3]],
        [normal.m[4], normal.m[5], normal.m[6], normal.m[7]],
        [normal.m[8], normal.m[9], normal.m[10], normal.m[11]],
        [normal.m[12], normal.m[13], normal.m[14], normal.m[15]],
    ])
}

impl ScenePlan {
    pub fn supports_wavefront_min(&self, scene: GpuSceneView<'_>) -> bool {
        !scene.lights.is_empty()
            && scene
                .geometry
                .iter()
                .all(|geometry| matches!(geometry, GpuGeometry::TriangleMesh(_)))
            && scene
                .transforms
                .iter()
                .all(|transform| matches!(transform, GpuTransform::Static(_)))
            && scene.images.iter().all(|image| image.mip_levels.len() <= 1)
            && scene.lights.iter().all(|light| {
                matches!(
                    light,
                    GpuLight::Point(_) | GpuLight::UniformInfinite(_) | GpuLight::DiffuseArea(_)
                )
            })
            && self
                .lights
                .iter()
                .take(
                    self.lights
                        .first()
                        .map_or(0, |light| (light.flags >> 16) as usize),
                )
                .all(|light| light.kind <= 2)
            && self
                .primitives
                .iter()
                .all(|primitive| primitive.alpha.is_none())
            && self
                .materials
                .iter()
                .all(|material| material.normal_map.is_none() && material.displacement.is_none())
    }

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
        let mut world_entries = Vec::new();
        for primitive_id in scene.world_primitives {
            world_entries.push((*primitive_id, primitive_transform(scene, *primitive_id)?));
        }
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for instance_id in scene.world_instances {
            collect_instance_primitives(
                scene,
                *instance_id,
                identity,
                &mut world_entries,
                &mut Vec::new(),
            )?;
        }
        if world_entries.is_empty() {
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
        let mut area_light_occurrences = Vec::new();

        for texture in scene.float_textures {
            float_textures.push(lower_float_texture(
                scene,
                *texture,
                &mut images,
                &mut image_to_plan,
            )?);
        }

        for (primitive_id, object_to_render) in world_entries {
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
            let material = lower_material(
                scene,
                primitive.material,
                primitive_id,
                &mut images,
                &mut spectrum_textures,
                &mut image_to_plan,
            )?;

            let alpha_requires_uv = primitive
                .alpha
                .and_then(|texture| scene.float_textures.get(texture.0 as usize))
                .is_some_and(|texture| matches!(texture, GpuFloatTexture::Image { .. }));
            if (material.requires_uv() || alpha_requires_uv)
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
                                normal: triangle_mesh
                                    .normals
                                    .as_deref()
                                    .and_then(|normals| normals.get(index).map(|normal| normal.0))
                                    .unwrap_or([0.0, 0.0, 0.0]),
                                tangent: triangle_mesh
                                    .tangents
                                    .as_deref()
                                    .and_then(|tangents| {
                                        tangents.get(index).map(|tangent| tangent.0)
                                    })
                                    .unwrap_or([0.0, 0.0, 0.0]),
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
            let alpha = primitive.alpha.map(|texture| {
                if usize::try_from(texture.0)
                    .ok()
                    .and_then(|index| float_textures.get(index))
                    .is_none()
                {
                    Err(BackendError::Plan(PlanError::InvalidReference {
                        resource: "alpha texture",
                        index: texture.0,
                    }))
                } else {
                    Ok(texture.0)
                }
            });
            let alpha = alpha.transpose()?;
            let triangle_count = blases[blas].index_count / 3;
            let constant_zero_alpha = primitive
                .alpha
                .and_then(|texture| scene.float_textures.get(texture.0 as usize))
                .is_some_and(|texture| {
                    matches!(texture, GpuFloatTexture::Constant { value } if *value == 0.0)
                });
            match &primitive.area_light {
                GpuAreaLightBinding::None => {}
                GpuAreaLightBinding::Uniform(light) => {
                    validate_area_light(scene, *light, primitive_id.0)?;
                    area_light_occurrences.extend((0..triangle_count).map(|triangle| {
                        AreaLightOccurrence {
                            light: *light,
                            primitive: custom_data,
                            triangle,
                            constant_zero_alpha,
                        }
                    }));
                }
                GpuAreaLightBinding::PerElement(lights) => {
                    if lights.len() != triangle_count as usize {
                        return Err(BackendError::Plan(PlanError::InvalidAreaLightBinding {
                            primitive: primitive_id.0,
                            expected: triangle_count,
                            actual: checked_len(lights.len(), "area light binding")?,
                        }));
                    }
                    for (triangle, light) in lights.iter().copied().enumerate() {
                        validate_area_light(scene, light, primitive_id.0)?;
                        area_light_occurrences.push(AreaLightOccurrence {
                            light,
                            primitive: custom_data,
                            triangle: triangle as u32,
                            constant_zero_alpha,
                        });
                    }
                }
            }
            primitives.push(PrimitivePlan {
                first_vertex: blases[blas].first_vertex,
                first_index: blases[blas].first_index,
                triangle_count,
                material: custom_data,
                alpha,
                reverse_orientation: primitive.reverse_orientation,
            });
            materials.push(material);
            transforms.push(TransformPlan {
                render_from_object: object_to_render,
                normal_from_object: normal_transform(object_to_render, primitive_id.0)?,
            });
            tlas_instances.push(TlasInstancePlan {
                blas: blas as u32,
                transform: tlas_transform(object_to_render),
                custom_data,
                mask: u8::MAX,
            });
        }

        let lights = lower_lights(scene, &area_light_occurrences)?;
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
    let displacement = diffuse
        .displacement
        .map(|texture| {
            if scene.float_textures.get(texture.0 as usize).is_none() {
                return Err(BackendError::Plan(PlanError::InvalidReference {
                    resource: "material displacement texture",
                    index: texture.0,
                }));
            }
            Ok(texture.0)
        })
        .transpose()?;
    let normal_map = diffuse
        .normal_map
        .map(|image| lower_normal_map_image(scene, image, images, image_to_plan, primitive))
        .transpose()?;
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
    Ok(MaterialPlan {
        reflectance,
        normal_map,
        displacement,
    })
}

fn lower_normal_map_image(
    scene: GpuSceneView<'_>,
    image: super::super::ir::ImageId,
    images: &mut Vec<ImagePlan>,
    image_to_plan: &mut [Option<u32>],
    primitive: PrimitiveId,
) -> Result<u32, BackendError> {
    let image_resource = scene
        .images
        .get(image.0 as usize)
        .ok_or(BackendError::Plan(PlanError::InvalidReference {
            resource: "normal map image",
            index: image.0,
        }))?;
    if !matches!(
        image_resource.channels,
        GpuImageChannels::Rgb | GpuImageChannels::Rgba
    ) {
        return Err(BackendError::Plan(PlanError::UnsupportedTexture {
            texture: primitive.0,
        }));
    }
    if let Some(index) = image_to_plan.get(image.0 as usize).and_then(|index| *index) {
        return Ok(index);
    }
    let index = checked_len(images.len(), "image table")?;
    images.push(lower_image(image_resource)?);
    if let Some(slot) = image_to_plan.get_mut(image.0 as usize) {
        *slot = Some(index);
    }
    Ok(index)
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
        || !supported_filter(filter)
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

fn supported_filter(filter: GpuImageFilter) -> bool {
    matches!(
        filter,
        GpuImageFilter::Point
            | GpuImageFilter::Bilinear
            | GpuImageFilter::Trilinear
            | GpuImageFilter::Ewa { .. }
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
                || !supported_filter(filter)
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

fn validate_area_light(
    scene: GpuSceneView<'_>,
    light: LightId,
    primitive: u32,
) -> Result<(), BackendError> {
    match scene.lights.get(light.0 as usize) {
        Some(GpuLight::DiffuseArea(_)) => Ok(()),
        _ => Err(BackendError::Plan(PlanError::UnsupportedAreaLight {
            primitive,
        })),
    }
}

fn area_emission_rgb(scene: GpuSceneView<'_>, light_id: LightId) -> Result<[f32; 3], BackendError> {
    let Some(GpuLight::DiffuseArea(area)) = scene.lights.get(light_id.0 as usize) else {
        return Err(BackendError::Plan(PlanError::UnsupportedLight {
            light: light_id.0,
        }));
    };
    let texture = scene
        .spectrum_textures
        .get(area.emission.0 as usize)
        .ok_or(BackendError::Plan(PlanError::InvalidReference {
            resource: "area light emission texture",
            index: area.emission.0,
        }))?;
    let GpuSpectrumTexture::Constant { value } = texture else {
        return Err(BackendError::Plan(PlanError::UnsupportedLight {
            light: light_id.0,
        }));
    };
    spectrum_rgb(scene, *value, light_id.0)
}

fn lower_lights(
    scene: GpuSceneView<'_>,
    area_light_occurrences: &[AreaLightOccurrence],
) -> Result<Vec<LightPlan>, BackendError> {
    let mut lights = Vec::with_capacity(scene.lights.len().max(1));
    let mut area_sources = Vec::new();
    for (index, light) in scene.lights.iter().enumerate() {
        match light {
            GpuLight::Point(point) => {
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
                    kind: 0,
                    primitive: None,
                    triangle: 0,
                    flags: 0,
                });
            }
            GpuLight::UniformInfinite(infinite) => {
                let rgb = spectrum_rgb(scene, infinite.radiance, index as u32)?;
                lights.push(LightPlan {
                    position: [0.0; 4],
                    intensity: [
                        rgb[0] * infinite.scale,
                        rgb[1] * infinite.scale,
                        rgb[2] * infinite.scale,
                        1.0,
                    ],
                    kind: 1,
                    primitive: None,
                    triangle: 0,
                    flags: 0,
                });
            }
            GpuLight::DiffuseArea(area) => {
                let light_id = LightId(index as u32);
                let rgb = area_emission_rgb(scene, light_id)?;
                let source_index = checked_len(lights.len(), "light source table")?;
                lights.push(LightPlan {
                    position: [0.0; 4],
                    intensity: [
                        rgb[0] * area.scale,
                        rgb[1] * area.scale,
                        rgb[2] * area.scale,
                        1.0,
                    ],
                    kind: 2,
                    primitive: Some(0),
                    triangle: 0,
                    flags: u32::from(area.two_sided),
                });
                area_sources.push((source_index, light_id));
            }
        }
    }

    let source_count = checked_len(lights.len(), "light source table")?;
    for (source_index, light_id) in area_sources {
        let geometry_offset = checked_len(lights.len(), "area light geometry table")?;
        let mut geometry_count = 0u32;
        let mut all_constant_zero_alpha = true;
        for occurrence in area_light_occurrences
            .iter()
            .filter(|occurrence| occurrence.light == light_id)
        {
            all_constant_zero_alpha &= occurrence.constant_zero_alpha;
            lights.push(LightPlan {
                position: [0.0; 4],
                intensity: [0.0; 4],
                kind: 3,
                primitive: Some(occurrence.primitive),
                triangle: occurrence.triangle,
                flags: u32::from(occurrence.constant_zero_alpha) << 1,
            });
            geometry_count += 1;
        }
        lights[source_index as usize].primitive = Some(geometry_offset);
        lights[source_index as usize].triangle = geometry_count;
        if geometry_count != 0 && all_constant_zero_alpha {
            lights[source_index as usize].flags |= 1 << 1;
        }
    }
    if lights.is_empty() {
        lights.push(LightPlan {
            position: [0.0; 4],
            intensity: [0.0; 4],
            kind: 0,
            primitive: None,
            triangle: 0,
            flags: 0,
        });
    }
    if source_count > 0 {
        if source_count > 0xffff {
            return Err(BackendError::Plan(PlanError::LimitExceeded {
                resource: "light source table",
                value: source_count,
                maximum: 0xffff,
            }));
        }
        lights[0].flags |= source_count << 16;
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
                .chain(vertex.normal)
                .chain(std::iter::once(0.0))
                .chain(vertex.tangent)
                .chain(std::iter::once(0.0))
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
                primitive.alpha.unwrap_or(u32::MAX),
                u32::MAX,
                u32::from(primitive.reverse_orientation),
                0,
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
            let normal_map = material.normal_map.unwrap_or(u32::MAX);
            let displacement = material.displacement.unwrap_or(u32::MAX);
            let flags = flags
                | u32::from(material.normal_map.is_some()) << 1
                | u32::from(material.displacement.is_some()) << 2;
            reflectance
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .chain(texture.to_le_bytes())
                .chain(normal_map.to_le_bytes())
                .chain(displacement.to_le_bytes())
                .chain(flags.to_le_bytes())
        })
        .collect()
}

/// Serializes the image-texture ABI. All offsets are u32 word offsets relative
/// to this buffer. The descriptor tables retain every mip level for
/// shader-side MIPMap filtering.
pub fn texture_bytes(plan: &ScenePlan) -> Vec<u8> {
    const EWA_LUT_SIZE: u32 = 128;
    let image_offset = 8 + EWA_LUT_SIZE;
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
    let exp_minus_two = (-2.0f32).exp();
    words.extend((0..EWA_LUT_SIZE).map(|index| {
        let radius_squared = index as f32 / (EWA_LUT_SIZE - 1) as f32;
        ((-2.0 * radius_squared).exp() - exp_minus_two).to_bits()
    }));
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
                    | (filter_bits(*filter) << 5)
                    | (float_channel_bits(*channel) << 9);
                words.extend([
                    *image,
                    mapping[0],
                    mapping[1],
                    mapping[2],
                    mapping[3],
                    scale.to_bits(),
                    flags,
                    max_anisotropy_bits(*filter),
                ]);
            }
        }
    }
    for texture in &plan.spectrum_textures {
        let mapping = uv_mapping_words(texture.mapping);
        let flags = u32::from(texture.invert)
            | wrap_bits(texture.swrap, 1)
            | wrap_bits(texture.twrap, 3)
            | (filter_bits(texture.filter) << 5)
            | (spectrum_bits(texture.spectrum_type) << 7);
        words.extend([
            texture.image,
            mapping[0],
            mapping[1],
            mapping[2],
            mapping[3],
            texture.scale.to_bits(),
            flags,
            max_anisotropy_bits(texture.filter),
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

fn filter_bits(filter: GpuImageFilter) -> u32 {
    match filter {
        GpuImageFilter::Point => 0,
        GpuImageFilter::Bilinear => 1,
        GpuImageFilter::Trilinear => 2,
        GpuImageFilter::Ewa { .. } => 3,
    }
}

fn max_anisotropy_bits(filter: GpuImageFilter) -> u32 {
    match filter {
        GpuImageFilter::Ewa { max_anisotropy } => max_anisotropy.to_bits(),
        _ => 0,
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
    let mut bytes = Vec::with_capacity(plan.transforms.len() * 2 * 16 * size_of::<f32>());
    for transform in &plan.transforms {
        append_wgsl_matrix(&mut bytes, transform.render_from_object);
        append_wgsl_matrix(&mut bytes, transform.normal_from_object);
    }
    bytes
}

pub fn append_wgsl_matrix(bytes: &mut Vec<u8>, matrix: [[f32; 4]; 4]) {
    for column in 0..4 {
        for row in &matrix {
            bytes.extend(row[column].to_le_bytes());
        }
    }
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
                .chain(light.kind.to_le_bytes())
                .chain(light.primitive.unwrap_or(u32::MAX).to_le_bytes())
                .chain(light.triangle.to_le_bytes())
                .chain(light.flags.to_le_bytes())
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
                // Keep triangle candidates visible to the shader.  Alpha-masked
                // primitives must be rejected or confirmed per candidate.
                flags: AccelerationStructureGeometryFlags::empty(),
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
                    vertex_stride: std::mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
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
