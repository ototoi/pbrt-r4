use bytemuck::{cast_slice, Zeroable};
use wgpu::util::DeviceExt;

use crate::gpu::ir::flat;
use crate::util::error::PbrtError;

use super::abi::{
    camera_uniform, inverse_transpose_linear, row_major_to_columns, viewport_uniform, AreaLight,
    Geometry, Instance, LightRecord, PointLight, TriangleDistributionEntry, Vertex,
    ViewportUniform, LIGHT_KIND_AREA, LIGHT_KIND_POINT,
};
use super::acceleration::{self, Acceleration};
use super::material::MaterialKind;
use super::output::Output;

pub struct Scene {
    pub camera: super::abi::CameraUniform,
    pub viewport: ViewportUniform,
    pub output: Output,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub geometry_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub material_buffer: wgpu::Buffer,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<Instance>,
    pub materials: Vec<super::abi::Material>,
    pub point_lights: Vec<PointLight>,
    pub area_lights: Vec<AreaLight>,
    pub light_records: Vec<LightRecord>,
    pub area_light_buffer: wgpu::Buffer,
    pub light_record_buffer: wgpu::Buffer,
    pub triangle_distribution_buffer: wgpu::Buffer,
    pub render_settings: flat::RenderSettings,
    pub acceleration: Acceleration,
}

impl Scene {
    pub fn from_flat(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        flat: flat::Scene,
    ) -> Result<Self, PbrtError> {
        if flat.output.filename.is_empty() {
            return Err(PbrtError::error(
                "WebGPU output filename must not be empty.",
            ));
        }
        let (vertices, geometries, indices) = convert_geometry(&flat)?;
        let instances = flat
            .instances
            .iter()
            .enumerate()
            .map(|(index, instance)| {
                if instance.geometry as usize >= geometries.len() {
                    return Err(PbrtError::error(&format!(
                        "Flat instance {index} references an invalid geometry."
                    )));
                }
                if instance.material as usize >= flat.materials.len() {
                    return Err(PbrtError::error(&format!(
                        "Flat instance {index} references an invalid material."
                    )));
                }
                let label = format!("Flat instance {index}");
                Ok(Instance {
                    geometry: instance.geometry,
                    material: instance.material,
                    area_light: resolve_area_light_index(instance.area_light, &flat)?,
                    padding: 0,
                    world_from_object: row_major_to_columns(instance.transform),
                    normal_from_object: inverse_transpose_linear(instance.transform, &label)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let materials = flat
            .materials
            .iter()
            .map(|material| {
                Ok(super::abi::Material {
                    kind_tag: MaterialKind::from_flat(&material.kind)?.tag(),
                    padding: [0; 3],
                })
            })
            .collect::<Result<Vec<_>, PbrtError>>()?;
        let point_lights = flat
            .point_lights
            .iter()
            .map(|light| PointLight {
                position: [light.position[0], light.position[1], light.position[2], 1.0],
                intensity: [
                    light.intensity[0],
                    light.intensity[1],
                    light.intensity[2],
                    0.0,
                ],
            })
            .collect::<Vec<_>>();
        let mut area_lights = flat
            .area_lights
            .iter()
            .map(|light| AreaLight {
                instance: light.instance,
                two_sided: u32::from(light.two_sided),
                emission: [light.emission[0], light.emission[1], light.emission[2], 0.0],
                total_area: 0.0,
                triangle_distribution_offset: u32::MAX,
                triangle_distribution_count: 0,
                padding: [0; 3],
            })
            .collect::<Vec<_>>();
        let mut triangle_distributions = Vec::new();
        build_triangle_distributions(
            &mut area_lights,
            &flat.instances,
            &geometries,
            &flat.geometries,
            &vertices,
            &flat.indices,
            &mut triangle_distributions,
        )?;
        let light_records = flat
            .lights
            .iter()
            .map(|record| LightRecord {
                kind: match record.kind {
                    flat::LightKind::Point => LIGHT_KIND_POINT,
                    flat::LightKind::Area => LIGHT_KIND_AREA,
                },
                payload: record.payload,
                padding: [0; 2],
            })
            .collect::<Vec<_>>();
        for (index, record) in flat.lights.iter().enumerate() {
            match record.kind {
                flat::LightKind::Point if record.payload as usize >= point_lights.len() => {
                    return Err(PbrtError::error(&format!(
                        "Flat light record {index} references an invalid point light."
                    )));
                }
                flat::LightKind::Area if record.payload as usize >= area_lights.len() => {
                    return Err(PbrtError::error(&format!(
                        "Flat light record {index} references an invalid area light."
                    )));
                }
                _ => {}
            }
        }
        for (index, instance) in flat.instances.iter().enumerate() {
            if instance.area_light != flat::INVALID_INDEX {
                let light = flat
                    .lights
                    .get(instance.area_light as usize)
                    .ok_or_else(|| {
                        PbrtError::error(&format!(
                            "Flat instance {index} references an invalid area-light handle."
                        ))
                    })?;
                if light.kind != flat::LightKind::Area {
                    return Err(PbrtError::error(&format!(
                        "Flat instance {index} area-light handle does not reference an area light."
                    )));
                }
            }
        }
        let material_words = std::mem::size_of::<super::abi::Material>()
            .checked_div(std::mem::size_of::<u32>())
            .ok_or_else(|| PbrtError::error("WebGPU material ABI is not word-aligned."))?;
        let light_words = std::mem::size_of::<PointLight>()
            .checked_div(std::mem::size_of::<u32>())
            .ok_or_else(|| PbrtError::error("WebGPU point-light ABI is not word-aligned."))?;
        let light_record_words = std::mem::size_of::<LightRecord>()
            .checked_div(std::mem::size_of::<u32>())
            .ok_or_else(|| PbrtError::error("WebGPU light-record ABI is not word-aligned."))?;
        let area_light_words = std::mem::size_of::<AreaLight>()
            .checked_div(std::mem::size_of::<u32>())
            .ok_or_else(|| PbrtError::error("WebGPU area-light ABI is not word-aligned."))?;
        let material_words_total = materials
            .len()
            .checked_mul(material_words)
            .ok_or_else(|| PbrtError::error("WebGPU material buffer size overflowed."))?;
        let area_light_data_offset = material_words_total
            .checked_add(
                light_records
                    .len()
                    .checked_mul(light_record_words)
                    .ok_or_else(|| {
                        PbrtError::error("WebGPU light-record buffer offset overflowed.")
                    })?,
            )
            .and_then(|offset| offset.checked_add(point_lights.len().checked_mul(light_words)?))
            .ok_or_else(|| PbrtError::error("WebGPU area-light buffer offset overflowed."))?;
        let triangle_distribution_base = area_light_data_offset
            .checked_add(
                area_lights
                    .len()
                    .checked_mul(area_light_words)
                    .ok_or_else(|| PbrtError::error("WebGPU area-light buffer size overflowed."))?,
            )
            .ok_or_else(|| PbrtError::error("WebGPU triangle distribution offset overflowed."))?;
        let triangle_distribution_words =
            std::mem::size_of::<TriangleDistributionEntry>() / std::mem::size_of::<u32>();
        for area_light in &mut area_lights {
            if area_light.triangle_distribution_offset != u32::MAX {
                let offset = usize::try_from(area_light.triangle_distribution_offset)
                    .map_err(|_| PbrtError::error("Invalid triangle distribution offset."))?
                    .checked_mul(triangle_distribution_words)
                    .and_then(|offset| triangle_distribution_base.checked_add(offset))
                    .and_then(|offset| u32::try_from(offset).ok())
                    .ok_or_else(|| {
                        PbrtError::error("WebGPU triangle distribution offset overflowed.")
                    })?;
                area_light.triangle_distribution_offset = offset;
            }
        }
        let camera = camera_uniform(&flat.camera, &flat.viewport)?;
        let viewport = viewport_uniform(
            &flat.viewport,
            &flat.render_settings,
            materials.len(),
            light_records.len(),
            area_light_data_offset,
        )?;
        if vertices.is_empty() || indices.is_empty() || instances.is_empty() || materials.is_empty()
        {
            return Err(PbrtError::error(
                "WebGPU primary-ray rendering requires non-empty geometry, instances, and materials.",
            ));
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 vertex SBO"),
            contents: cast_slice(&vertices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 local index SBO"),
            contents: cast_slice(&indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        });
        let geometry_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 geometry SBO"),
            contents: cast_slice(&geometries),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 instance SBO"),
            contents: cast_slice(&instances),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut material_light_data = Vec::<u32>::with_capacity(
            material_words_total
                .checked_add(
                    light_records
                        .len()
                        .checked_mul(light_record_words)
                        .ok_or_else(|| {
                            PbrtError::error("WebGPU light-record buffer size overflowed.")
                        })?,
                )
                .and_then(|size| size.checked_add(point_lights.len().checked_mul(light_words)?))
                .and_then(|size| size.checked_add(area_lights.len().checked_mul(area_light_words)?))
                .and_then(|size| {
                    size.checked_add(
                        triangle_distributions
                            .len()
                            .checked_mul(triangle_distribution_words)?,
                    )
                })
                .ok_or_else(|| PbrtError::error("WebGPU material/light buffer size overflowed."))?,
        );
        for material in &materials {
            material_light_data.extend_from_slice(cast_slice(std::slice::from_ref(material)));
        }
        for record in &light_records {
            material_light_data.extend_from_slice(cast_slice(std::slice::from_ref(record)));
        }
        for light in &point_lights {
            material_light_data.extend_from_slice(cast_slice(std::slice::from_ref(light)));
        }
        for light in &area_lights {
            material_light_data.extend_from_slice(cast_slice(std::slice::from_ref(light)));
        }
        material_light_data.extend_from_slice(cast_slice(&triangle_distributions));
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 material and point-light SBO"),
            contents: cast_slice(&material_light_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let area_light_storage = if area_lights.is_empty() {
            vec![AreaLight::zeroed()]
        } else {
            area_lights.clone()
        };
        let light_record_storage = if light_records.is_empty() {
            vec![LightRecord::zeroed()]
        } else {
            light_records.clone()
        };
        let area_light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 area-light SBO"),
            contents: bytemuck::cast_slice(&area_light_storage),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let light_record_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 light-record SBO"),
            contents: bytemuck::cast_slice(&light_record_storage),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let triangle_distribution_storage = if triangle_distributions.is_empty() {
            vec![TriangleDistributionEntry::zeroed()]
        } else {
            triangle_distributions.clone()
        };
        let triangle_distribution_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 triangle-distribution SBO"),
                contents: bytemuck::cast_slice(&triangle_distribution_storage),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let acceleration = acceleration::build(
            device,
            queue,
            &vertex_buffer,
            &index_buffer,
            &geometries,
            &instances,
            &flat.instances,
        )?;
        Ok(Self {
            camera,
            viewport,
            output: Output::from_flat(flat.output),
            vertex_buffer,
            index_buffer,
            geometry_buffer,
            instance_buffer,
            material_buffer,
            geometries,
            instances,
            materials,
            point_lights,
            area_lights,
            light_records,
            area_light_buffer,
            light_record_buffer,
            triangle_distribution_buffer,
            render_settings: flat.render_settings,
            acceleration,
        })
    }

    pub fn replace_material_kind(&mut self, queue: &wgpu::Queue, kind: MaterialKind) {
        for material in &mut self.materials {
            material.kind_tag = kind.tag();
        }
        queue.write_buffer(
            &self.material_buffer,
            0,
            bytemuck::cast_slice(&self.materials),
        );
    }
}

fn resolve_area_light_index(handle: u32, flat: &flat::Scene) -> Result<u32, PbrtError> {
    if handle == flat::INVALID_INDEX {
        return Ok(flat::INVALID_INDEX);
    }
    let record = flat
        .lights
        .get(handle as usize)
        .ok_or_else(|| PbrtError::error("Flat instance references an invalid light handle."))?;
    if record.kind != flat::LightKind::Area {
        return Err(PbrtError::error(
            "Flat instance area-light handle does not reference an area light.",
        ));
    }
    if record.payload as usize >= flat.area_lights.len() {
        return Err(PbrtError::error(
            "Flat instance area-light record references an invalid area light.",
        ));
    }
    Ok(record.payload)
}

fn convert_geometry(
    flat: &flat::Scene,
) -> Result<(Vec<Vertex>, Vec<Geometry>, Vec<u32>), PbrtError> {
    let vertices = flat
        .vertices
        .iter()
        .map(|vertex| {
            if !vertex.position.iter().all(|value| value.is_finite()) {
                return Err(PbrtError::error(
                    "Flat vertex position contains a non-finite value.",
                ));
            }
            if !vertex.uv.iter().all(|value| value.is_finite()) {
                return Err(PbrtError::error(
                    "Flat vertex UV contains a non-finite value.",
                ));
            }
            if !vertex.normal.iter().all(|value| value.is_finite())
                || !vertex.tangent.iter().all(|value| value.is_finite())
            {
                return Err(PbrtError::error(
                    "Flat vertex normal or tangent contains a non-finite value.",
                ));
            }
            Ok(Vertex {
                position: [
                    vertex.position[0],
                    vertex.position[1],
                    vertex.position[2],
                    1.0,
                ],
                normal: [vertex.normal[0], vertex.normal[1], vertex.normal[2], 0.0],
                tangent: [vertex.tangent[0], vertex.tangent[1], vertex.tangent[2], 0.0],
                uv: vertex.uv,
                padding: [0; 2],
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut local_indices = Vec::new();
    let mut geometries = Vec::with_capacity(flat.geometries.len());
    for (index, geometry) in flat.geometries.iter().enumerate() {
        let vertex_end = geometry
            .first_vertex
            .checked_add(geometry.vertex_count)
            .ok_or_else(|| {
                PbrtError::error(&format!("Flat geometry {index} vertex range overflowed."))
            })?;
        let index_end = geometry
            .first_index
            .checked_add(geometry.index_count)
            .ok_or_else(|| {
                PbrtError::error(&format!("Flat geometry {index} index range overflowed."))
            })?;
        if geometry.index_count == 0 || geometry.index_count % 3 != 0 {
            return Err(PbrtError::error(&format!(
                "Flat geometry {index} must contain a non-empty multiple of three indices."
            )));
        }
        if vertex_end as usize > vertices.len() || index_end as usize > flat.indices.len() {
            return Err(PbrtError::error(&format!(
                "Flat geometry {index} range is out of bounds."
            )));
        }
        let index_offset = u32::try_from(local_indices.len()).map_err(|_| {
            PbrtError::error(&format!(
                "Flat geometry {index} index offset does not fit in u32."
            ))
        })?;
        for &absolute_index in &flat.indices[geometry.first_index as usize..index_end as usize] {
            if absolute_index < geometry.first_vertex || absolute_index >= vertex_end {
                return Err(PbrtError::error(&format!(
                    "Flat geometry {index} contains an index outside its vertex range."
                )));
            }
            local_indices.push(absolute_index - geometry.first_vertex);
        }
        for triangle in local_indices[index_offset as usize..].chunks_exact(3) {
            let p0 = vertices[geometry.first_vertex as usize + triangle[0] as usize].position;
            let p1 = vertices[geometry.first_vertex as usize + triangle[1] as usize].position;
            let p2 = vertices[geometry.first_vertex as usize + triangle[2] as usize].position;
            let edge0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let edge1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                edge0[1] * edge1[2] - edge0[2] * edge1[1],
                edge0[2] * edge1[0] - edge0[0] * edge1[2],
                edge0[0] * edge1[1] - edge0[1] * edge1[0],
            ];
            let norm_squared = cross.iter().map(|value| value * value).sum::<f32>();
            if !cross.iter().all(|value| value.is_finite())
                || !norm_squared.is_finite()
                || norm_squared == 0.0
            {
                return Err(PbrtError::error(&format!(
                    "Flat geometry {index} contains a zero-area or non-finite triangle."
                )));
            }
        }
        geometries.push(Geometry {
            vertex_offset: geometry.first_vertex,
            vertex_count: geometry.vertex_count,
            index_offset,
            index_count: geometry.index_count,
        });
    }
    Ok((vertices, geometries, local_indices))
}

fn build_triangle_distributions(
    area_lights: &mut [AreaLight],
    flat_instances: &[flat::Instance],
    geometries: &[Geometry],
    flat_geometries: &[flat::Geometry],
    vertices: &[Vertex],
    flat_indices: &[u32],
    distributions: &mut Vec<TriangleDistributionEntry>,
) -> Result<(), PbrtError> {
    for (area_index, area_light) in area_lights.iter_mut().enumerate() {
        let instance = flat_instances
            .get(area_light.instance as usize)
            .ok_or_else(|| {
                PbrtError::error(&format!(
                    "WebGPU area light {area_index} references an invalid instance."
                ))
            })?;
        let geometry_index = usize::try_from(instance.geometry).map_err(|_| {
            PbrtError::error(&format!(
                "WebGPU area light {area_index} has an invalid geometry index."
            ))
        })?;
        let geometry = geometries.get(geometry_index).ok_or_else(|| {
            PbrtError::error(&format!(
                "WebGPU area light {area_index} references an invalid geometry."
            ))
        })?;
        let flat_geometry = flat_geometries.get(geometry_index).ok_or_else(|| {
            PbrtError::error(&format!(
                "WebGPU area light {area_index} has no source geometry."
            ))
        })?;
        let triangle_count = usize::try_from(geometry.index_count / 3)
            .map_err(|_| PbrtError::error("WebGPU triangle count does not fit in usize."))?;
        let distribution_offset = u32::try_from(distributions.len()).map_err(|_| {
            PbrtError::error("WebGPU triangle distribution offset does not fit in u32.")
        })?;
        let mut areas = Vec::with_capacity(triangle_count);
        let mut total_area = 0.0f32;
        for primitive in 0..triangle_count {
            let index_start = usize::try_from(flat_geometry.first_index)
                .ok()
                .and_then(|start| start.checked_add(primitive.checked_mul(3)?))
                .ok_or_else(|| {
                    PbrtError::error(&format!(
                        "WebGPU area light {area_index} triangle index overflowed."
                    ))
                })?;
            let [i0, i1, i2] = flat_indices
                .get(index_start..index_start + 3)
                .and_then(|indices| indices.try_into().ok())
                .ok_or_else(|| {
                    PbrtError::error(&format!(
                        "WebGPU area light {area_index} triangle index is out of bounds."
                    ))
                })?;
            let positions = [i0, i1, i2].map(|index| {
                let vertex = vertices.get(index as usize).ok_or_else(|| {
                    PbrtError::error(&format!(
                        "WebGPU area light {area_index} references an invalid vertex."
                    ))
                })?;
                Ok(transform_point(instance.transform, vertex.position))
            });
            let [p0, p1, p2] = positions
                .into_iter()
                .collect::<Result<Vec<_>, PbrtError>>()?
                .try_into()
                .map_err(|_| PbrtError::error("WebGPU triangle position conversion failed."))?;
            let edge0 = sub(p1, p0);
            let edge1 = sub(p2, p0);
            let cross = cross(edge0, edge1);
            let area = 0.5 * length(cross);
            if !area.is_finite() || area == 0.0 {
                continue;
            }
            total_area += area;
            areas.push((
                u32::try_from(primitive)
                    .map_err(|_| PbrtError::error("WebGPU primitive index does not fit in u32."))?,
                area,
            ));
        }
        if areas.is_empty() || !total_area.is_finite() || total_area <= 0.0 {
            return Err(PbrtError::error(&format!(
                "WebGPU area light {area_index} has no valid triangles."
            )));
        }
        let mut cumulative = 0.0f32;
        let mut previous_cdf = 0.0f32;
        for (entry_index, (primitive, area)) in areas.iter().enumerate() {
            cumulative += *area / total_area;
            let cdf = if entry_index + 1 == areas.len() {
                1.0
            } else {
                cumulative
            };
            if !cdf.is_finite() || (entry_index > 0 && cdf <= previous_cdf) {
                return Err(PbrtError::error(&format!(
                    "WebGPU area light {area_index} triangle CDF is not strictly increasing."
                )));
            }
            distributions.push(TriangleDistributionEntry {
                primitive: *primitive,
                cdf,
                padding: [0; 2],
            });
            previous_cdf = cdf;
        }
        area_light.total_area = total_area;
        area_light.triangle_distribution_offset = distribution_offset;
        area_light.triangle_distribution_count = u32::try_from(areas.len()).map_err(|_| {
            PbrtError::error("WebGPU triangle distribution count does not fit in u32.")
        })?;
    }
    Ok(())
}

fn transform_point(matrix: [f32; 16], point: [f32; 4]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3],
        matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7],
        matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11],
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(value: [f32; 3]) -> f32 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}
