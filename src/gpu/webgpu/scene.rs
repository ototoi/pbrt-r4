use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use crate::gpu::ir::flat;
use crate::util::error::PbrtError;

use super::abi::{
    camera_uniform, inverse_transpose_linear, row_major_to_columns, viewport_uniform, AreaLight,
    Geometry, Instance, LightRecord, PointLight, Vertex, ViewportUniform, LIGHT_KIND_AREA,
    LIGHT_KIND_POINT,
};
use super::acceleration::{self, Acceleration};
use super::light::triangle_world_area;
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
                validate_instance_area_lights(index, instance, &flat)?;
                let label = format!("Flat instance {index}");
                Ok(Instance {
                    geometry: instance.geometry,
                    material: instance.material,
                    first_area_light: instance.first_area_light,
                    orientation_flags: u32::from(instance.reverse_orientation)
                        | (u32::from(flat::transform_swaps_handedness(instance.transform)) << 1),
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
                primitive: light.primitive,
                reserved: 0,
                padding: [0; 3],
            })
            .collect::<Vec<_>>();
        build_area_light_areas(
            &mut area_lights,
            &flat.instances,
            &geometries,
            &flat.geometries,
            &vertices,
            &flat.indices,
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
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 material and point-light SBO"),
            contents: cast_slice(&material_light_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

fn validate_instance_area_lights(
    instance_index: usize,
    instance: &flat::Instance,
    flat: &flat::Scene,
) -> Result<(), PbrtError> {
    if instance.first_area_light == flat::INVALID_INDEX {
        return Ok(());
    }
    let geometry = flat
        .geometries
        .get(instance.geometry as usize)
        .ok_or_else(|| PbrtError::error("Flat area-light instance has an invalid geometry."))?;
    let triangle_count = geometry.index_count / 3;
    if triangle_count == 0 {
        return Err(PbrtError::error(
            "Flat area-light instance geometry contains no triangles.",
        ));
    }
    for primitive in 0..triangle_count {
        let handle = instance
            .first_area_light
            .checked_add(primitive)
            .ok_or_else(|| PbrtError::error("Flat area-light handle overflowed."))?;
        let record = flat.lights.get(handle as usize).ok_or_else(|| {
            PbrtError::error(&format!(
                "Flat instance {instance_index} has an incomplete area-light range."
            ))
        })?;
        if record.kind != flat::LightKind::Area {
            return Err(PbrtError::error(&format!(
                "Flat instance {instance_index} area-light range contains a non-area light."
            )));
        }
        let area_light = flat
            .area_lights
            .get(record.payload as usize)
            .ok_or_else(|| {
                PbrtError::error(&format!(
                    "Flat instance {instance_index} references an invalid area-light payload."
                ))
            })?;
        if area_light.instance as usize != instance_index || area_light.primitive != primitive {
            return Err(PbrtError::error(&format!(
                "Flat instance {instance_index} area-light range does not match its triangles."
            )));
        }
    }
    Ok(())
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

fn build_area_light_areas(
    area_lights: &mut [AreaLight],
    flat_instances: &[flat::Instance],
    geometries: &[Geometry],
    flat_geometries: &[flat::Geometry],
    vertices: &[Vertex],
    flat_indices: &[u32],
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
        let primitive = usize::try_from(area_light.primitive)
            .map_err(|_| PbrtError::error("WebGPU primitive index does not fit in usize."))?;
        if area_light.primitive >= geometry.index_count / 3 {
            return Err(PbrtError::error(&format!(
                "WebGPU area light {area_index} references an invalid primitive."
            )));
        }
        let index_start = usize::try_from(flat_geometry.first_index)
            .ok()
            .and_then(|start| start.checked_add(primitive.checked_mul(3)?))
            .ok_or_else(|| PbrtError::error("WebGPU area-light index overflowed."))?;
        let [i0, i1, i2] = flat_indices
            .get(index_start..index_start + 3)
            .and_then(|indices| indices.try_into().ok())
            .ok_or_else(|| PbrtError::error("WebGPU area-light index is out of bounds."))?;
        let mut positions = [[0.0; 4]; 3];
        for (position, index) in positions.iter_mut().zip([i0, i1, i2]) {
            let vertex = vertices.get(index as usize).ok_or_else(|| {
                PbrtError::error("WebGPU area light references an invalid vertex.")
            })?;
            *position = vertex.position;
        }
        area_light.total_area =
            triangle_world_area(instance.transform, positions).ok_or_else(|| {
                PbrtError::error(&format!(
                    "WebGPU area light {area_index} has a degenerate primitive."
                ))
            })?;
    }
    Ok(())
}
