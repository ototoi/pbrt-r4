use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use crate::gpu::ir::flat;
use crate::util::error::PbrtError;

use super::abi::{
    camera_uniform, film_uniform, inverse_transpose_linear, light_table_uniform,
    material_table_uniform, row_major_to_columns, viewport_uniform, AreaLight, FilmUniform,
    Geometry, Instance, LightRecord, LightTableUniform, MaterialAttributeRef, MaterialRecord,
    MaterialTableUniform, PointLight, ScatteringModelRecord, ScatteringNodeRecord,
    TriangleDistributionEntry, Vertex, ViewportUniform, INVALID_INDEX, LIGHT_KIND_AREA,
    LIGHT_KIND_POINT,
};
use super::acceleration::{self, Acceleration};
use super::light_bvh::pack_light_bvh;
use super::light_sampler::{resolve_scene_light_sampler_count, LightSamplerKind};
use super::material::{scattering_node_tag, MaterialKind, MaterialTable};
use super::output::Output;

pub struct Scene {
    pub camera: super::abi::CameraUniform,
    pub viewport: ViewportUniform,
    pub film: FilmUniform,
    pub material_table: MaterialTableUniform,
    pub light_table: LightTableUniform,
    pub output: Output,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub geometry_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub material_buffer: wgpu::Buffer,
    pub material_attribute_buffer: wgpu::Buffer,
    pub scalar_attribute_buffer: wgpu::Buffer,
    pub scattering_model_buffer: wgpu::Buffer,
    pub scattering_node_buffer: wgpu::Buffer,
    pub scattering_child_buffer: wgpu::Buffer,
    pub spectrum_attribute_buffer: wgpu::Buffer,
    pub spectrum_sample_buffer: wgpu::Buffer,
    pub spectrum_metadata_buffer: wgpu::Buffer,
    pub texture_attribute_buffer: wgpu::Buffer,
    pub light_record_buffer: wgpu::Buffer,
    pub point_light_buffer: wgpu::Buffer,
    pub area_light_buffer: wgpu::Buffer,
    pub distribution_buffer: wgpu::Buffer,
    pub light_bvh_header_buffer: wgpu::Buffer,
    pub light_bvh_node_buffer: wgpu::Buffer,
    pub light_leaf_buffer: wgpu::Buffer,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<Instance>,
    pub materials: Vec<MaterialRecord>,
    pub scattering_models: Vec<ScatteringModelRecord>,
    pub scattering_nodes: Vec<ScatteringNodeRecord>,
    pub material_attributes: Vec<MaterialAttributeRef>,
    pub point_lights: Vec<PointLight>,
    pub area_lights: Vec<AreaLight>,
    pub light_records: Vec<LightRecord>,
    pub light_sampler_kind: LightSamplerKind,
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
                    area_light: instance.area_light,
                    orientation_flags: u32::from(instance.reverse_orientation)
                        | (u32::from(flat::transform_swaps_handedness(instance.transform)) << 1),
                    world_from_object: row_major_to_columns(instance.transform),
                    normal_from_object: inverse_transpose_linear(instance.transform, &label)?,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let material_table = MaterialTable::from_flat(&flat)?;
        let materials = material_table.records;
        let material_attributes = material_table.attributes;
        let scalar_attributes = flat.attribute_tables.scalars.clone();
        let spectrum_attributes = flat
            .attribute_tables
            .spectra
            .iter()
            .map(|v| v.0)
            .collect::<Vec<_>>();
        let texture_attributes = flat.attribute_tables.textures.clone();
        let scattering_models = flat
            .scattering_models
            .iter()
            .map(|model| ScatteringModelRecord {
                surface_root: model.surface_root,
                bssrdf_root: model.bssrdf_root,
                padding: [0; 2],
            })
            .collect::<Vec<_>>();
        let mut node_attribute_ranges = vec![(0u32, 0u32); flat.scattering_nodes.len()];
        for (material_index, material) in materials.iter().enumerate() {
            let Some(flat_material) = flat.materials.get(material_index) else {
                continue;
            };
            let mut pending;
            let Some(model) = flat
                .scattering_models
                .get(flat_material.scattering_model as usize)
            else {
                continue;
            };
            pending = vec![model.surface_root];
            while let Some(node_id) = pending.pop() {
                let Some(node) = flat.scattering_nodes.get(node_id as usize) else {
                    continue;
                };
                node_attribute_ranges[node_id as usize] =
                    (material.attribute_offset, material.attribute_count);
                let end = node.child_offset.saturating_add(node.child_count);
                if let Some(children) = flat
                    .scattering_child_refs
                    .node_ids
                    .get(node.child_offset as usize..end as usize)
                {
                    pending.extend(children.iter().copied());
                }
            }
        }
        let scattering_nodes = flat
            .scattering_nodes
            .iter()
            .enumerate()
            .map(|(node_id, node)| {
                Ok(ScatteringNodeRecord {
                    kind_tag: scattering_node_tag(&node.kind)?,
                    event_flags: node.event_flags,
                    attribute_offset: node_attribute_ranges[node_id].0,
                    child_offset: node.child_offset,
                    child_count: node.child_count,
                    attribute_count: node_attribute_ranges[node_id].1,
                    padding: [0; 2],
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
                distribution_offset_words: 0,
                distribution_count: light.distribution.count,
                total_area: light.distribution.total_area,
                emission: light.emission,
                flags: u32::from(light.two_sided),
            })
            .collect::<Vec<_>>();
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
        let light_sampler_kind = resolve_scene_light_sampler_count(
            &flat.render_settings,
            flat.light_bvh.bounded_handles.len(),
        )?;
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
        let scattering_child_words_total = flat.scattering_child_refs.node_ids.len();
        for (area_index, area_light) in area_lights.iter_mut().enumerate() {
            let flat_area = flat
                .area_lights
                .get(area_index)
                .ok_or_else(|| PbrtError::error("WebGPU area-light table is inconsistent."))?;
            area_light.distribution_offset_words = flat_area.distribution.offset;
        }
        let camera = camera_uniform(&flat.camera, &flat.viewport)?;
        let viewport = viewport_uniform(&flat.viewport, &flat.render_settings)?;
        let film = film_uniform(&flat.film);
        if vertices.is_empty() || indices.is_empty() || instances.is_empty() || materials.is_empty()
        {
            return Err(PbrtError::error(
                "WebGPU primary-ray rendering requires non-empty geometry, instances, and materials.",
            ));
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 vertex SBO"),
            contents: buffer_contents(&vertices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 local index SBO"),
            contents: buffer_contents(&indices),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT,
        });
        let geometry_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 geometry SBO"),
            contents: buffer_contents(&geometries),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 instance SBO"),
            contents: buffer_contents(&instances),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 material record SBO"),
            contents: buffer_contents(&materials),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let material_attribute_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 material attribute refs SBO"),
                contents: buffer_contents(&material_attributes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let scalar_attribute_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 scalar attributes SBO"),
                contents: buffer_contents(&scalar_attributes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let scattering_model_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 scattering model SBO"),
                contents: buffer_contents(&scattering_models),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let scattering_node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 scattering node SBO"),
            contents: buffer_contents(&scattering_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let scattering_child_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 scattering child SBO"),
                contents: buffer_contents(&flat.scattering_child_refs.node_ids),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let spectrum_attribute_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 spectrum attributes SBO"),
                contents: buffer_contents(&spectrum_attributes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let texture_attribute_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 texture attributes SBO"),
                contents: buffer_contents(&texture_attributes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        flat.spectrum_table.validate()?;
        let spectrum_metadata = flat
            .spectrum_table
            .metadata
            .iter()
            .map(|metadata| metadata.flags)
            .collect::<Vec<_>>();
        let spectrum_sample_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 dense spectrum samples SBO"),
            contents: buffer_contents(&flat.spectrum_table.samples),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let spectrum_metadata_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 spectrum metadata SBO"),
                contents: buffer_contents(&spectrum_metadata),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let distribution_entries = flat
            .triangle_distributions
            .iter()
            .map(|entry| TriangleDistributionEntry {
                primitive: entry.primitive,
                cdf: entry.cdf,
                area: entry.area,
                reserved: 0,
            })
            .collect::<Vec<_>>();
        let light_record_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 light record SBO"),
            contents: buffer_contents(&light_records),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let point_light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 point light SBO"),
            contents: buffer_contents(&point_lights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let area_light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 area light SBO"),
            contents: buffer_contents(&area_lights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let distribution_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 triangle distribution SBO"),
            contents: buffer_contents(&distribution_entries),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let packed_light_bvh = pack_light_bvh(&flat.light_bvh)?;
        let mut material_table = material_table_uniform(
            materials.len(),
            0,
            scattering_models.len(),
            0,
            scattering_nodes.len(),
            0,
            scattering_child_words_total,
            INVALID_INDEX as usize,
            0,
        )?;
        let mut light_table = light_table_uniform(
            light_records.len(),
            point_lights.len(),
            area_lights.len(),
            0,
            0,
            0,
        )?;
        material_table.debug_scattering_model = INVALID_INDEX;
        if let Some(packed) = &packed_light_bvh {
            if light_sampler_kind == LightSamplerKind::Bvh {
                light_table.light_sampler_kind = super::abi::LIGHT_SAMPLER_KIND_BVH;
            }
            light_table.light_sampler_data_offset = 0;
            light_table.light_bvh_node_offset = 0;
            light_table.light_bvh_node_count =
                u32::try_from(packed.node_words.len()).map_err(|_| {
                    PbrtError::error("WebGPU Light BVH node count does not fit in u32.")
                })?;
            light_table.light_leaf_offset = 0;
            light_table.light_leaf_count =
                u32::try_from(packed.handle_to_leaf.len()).map_err(|_| {
                    PbrtError::error("WebGPU Light BVH leaf count does not fit in u32.")
                })?;
        }
        let (light_bvh_header, light_bvh_nodes, light_leaf) = packed_light_bvh
            .as_ref()
            .map(|packed| {
                (
                    packed.header_words.to_vec(),
                    packed
                        .node_words
                        .iter()
                        .flat_map(|node| node.iter().copied())
                        .collect::<Vec<_>>(),
                    packed.handle_to_leaf.clone(),
                )
            })
            .unwrap_or_default();
        let light_bvh_header_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 light BVH header SBO"),
                contents: buffer_contents(&light_bvh_header),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let light_bvh_node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 light BVH node SBO"),
            contents: buffer_contents(&light_bvh_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let light_leaf_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 light BVH leaf SBO"),
            contents: buffer_contents(&light_leaf),
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
            film,
            material_table,
            light_table,
            output: Output::from_flat(flat.output),
            vertex_buffer,
            index_buffer,
            geometry_buffer,
            instance_buffer,
            material_buffer,
            material_attribute_buffer,
            scalar_attribute_buffer,
            scattering_model_buffer,
            scattering_node_buffer,
            scattering_child_buffer,
            spectrum_attribute_buffer,
            spectrum_sample_buffer,
            spectrum_metadata_buffer,
            texture_attribute_buffer,
            light_record_buffer,
            point_light_buffer,
            area_light_buffer,
            distribution_buffer,
            light_bvh_header_buffer,
            light_bvh_node_buffer,
            light_leaf_buffer,
            geometries,
            instances,
            materials,
            scattering_models,
            scattering_nodes,
            material_attributes,
            point_lights,
            area_lights,
            light_records,
            light_sampler_kind,
            render_settings: flat.render_settings,
            acceleration,
        })
    }

    pub fn replace_material_kind(&mut self, queue: &wgpu::Queue, kind: MaterialKind) {
        self.material_table.debug_scattering_model = kind.tag();
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

fn buffer_contents<T: bytemuck::Pod>(values: &[T]) -> &[u8] {
    if values.is_empty() {
        // WebGPU validates the minimum binding size against the declared
        // storage-array stride, so a four-byte sentinel is insufficient for
        // an empty array of a larger record type.
        cast_slice(&[0u32; 16])
    } else {
        cast_slice(values)
    }
}

fn validate_instance_area_lights(
    instance_index: usize,
    instance: &flat::Instance,
    flat: &flat::Scene,
) -> Result<(), PbrtError> {
    if instance.area_light == flat::INVALID_INDEX {
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
    let handle = instance.area_light;
    let record = flat.lights.get(handle as usize).ok_or_else(|| {
        PbrtError::error(&format!(
            "Flat instance {instance_index} has an invalid area-light handle."
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
    if area_light.instance as usize != instance_index {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light range does not match its triangles."
        )));
    }
    let offset = usize::try_from(area_light.distribution.offset)
        .map_err(|_| PbrtError::error("Flat area-light distribution offset does not fit usize."))?;
    let count = usize::try_from(area_light.distribution.count)
        .map_err(|_| PbrtError::error("Flat area-light distribution count does not fit usize."))?;
    if count == 0 {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light distribution is empty."
        )));
    }
    let end = offset
        .checked_add(count)
        .ok_or_else(|| PbrtError::error("Flat area-light distribution range overflowed."))?;
    let entries = flat
        .triangle_distributions
        .get(offset..end)
        .ok_or_else(|| {
            PbrtError::error(&format!(
                "Flat instance {instance_index} area-light distribution range is invalid."
            ))
        })?;
    if !area_light.distribution.total_area.is_finite() || area_light.distribution.total_area <= 0.0
    {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light total area is invalid."
        )));
    }
    let mut previous_cdf = 0.0;
    let mut area_sum = 0.0;
    for entry in entries {
        if entry.primitive >= triangle_count
            || !entry.area.is_finite()
            || entry.area <= 0.0
            || !entry.cdf.is_finite()
            || entry.cdf < previous_cdf
            || entry.cdf > 1.0
        {
            return Err(PbrtError::error(&format!(
                "Flat instance {instance_index} has an invalid area-light distribution entry."
            )));
        }
        previous_cdf = entry.cdf;
        area_sum += entry.area;
    }
    if previous_cdf != 1.0
        || (area_sum - area_light.distribution.total_area).abs()
            > area_light.distribution.total_area.abs() * 1e-5
    {
        return Err(PbrtError::error(&format!(
            "Flat instance {instance_index} area-light distribution does not match total area."
        )));
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
