use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

use crate::gpu::ir::flat;
use crate::util::error::PbrtError;

use super::abi::{
    camera_uniform, inverse_transpose_linear, light_table_uniform, material_table_uniform,
    row_major_to_columns, viewport_uniform, AreaLight, DielectricMaterialData, DiffuseMaterialData,
    Geometry, Instance, LayeredBxDFData, LightRecord, LightTableUniform, MaterialRecord,
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
    pub material_table: MaterialTableUniform,
    pub light_table: LightTableUniform,
    pub output: Output,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub geometry_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub material_buffer: wgpu::Buffer,
    pub diffuse_material_buffer: wgpu::Buffer,
    pub scene_data_buffer: wgpu::Buffer,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<Instance>,
    pub materials: Vec<MaterialRecord>,
    pub scattering_models: Vec<ScatteringModelRecord>,
    pub scattering_nodes: Vec<ScatteringNodeRecord>,
    pub layered_bxdf: Vec<LayeredBxDFData>,
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
        let diffuse_materials = material_table.diffuse;
        let dielectric_materials = material_table.dielectric;
        let layered_bxdf = material_table.layered;
        let scattering_models = flat
            .scattering_models
            .iter()
            .map(|model| ScatteringModelRecord {
                surface_root: model.surface_root,
                bssrdf_root: model.bssrdf_root,
                padding: [0; 2],
            })
            .collect::<Vec<_>>();
        let scattering_nodes = flat
            .scattering_nodes
            .iter()
            .map(|node| {
                Ok(ScatteringNodeRecord {
                    kind_tag: scattering_node_tag(&node.kind)?,
                    event_flags: node.event_flags,
                    data_index: node.data_index,
                    child_offset: node.child_offset,
                    child_count: node.child_count,
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
        let material_words = std::mem::size_of::<MaterialRecord>()
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
        let distribution_words = std::mem::size_of::<TriangleDistributionEntry>()
            .checked_div(std::mem::size_of::<u32>())
            .ok_or_else(|| PbrtError::error("WebGPU distribution ABI is not word-aligned."))?;
        let material_words_total = materials
            .len()
            .checked_mul(material_words)
            .ok_or_else(|| PbrtError::error("WebGPU material buffer size overflowed."))?;
        let diffuse_material_words = std::mem::size_of::<DiffuseMaterialData>()
            .checked_div(std::mem::size_of::<u32>())
            .ok_or_else(|| PbrtError::error("WebGPU diffuse-material ABI is not word-aligned."))?;
        let dielectric_material_words = std::mem::size_of::<DielectricMaterialData>()
            .checked_div(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                PbrtError::error("WebGPU dielectric-material ABI is not word-aligned.")
            })?;
        let scattering_model_words = std::mem::size_of::<ScatteringModelRecord>()
            .checked_div(std::mem::size_of::<u32>())
            .ok_or_else(|| PbrtError::error("WebGPU scattering-model ABI is not word-aligned."))?;
        let scattering_node_words = std::mem::size_of::<ScatteringNodeRecord>()
            .checked_div(std::mem::size_of::<u32>())
            .ok_or_else(|| PbrtError::error("WebGPU scattering-node ABI is not word-aligned."))?;
        let diffuse_material_data_offset = material_words_total;
        let diffuse_material_words_total = diffuse_materials
            .len()
            .checked_mul(diffuse_material_words)
            .ok_or_else(|| PbrtError::error("WebGPU diffuse-material buffer size overflowed."))?;
        let dielectric_material_data_offset = diffuse_material_data_offset
            .checked_add(diffuse_material_words_total)
            .ok_or_else(|| PbrtError::error("WebGPU dielectric-material offset overflowed."))?;
        let dielectric_material_words_total = dielectric_materials
            .len()
            .checked_mul(dielectric_material_words)
            .ok_or_else(|| {
                PbrtError::error("WebGPU dielectric-material buffer size overflowed.")
            })?;
        let scattering_model_data_offset = dielectric_material_data_offset
            .checked_add(dielectric_material_words_total)
            .ok_or_else(|| PbrtError::error("WebGPU scattering-model offset overflowed."))?;
        let scattering_model_words_total = scattering_models
            .len()
            .checked_mul(scattering_model_words)
            .ok_or_else(|| PbrtError::error("WebGPU scattering-model buffer size overflowed."))?;
        let scattering_node_data_offset = scattering_model_data_offset
            .checked_add(scattering_model_words_total)
            .ok_or_else(|| PbrtError::error("WebGPU scattering-node offset overflowed."))?;
        let scattering_node_words_total = scattering_nodes
            .len()
            .checked_mul(scattering_node_words)
            .ok_or_else(|| PbrtError::error("WebGPU scattering-node buffer size overflowed."))?;
        let scattering_child_data_offset = scattering_node_data_offset
            .checked_add(scattering_node_words_total)
            .ok_or_else(|| PbrtError::error("WebGPU scattering-child offset overflowed."))?;
        let scattering_child_words_total = flat.scattering_child_refs.node_ids.len();
        let layered_bxdf_data_offset = scattering_child_data_offset
            .checked_add(scattering_child_words_total)
            .ok_or_else(|| PbrtError::error("WebGPU layered-BxDF offset overflowed."))?;
        let layered_bxdf_words = std::mem::size_of::<LayeredBxDFData>()
            .checked_div(std::mem::size_of::<u32>())
            .ok_or_else(|| PbrtError::error("WebGPU layered-BxDF ABI is not word-aligned."))?;
        let layered_bxdf_words_total = layered_bxdf
            .len()
            .checked_mul(layered_bxdf_words)
            .ok_or_else(|| PbrtError::error("WebGPU layered-BxDF buffer size overflowed."))?;
        let distribution_words_total = flat
            .triangle_distributions
            .len()
            .checked_mul(distribution_words)
            .ok_or_else(|| PbrtError::error("WebGPU distribution buffer size overflowed."))?;
        let light_record_data_offset = layered_bxdf_data_offset
            .checked_add(layered_bxdf_words_total)
            .ok_or_else(|| PbrtError::error("WebGPU light-record offset overflowed."))?;
        let point_light_data_offset = light_record_data_offset
            .checked_add(
                light_records
                    .len()
                    .checked_mul(light_record_words)
                    .ok_or_else(|| {
                        PbrtError::error("WebGPU light-record buffer size overflowed.")
                    })?,
            )
            .ok_or_else(|| PbrtError::error("WebGPU point-light offset overflowed."))?;
        let area_light_data_offset =
            point_light_data_offset
                .checked_add(point_lights.len().checked_mul(light_words).ok_or_else(|| {
                    PbrtError::error("WebGPU point-light buffer size overflowed.")
                })?)
                .ok_or_else(|| PbrtError::error("WebGPU area-light buffer offset overflowed."))?;
        let distribution_data_offset = area_light_data_offset
            .checked_add(
                area_lights
                    .len()
                    .checked_mul(area_light_words)
                    .ok_or_else(|| PbrtError::error("WebGPU area-light buffer size overflowed."))?,
            )
            .ok_or_else(|| PbrtError::error("WebGPU distribution buffer offset overflowed."))?;
        for (area_index, area_light) in area_lights.iter_mut().enumerate() {
            let flat_area = flat
                .area_lights
                .get(area_index)
                .ok_or_else(|| PbrtError::error("WebGPU area-light table is inconsistent."))?;
            area_light.distribution_offset_words = to_u32_offset(
                distribution_data_offset
                    .checked_add(
                        usize::try_from(flat_area.distribution.offset)
                            .map_err(|_| {
                                PbrtError::error(
                                    "WebGPU distribution offset does not fit in usize.",
                                )
                            })?
                            .checked_mul(distribution_words)
                            .ok_or_else(|| {
                                PbrtError::error("WebGPU distribution offset overflowed.")
                            })?,
                    )
                    .ok_or_else(|| PbrtError::error("WebGPU distribution offset overflowed."))?,
                "distribution offset",
            )?;
        }
        let camera = camera_uniform(&flat.camera, &flat.viewport)?;
        let viewport = viewport_uniform(&flat.viewport, &flat.render_settings)?;
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
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 material record SBO"),
            contents: cast_slice(&materials),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let diffuse_material_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 diffuse material data SBO"),
                contents: cast_slice(&diffuse_materials),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let light_record_words_total = light_records
            .len()
            .checked_mul(light_record_words)
            .ok_or_else(|| PbrtError::error("WebGPU light-record buffer size overflowed."))?;
        let point_light_words_total = point_lights
            .len()
            .checked_mul(light_words)
            .ok_or_else(|| PbrtError::error("WebGPU point-light buffer size overflowed."))?;
        let area_light_words_total = area_lights
            .len()
            .checked_mul(area_light_words)
            .ok_or_else(|| PbrtError::error("WebGPU area-light buffer size overflowed."))?;
        let scene_data_capacity = material_words_total
            .checked_add(diffuse_material_words_total)
            .and_then(|size| size.checked_add(dielectric_material_words_total))
            .and_then(|size| size.checked_add(scattering_model_words_total))
            .and_then(|size| size.checked_add(scattering_node_words_total))
            .and_then(|size| size.checked_add(scattering_child_words_total))
            .and_then(|size| size.checked_add(layered_bxdf_words_total))
            .and_then(|size| size.checked_add(light_record_words_total))
            .and_then(|size| size.checked_add(point_light_words_total))
            .and_then(|size| size.checked_add(area_light_words_total))
            .and_then(|size| size.checked_add(distribution_words_total))
            .ok_or_else(|| PbrtError::error("WebGPU material/light buffer size overflowed."))?;
        let mut scene_data = Vec::<u32>::with_capacity(scene_data_capacity);
        for material in &materials {
            scene_data.extend_from_slice(cast_slice(std::slice::from_ref(material)));
        }
        for material in &diffuse_materials {
            scene_data.extend_from_slice(cast_slice(std::slice::from_ref(material)));
        }
        for material in &dielectric_materials {
            scene_data.extend_from_slice(cast_slice(std::slice::from_ref(material)));
        }
        for model in &scattering_models {
            scene_data.extend_from_slice(cast_slice(std::slice::from_ref(model)));
        }
        for node in &scattering_nodes {
            scene_data.extend_from_slice(cast_slice(std::slice::from_ref(node)));
        }
        scene_data.extend_from_slice(&flat.scattering_child_refs.node_ids);
        for data in &layered_bxdf {
            scene_data.extend_from_slice(cast_slice(std::slice::from_ref(data)));
        }
        for record in &light_records {
            scene_data.extend_from_slice(cast_slice(std::slice::from_ref(record)));
        }
        for light in &point_lights {
            scene_data.extend_from_slice(cast_slice(std::slice::from_ref(light)));
        }
        for light in &area_lights {
            scene_data.extend_from_slice(cast_slice(std::slice::from_ref(light)));
        }
        for entry in &flat.triangle_distributions {
            scene_data.extend_from_slice(cast_slice(std::slice::from_ref(
                &TriangleDistributionEntry {
                    primitive: entry.primitive,
                    cdf: entry.cdf,
                    area: entry.area,
                    reserved: 0,
                },
            )));
        }
        let packed_light_bvh = pack_light_bvh(&flat.light_bvh)?;
        let (light_sampler_data_offset, light_bvh_node_offset, light_leaf_offset) =
            if let Some(packed) = &packed_light_bvh {
                let header_offset = align_words(scene_data.len(), 8)?;
                scene_data.resize(header_offset, 0);
                scene_data.extend_from_slice(&packed.header_words);
                let node_offset = scene_data.len();
                for node in &packed.node_words {
                    scene_data.extend_from_slice(node);
                }
                let leaf_offset = scene_data.len();
                scene_data.extend_from_slice(&packed.handle_to_leaf);
                (header_offset, node_offset, leaf_offset)
            } else {
                (
                    INVALID_INDEX as usize,
                    INVALID_INDEX as usize,
                    INVALID_INDEX as usize,
                )
            };
        let limits = device.limits();
        validate_scene_data_size(
            scene_data.len(),
            limits.max_buffer_size,
            u64::from(limits.max_storage_buffer_binding_size),
        )?;
        let mut material_table = material_table_uniform(
            materials.len(),
            diffuse_material_data_offset,
            diffuse_materials.len(),
            dielectric_material_data_offset,
            dielectric_materials.len(),
            scattering_model_data_offset,
            scattering_models.len(),
            scattering_node_data_offset,
            scattering_nodes.len(),
            scattering_child_data_offset,
            scattering_child_words_total,
            INVALID_INDEX as usize,
            0,
            layered_bxdf_data_offset,
            layered_bxdf.len(),
        )?;
        let mut light_table = light_table_uniform(
            light_records.len(),
            point_lights.len(),
            area_lights.len(),
            light_record_data_offset,
            point_light_data_offset,
            area_light_data_offset,
        )?;
        material_table.debug_scattering_model = INVALID_INDEX;
        if let Some(packed) = &packed_light_bvh {
            if light_sampler_kind == LightSamplerKind::Bvh {
                light_table.light_sampler_kind = super::abi::LIGHT_SAMPLER_KIND_BVH;
            }
            light_table.light_sampler_data_offset =
                to_u32_offset(light_sampler_data_offset, "light sampler data offset")?;
            light_table.light_bvh_node_offset =
                to_u32_offset(light_bvh_node_offset, "light BVH node offset")?;
            light_table.light_bvh_node_count =
                u32::try_from(packed.node_words.len()).map_err(|_| {
                    PbrtError::error("WebGPU Light BVH node count does not fit in u32.")
                })?;
            light_table.light_leaf_offset =
                to_u32_offset(light_leaf_offset, "light handle-to-leaf offset")?;
            light_table.light_leaf_count =
                u32::try_from(packed.handle_to_leaf.len()).map_err(|_| {
                    PbrtError::error("WebGPU Light BVH leaf count does not fit in u32.")
                })?;
        }
        let scene_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 scene data SBO"),
            contents: cast_slice(&scene_data),
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
            material_table,
            light_table,
            output: Output::from_flat(flat.output),
            vertex_buffer,
            index_buffer,
            geometry_buffer,
            instance_buffer,
            material_buffer,
            diffuse_material_buffer,
            scene_data_buffer,
            geometries,
            instances,
            materials,
            scattering_models,
            scattering_nodes,
            layered_bxdf,
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
        queue.write_buffer(
            &self.scene_data_buffer,
            0,
            bytemuck::cast_slice(&self.materials),
        );
    }
}

fn align_words(value: usize, alignment: usize) -> Result<usize, PbrtError> {
    let remainder = value % alignment;
    value
        .checked_add((alignment - remainder) % alignment)
        .ok_or_else(|| PbrtError::error("WebGPU scene-data alignment overflowed."))
}

fn to_u32_offset(value: usize, label: &str) -> Result<u32, PbrtError> {
    u32::try_from(value)
        .map_err(|_| PbrtError::error(&format!("WebGPU {label} does not fit in u32.")))
}

pub fn validate_scene_data_size(
    word_count: usize,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
) -> Result<u64, PbrtError> {
    let byte_count = u64::try_from(word_count)
        .ok()
        .and_then(|count| count.checked_mul(std::mem::size_of::<u32>() as u64))
        .ok_or_else(|| PbrtError::error("WebGPU scene-data byte size overflowed."))?;
    if byte_count > max_buffer_size {
        return Err(PbrtError::error(&format!(
            "WebGPU scene data requires {byte_count} bytes, exceeding max_buffer_size {max_buffer_size}."
        )));
    }
    if byte_count > max_storage_buffer_binding_size {
        return Err(PbrtError::error(&format!(
            "WebGPU scene data requires {byte_count} bytes, exceeding max_storage_buffer_binding_size {max_storage_buffer_binding_size}."
        )));
    }
    Ok(byte_count)
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
