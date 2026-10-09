use bytemuck::cast_slice;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::gpu::flat;
use crate::gpu::flat::texture::{ImageFilterMode, ImageWrapMode};
use crate::util::error::PbrtError;

use super::abi::{
    camera_uniform, film_uniform, light_table_uniform, material_table_uniform, viewport_uniform,
    CameraUniform, DenseSpectrum, FilmUniform, Geometry, Instance, LightRecord, LightSamplingModel,
    LightTableUniform, MaterialNode, MaterialTableUniform, ViewportUniform, INVALID_INDEX,
    LIGHT_SAMPLER_KIND_BVH,
};
use super::acceleration::{self, Acceleration};
use super::bssrdf::{convert_bssrdf_materials, BSSRDFTableData, BSSRDFTableResources};
use super::light_bvh::pack_light_bvh;
use super::light_sampler::{resolve_scene_light_sampler_count, LightSamplerKind};
use super::material::MaterialKind;
use super::material::MaterialTable;
use super::measured::MeasuredBsdfRecords;
use super::output::Output;
use super::render_settings::RenderSettings;
use super::sampler::SamplerResources;
use super::stages::ResourceId;

mod geometry;
pub mod instance;
pub mod light;
pub mod medium;
mod texture;
mod upload;

pub use texture::{lower_texture_library_records, texture_binding_counts};

use geometry::convert_geometry;
use instance::convert_instances;
use light::{convert_lights, LightSamplingData};
use medium::convert_media;
use texture::{
    infinite_image_payload, lower_texture_library, scene_texture_views, texture_binding_plan,
    validate_image_infinite_buffer_size,
};
use upload::{upload_measured_atlas, upload_texture_images};

pub struct Scene {
    pub camera: CameraUniform,
    pub viewport: ViewportUniform,
    pub film: FilmUniform,
    pub film_output_matrix: [[f32; 3]; 3],
    pub film_scale: f32,
    pub material_table: MaterialTableUniform,
    pub light_table: LightTableUniform,
    pub sampler: SamplerResources,
    pub output: Output,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub geometry_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub medium_buffer: wgpu::Buffer,
    pub bssrdf_material_buffer: wgpu::Buffer,
    pub bssrdf_tables: BSSRDFTableResources,
    pub material_root_buffer: wgpu::Buffer,
    pub material_node_buffer: wgpu::Buffer,
    pub attribute_ref_buffer: wgpu::Buffer,
    pub scalar_attribute_buffer: wgpu::Buffer,
    pub spectrum_attribute_buffer: wgpu::Buffer,
    pub measured_bsdf_buffer: wgpu::Buffer,
    pub measured_table_buffer: wgpu::Buffer,
    pub texture_root_buffer: wgpu::Buffer,
    pub texture_node_buffer: wgpu::Buffer,
    pub texture_child_buffer: wgpu::Buffer,
    pub rgb_spectrum_table_buffer: wgpu::Buffer,
    pub texture_images: Vec<wgpu::Texture>,
    pub texture_image_view: wgpu::TextureView,
    pub texture_image_views: Vec<wgpu::TextureView>,
    pub texture_samplers: Vec<wgpu::Sampler>,
    pub light_record_buffer: wgpu::Buffer,
    pub light_sampling_model_buffer: wgpu::Buffer,
    pub light_position_buffer: wgpu::Buffer,
    pub distribution_buffer: wgpu::Buffer,
    pub portal_image_buffer: wgpu::Buffer,
    pub portal_distribution_buffer: wgpu::Buffer,
    pub image_infinite_sampling_buffer: wgpu::Buffer,
    pub image_infinite_distribution_buffer: wgpu::Buffer,
    pub image_infinite_row_cdf_buffer: wgpu::Buffer,
    pub light_bvh_header_buffer: wgpu::Buffer,
    pub light_bvh_node_buffer: wgpu::Buffer,
    pub light_leaf_buffer: wgpu::Buffer,
    pub geometries: Vec<Geometry>,
    pub instances: Vec<Instance>,
    pub material_nodes: Vec<MaterialNode>,
    pub light_sampling_models: Vec<LightSamplingModel>,
    pub light_records: Vec<LightRecord>,
    pub light_sampler_kind: LightSamplerKind,
    pub render_settings: RenderSettings,
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
        if flat.camera.medium != INVALID_INDEX && flat.camera.medium as usize >= flat.media.len() {
            return Err(PbrtError::error(
                "Flat camera references an invalid medium.",
            ));
        }
        let instances = convert_instances(&flat, geometries.len())?;
        let media = convert_media(&flat.media, flat.spectrum_attributes.len())?;
        let material_table = MaterialTable::from_flat(&flat)?;
        let material_nodes = material_table.nodes;
        let measured_records = MeasuredBsdfRecords::from_flat(&flat.measured_bsdfs);
        let measured_bsdfs = measured_records.bsdfs;
        let measured_tables = measured_records.tables;
        flat.texture_library.validate()?;
        let scalar_attributes = flat.scalar_attributes.clone();
        // Keep the infinite-image sampler first. pbrt-v4 uses nearest lookup
        // for ImageInfiniteLight after equal-area sphere-to-square mapping.
        let (texture_views, material_view_offset) = scene_texture_views(&flat);
        let texture_binding_plan = texture_binding_plan(&texture_views)?;
        let (texture_nodes, texture_children, texture_roots) = lower_texture_library(
            &flat.texture_library,
            &texture_binding_plan,
            material_view_offset,
        )?;
        let infinite_image_bindings = flat
            .infinite_lights
            .iter()
            .map(|light| {
                if light.image_index == flat::INVALID_INDEX {
                    return Ok(None);
                }
                let view_index = texture_views
                    .iter()
                    .position(|view| view.mipmap == light.image_index)
                    .ok_or_else(|| {
                        PbrtError::error("Infinite light image view was not registered.")
                    })?;
                let binding = texture_binding_plan
                    .image_views
                    .iter()
                    .position(|&index| index == view_index)
                    .ok_or_else(|| {
                        PbrtError::error("Infinite light image binding was not generated.")
                    })?;
                let mipmap = flat
                    .texture_library
                    .mipmaps
                    .get(light.image_index as usize)
                    .ok_or_else(|| {
                        PbrtError::error("Infinite light references an invalid mipmap.")
                    })?;
                let payload = infinite_image_payload(binding, mipmap.color_space)?;
                Ok(Some((light.sampling_model, payload)))
            })
            .collect::<Result<Vec<_>, PbrtError>>()?
            .into_iter()
            .flatten()
            .collect::<HashMap<_, _>>();
        let light_data = convert_lights(
            &flat.light_sampling_models,
            &flat.lights,
            &flat.infinite_lights,
            material_table.attributes,
            &infinite_image_bindings,
        )?;
        let attribute_refs = light_data.attributes;
        let light_records = light_data.records;
        let light_sampling_models = light_data.sampling_models;
        let camera = camera_uniform(&flat.camera, &flat.viewport)?;
        let viewport = viewport_uniform(&flat.viewport, &flat.render_settings)?;
        let sampler = SamplerResources::new_with_subsurface(
            device,
            queue,
            &flat.render_settings,
            flat.viewport.resolution,
            !flat.bssrdfs.is_empty(),
        )?;
        let film = film_uniform(&flat.film);
        let film_output_matrix = flat.film.output_rgb_from_sensor_rgb;
        let film_scale = flat.film.scale;
        if vertices.is_empty()
            || indices.is_empty()
            || instances.is_empty()
            || material_nodes.is_empty()
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
        let medium_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 medium records SBO"),
            contents: buffer_contents(&media),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let material_root_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 material tree layouts SBO"),
            contents: buffer_contents(&material_table.roots),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let material_node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 material tree nodes SBO"),
            contents: buffer_contents(&material_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let attribute_ref_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 material attribute refs SBO"),
            contents: buffer_contents(&attribute_refs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let scalar_attribute_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 scalar attributes SBO"),
                contents: buffer_contents(&scalar_attributes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let texture_root_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 texture roots SBO"),
            contents: buffer_contents(&texture_roots),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let texture_node_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 texture nodes SBO"),
            contents: buffer_contents(&texture_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let texture_child_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 texture child indices SBO"),
            contents: buffer_contents(&texture_children),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut rgb_spectrum_table = Vec::new();
        for table in [
            include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_srgb.bin")),
            include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_aces.bin")),
            include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_dci_p3.bin")),
            include_bytes!(concat!(env!("OUT_DIR"), "/rgb_to_spectrum_rec2020.bin")),
        ] {
            rgb_spectrum_table.extend_from_slice(table);
        }
        let rgb_spectrum_table_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 RGB spectrum tables SBO"),
                contents: &rgb_spectrum_table,
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mut texture_images = upload_texture_images(
            device,
            queue,
            &flat.texture_library.mipmaps,
            &texture_views,
            &texture_binding_plan.image_views,
        )?;
        let measured_texture_binding_base = u32::try_from(texture_images.len())
            .map_err(|_| PbrtError::error("Texture image binding count exceeds u32."))?;
        texture_images.extend(upload_measured_atlas(
            device,
            queue,
            &flat.measured_bsdfs.atlas_pages,
        )?);
        if texture_images.is_empty() {
            texture_images.push(device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pbrt-r4 empty texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            }));
        }
        let texture_image_view =
            texture_images[0].create_view(&wgpu::TextureViewDescriptor::default());
        let texture_image_views = texture_images
            .iter()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();
        let texture_samplers = if texture_binding_plan.samplers.is_empty() {
            vec![device.create_sampler(&wgpu::SamplerDescriptor::default())]
        } else {
            texture_binding_plan
                .samplers
                .iter()
                .map(|sampler| {
                    let address_mode = |mode| match mode {
                        ImageWrapMode::Clamp | ImageWrapMode::Black => {
                            wgpu::AddressMode::ClampToEdge
                        }
                        ImageWrapMode::Repeat => wgpu::AddressMode::Repeat,
                    };
                    let filter = if sampler.filter == ImageFilterMode::Nearest {
                        wgpu::FilterMode::Nearest
                    } else {
                        wgpu::FilterMode::Linear
                    };
                    device.create_sampler(&wgpu::SamplerDescriptor {
                        label: Some("pbrt-r4 texture sampler"),
                        address_mode_u: address_mode(sampler.swrap),
                        address_mode_v: address_mode(sampler.twrap),
                        mag_filter: filter,
                        min_filter: filter,
                        mipmap_filter: if sampler.filter == ImageFilterMode::Trilinear {
                            wgpu::MipmapFilterMode::Linear
                        } else {
                            wgpu::MipmapFilterMode::Nearest
                        },
                        ..Default::default()
                    })
                })
                .collect()
        };
        flat::validate_dense_spectra(&flat.spectrum_attributes)?;
        let spectrum_attributes = flat
            .spectrum_attributes
            .iter()
            .map(|spectrum| DenseSpectrum {
                samples: spectrum.samples,
                flags: spectrum.flags,
            })
            .collect::<Vec<_>>();
        let spectrum_attribute_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 dense spectra SBO"),
                contents: buffer_contents(&spectrum_attributes),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let measured_bsdf_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 measured BSDF records SBO"),
            contents: buffer_contents(&measured_bsdfs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let measured_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 measured BSDF tables SBO"),
            contents: buffer_contents(&measured_tables),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let light_sampling_data = LightSamplingData::from_flat(&flat);
        let light_record_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 light record SBO"),
            contents: buffer_contents(&light_records),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let light_sampling_model_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 light sampling model SBO"),
                contents: buffer_contents(&light_sampling_models),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let light_position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 light position SBO"),
            contents: buffer_contents(&light_sampling_data.positions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let distribution_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 triangle distribution SBO"),
            contents: buffer_contents(&light_sampling_data.triangle_distributions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let portal_image_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pbrt-r4 portal image infinite records SBO"),
            contents: buffer_contents(&light_sampling_data.portal_records),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let portal_distribution_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 portal distribution SBO"),
                contents: buffer_contents(&light_sampling_data.portal_distribution),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let limits = device.limits();
        validate_image_infinite_buffer_size(
            "sampling records",
            buffer_contents(&light_sampling_data.image_infinite_records).len(),
            &limits,
        )?;
        let image_infinite_sampling_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 image infinite sampling records SBO"),
                contents: buffer_contents(&light_sampling_data.image_infinite_records),
                usage: wgpu::BufferUsages::STORAGE,
            });
        validate_image_infinite_buffer_size(
            "distribution",
            buffer_contents(&light_sampling_data.image_infinite_distribution).len(),
            &limits,
        )?;
        let image_infinite_distribution_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 image infinite distribution SBO"),
                contents: buffer_contents(&light_sampling_data.image_infinite_distribution),
                usage: wgpu::BufferUsages::STORAGE,
            });
        validate_image_infinite_buffer_size(
            "row CDF",
            buffer_contents(&flat.image_infinite_row_cdf).len(),
            &limits,
        )?;
        let image_infinite_row_cdf_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pbrt-r4 image infinite row CDF SBO"),
                contents: buffer_contents(&flat.image_infinite_row_cdf),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let packed_light_bvh = pack_light_bvh(&flat.light_bvh)?;
        let light_sampler_kind =
            resolve_scene_light_sampler_count(&flat.render_settings, light_records.len())?;
        let table_data = BSSRDFTableData::from_flat(&flat.bssrdf_tables)?;
        let bssrdf_tables = table_data.upload(device)?;
        let bssrdf_materials = convert_bssrdf_materials(&flat.bssrdfs, flat.bssrdf_tables.len())?;
        if !bssrdf_materials.is_empty() && !flat.media.is_empty() {
            return Err(PbrtError::error(
                "WebGPU subsurface with participating media is not supported yet.",
            ));
        }
        let bssrdf_material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BSSRDF materials"),
            contents: buffer_contents(&bssrdf_materials),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut material_table = material_table_uniform(material_nodes.len())?;
        material_table.have_subsurface = u32::from(!bssrdf_materials.is_empty());
        material_table.measured_texture_base = measured_texture_binding_base;
        material_table.measured_texture_width = flat::MEASURED_ATLAS_WIDTH;
        material_table.measured_texture_height = flat::MEASURED_ATLAS_HEIGHT;
        material_table.measured_texture_count =
            u32::try_from(flat.measured_bsdfs.atlas_pages.len())
                .map_err(|_| PbrtError::error("Measured BSDF atlas page count exceeds u32."))?;
        let mut light_table =
            light_table_uniform(flat.lights.len(), flat.infinite_lights.len(), 0)?;
        material_table.debug_material_kind = INVALID_INDEX;
        if let Some(packed) = &packed_light_bvh {
            if light_sampler_kind == LightSamplerKind::Bvh {
                light_table.light_sampler_kind = LIGHT_SAMPLER_KIND_BVH;
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
            &flat.material_roots,
            &flat.material_nodes,
            &flat.lights,
            &flat.light_sampling_models,
        )?;
        Ok(Self {
            camera,
            viewport,
            film,
            film_output_matrix,
            film_scale,
            material_table,
            light_table,
            sampler,
            output: Output::from_flat(flat.output),
            vertex_buffer,
            index_buffer,
            geometry_buffer,
            instance_buffer,
            medium_buffer,
            bssrdf_material_buffer,
            bssrdf_tables,
            material_root_buffer,
            material_node_buffer,
            attribute_ref_buffer,
            scalar_attribute_buffer,
            spectrum_attribute_buffer,
            measured_bsdf_buffer,
            measured_table_buffer,
            texture_root_buffer,
            texture_node_buffer,
            texture_child_buffer,
            rgb_spectrum_table_buffer,
            texture_images,
            texture_image_view,
            texture_image_views,
            texture_samplers,
            light_record_buffer,
            light_sampling_model_buffer,
            light_position_buffer,
            distribution_buffer,
            portal_image_buffer,
            portal_distribution_buffer,
            image_infinite_sampling_buffer,
            image_infinite_distribution_buffer,
            image_infinite_row_cdf_buffer,
            light_bvh_header_buffer,
            light_bvh_node_buffer,
            light_leaf_buffer,
            geometries,
            instances,
            material_nodes,
            light_sampling_models,
            light_records,
            light_sampler_kind,
            render_settings: RenderSettings::from_flat(flat.render_settings),
            acceleration,
        })
    }

    pub fn binding_resource(&self, resource: ResourceId) -> Option<wgpu::BindingResource<'_>> {
        match resource {
            ResourceId::Tlas => Some(wgpu::BindingResource::AccelerationStructure(
                &self.acceleration.tlas,
            )),
            ResourceId::Vertex => Some(self.vertex_buffer.as_entire_binding()),
            ResourceId::Index => Some(self.index_buffer.as_entire_binding()),
            ResourceId::Geometry => Some(self.geometry_buffer.as_entire_binding()),
            ResourceId::Instance => Some(self.instance_buffer.as_entire_binding()),
            ResourceId::Medium => Some(self.medium_buffer.as_entire_binding()),
            ResourceId::BSSRDFMaterial => Some(self.bssrdf_material_buffer.as_entire_binding()),
            ResourceId::BSSRDFTable => Some(self.bssrdf_tables.records.as_entire_binding()),
            ResourceId::BSSRDFValues => Some(self.bssrdf_tables.values.as_entire_binding()),
            ResourceId::SamplerParams | ResourceId::SamplerTable => {
                self.sampler.bindings().resource(resource)
            }
            ResourceId::MaterialRoot => Some(self.material_root_buffer.as_entire_binding()),
            ResourceId::MaterialNode => Some(self.material_node_buffer.as_entire_binding()),
            ResourceId::AttributeRef => Some(self.attribute_ref_buffer.as_entire_binding()),
            ResourceId::ScalarAttribute => Some(self.scalar_attribute_buffer.as_entire_binding()),
            ResourceId::SpectrumAttribute => {
                Some(self.spectrum_attribute_buffer.as_entire_binding())
            }
            ResourceId::MeasuredBsdf => Some(self.measured_bsdf_buffer.as_entire_binding()),
            ResourceId::MeasuredTable => Some(self.measured_table_buffer.as_entire_binding()),
            ResourceId::TextureRoot => Some(self.texture_root_buffer.as_entire_binding()),
            ResourceId::TextureNode => Some(self.texture_node_buffer.as_entire_binding()),
            ResourceId::TextureChild => Some(self.texture_child_buffer.as_entire_binding()),
            ResourceId::RgbSpectrumTable => {
                Some(self.rgb_spectrum_table_buffer.as_entire_binding())
            }
            ResourceId::LightRecord => Some(self.light_record_buffer.as_entire_binding()),
            ResourceId::LightSamplingModel => {
                Some(self.light_sampling_model_buffer.as_entire_binding())
            }
            ResourceId::LightPosition => Some(self.light_position_buffer.as_entire_binding()),
            ResourceId::TriangleDistribution => Some(self.distribution_buffer.as_entire_binding()),
            ResourceId::PortalInfiniteLight => Some(self.portal_image_buffer.as_entire_binding()),
            ResourceId::PortalDistribution => {
                Some(self.portal_distribution_buffer.as_entire_binding())
            }
            ResourceId::ImageInfiniteSampling => {
                Some(self.image_infinite_sampling_buffer.as_entire_binding())
            }
            ResourceId::ImageInfiniteDistribution => {
                Some(self.image_infinite_distribution_buffer.as_entire_binding())
            }
            ResourceId::ImageInfiniteRowCdf => {
                Some(self.image_infinite_row_cdf_buffer.as_entire_binding())
            }
            ResourceId::LightBvhHeader => Some(self.light_bvh_header_buffer.as_entire_binding()),
            ResourceId::LightBvhNode => Some(self.light_bvh_node_buffer.as_entire_binding()),
            ResourceId::LightLeaf => Some(self.light_leaf_buffer.as_entire_binding()),
            _ => None,
        }
    }

    pub fn replace_material_kind(&mut self, kind: MaterialKind) {
        self.material_table.debug_material_kind = kind.tag();
    }
}

fn buffer_contents<T: bytemuck::Pod>(values: &[T]) -> &[u8] {
    if values.is_empty() {
        // WebGPU validates the minimum binding size against the declared
        // storage-array stride. Keep one zeroed element for an empty runtime
        // array; a fixed byte count is insufficient for larger records such
        // as TextureNodeRecord and DenseSpectrum.
        static EMPTY_STORAGE_ELEMENT: [u8; 2048] = [0; 2048];
        let element_size = std::mem::size_of::<T>();
        assert!(element_size <= EMPTY_STORAGE_ELEMENT.len());
        &EMPTY_STORAGE_ELEMENT[..element_size]
    } else {
        cast_slice(values)
    }
}
