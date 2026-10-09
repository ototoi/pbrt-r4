use bytemuck::{cast_slice, Zeroable};
use wgpu::util::DeviceExt;

use super::abi::{BSSRDFMaterialRecord, BSSRDFTableRecord};
use super::material::MaterialKind;
use super::scene::Scene;
use crate::gpu::flat::{self, TabulatedBSSRDFTable, BSSRDF};
use crate::util::error::PbrtError;

pub struct BSSRDFTableData {
    pub records: Vec<BSSRDFTableRecord>,
    pub values: Vec<f32>,
}

pub fn convert_bssrdf_materials(
    materials: &[BSSRDF],
    table_count: usize,
) -> Result<Vec<BSSRDFMaterialRecord>, PbrtError> {
    if materials
        .iter()
        .any(|material| material.table_index as usize >= table_count)
    {
        return Err(PbrtError::error(
            "BSSRDF material references an invalid table.",
        ));
    }
    Ok(materials
        .iter()
        .map(|material| BSSRDFMaterialRecord {
            scale: material.scale,
            eta: material.eta,
            table_index: material.table_index,
            coefficient_kind: match material.coefficient_kind {
                flat::BSSRDFCoefficientKind::Sigma => 0,
                flat::BSSRDFCoefficientKind::ReflectanceMfp => 1,
            },
        })
        .collect())
}

impl BSSRDFTableData {
    pub fn from_flat(tables: &[TabulatedBSSRDFTable]) -> Result<Self, PbrtError> {
        let mut records = Vec::new();
        let mut values = Vec::new();
        for table in tables {
            let n_rho = table.rho_samples.len();
            let n_radius = table.radius_samples.len();
            let count = n_rho
                .checked_mul(n_radius)
                .ok_or_else(|| PbrtError::error("BSSRDF table dimensions overflowed."))?;
            if n_rho < 2
                || n_radius < 2
                || table.profile.len() != count
                || table.profile_cdf.len() != count
                || table.rho_eff.len() != n_rho
            {
                return Err(PbrtError::error(
                    "BSSRDF table dimensions are inconsistent.",
                ));
            }
            let mut offsets = [0u32; 5];
            for (i, data) in [
                &table.rho_samples,
                &table.radius_samples,
                &table.profile,
                &table.rho_eff,
                &table.profile_cdf,
            ]
            .into_iter()
            .enumerate()
            {
                if data.iter().any(|v| !v.is_finite()) {
                    return Err(PbrtError::error("BSSRDF table contains non-finite values."));
                }
                offsets[i] = u32::try_from(values.len())
                    .map_err(|_| PbrtError::error("BSSRDF table storage exceeds u32."))?;
                values.extend_from_slice(data);
            }
            if table.rho_samples.windows(2).any(|w| w[0] >= w[1])
                || table.radius_samples.windows(2).any(|w| w[0] >= w[1])
            {
                return Err(PbrtError::error(
                    "BSSRDF table nodes are not strictly increasing.",
                ));
            }
            records.push(BSSRDFTableRecord {
                rho_offset: offsets[0],
                radius_offset: offsets[1],
                profile_offset: offsets[2],
                rho_eff_offset: offsets[3],
                cdf_offset: offsets[4],
                rho_count: u32::try_from(n_rho)
                    .map_err(|_| PbrtError::error("BSSRDF rho count exceeds u32."))?,
                radius_count: u32::try_from(n_radius)
                    .map_err(|_| PbrtError::error("BSSRDF radius count exceeds u32."))?,
                padding: 0,
            });
        }
        u32::try_from(values.len())
            .map_err(|_| PbrtError::error("BSSRDF table storage exceeds u32."))?;
        Ok(Self { records, values })
    }

    pub fn upload(&self, device: &wgpu::Device) -> Result<BSSRDFTableResources, PbrtError> {
        let empty_record = [BSSRDFTableRecord::zeroed()];
        let records = if self.records.is_empty() {
            &empty_record[..]
        } else {
            &self.records
        };
        let values = if self.values.is_empty() {
            &[0.0][..]
        } else {
            &self.values
        };
        let limit = u64::from(device.limits().max_storage_buffer_binding_size);
        if std::mem::size_of_val(records) as u64 > limit
            || std::mem::size_of_val(values) as u64 > limit
        {
            return Err(PbrtError::error(
                "BSSRDF table storage exceeds device limits.",
            ));
        }
        Ok(BSSRDFTableResources {
            records: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BSSRDF tables"),
                contents: cast_slice(records),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            values: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BSSRDF table values"),
                contents: cast_slice(values),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        })
    }
}

pub struct BSSRDFTableResources {
    pub records: wgpu::Buffer,
    pub values: wgpu::Buffer,
}

pub struct BSSRDFProbePipeline {
    pub pipeline: wgpu::ComputePipeline,
}

pub fn probe_shader_source() -> String {
    [
        "enable wgpu_ray_query;\n",
        include_str!("shaders/types.wgsl"),
        include_str!("shaders/bssrdf_probe_bindings.wgsl"),
        include_str!("shaders/lib/float.wgsl"),
        include_str!("shaders/lib/hash_u32.wgsl"),
        include_str!("shaders/lib/vecmath.wgsl"),
        include_str!("shaders/lib/interaction.wgsl"),
        include_str!("shaders/lib/bssrdf.wgsl"),
        include_str!("shaders/sample_subsurface_probe.wgsl"),
    ]
    .join("\n")
}

impl BSSRDFProbePipeline {
    pub fn new(device: &wgpu::Device) -> Result<Self, PbrtError> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BSSRDF probe"),
            source: wgpu::ShaderSource::Wgsl(probe_shader_source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BSSRDF probe"),
            layout: None,
            module: &module,
            entry_point: Some("sample_subsurface_probe"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(PbrtError::error(&format!(
                "BSSRDF probe pipeline creation failed: {error}"
            )));
        }
        Ok(Self { pipeline })
    }

    pub fn bind_group(
        &self,
        device: &wgpu::Device,
        scene: &Scene,
        tables: &BSSRDFTableResources,
        work: &wgpu::Buffer,
        results: &wgpu::Buffer,
        error: &wgpu::Buffer,
    ) -> Result<wgpu::BindGroup, PbrtError> {
        if scene
            .material_nodes
            .iter()
            .any(|node| node.kind == MaterialKind::AlphaMask.tag())
        {
            return Err(PbrtError::error(
                "Subsurface probe alpha-mask traversal is not implemented yet.",
            ));
        }
        let buffers = [
            &scene.vertex_buffer,
            &scene.index_buffer,
            &scene.geometry_buffer,
            &scene.instance_buffer,
            &tables.records,
            &tables.values,
            work,
            results,
            error,
        ];
        let mut entries = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::AccelerationStructure(&scene.acceleration.tlas),
        });
        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BSSRDF probe"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        }))
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, group: &wgpu::BindGroup, count: u32) {
        if count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        let groups = count.div_ceil(64);
        pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
    }
}
