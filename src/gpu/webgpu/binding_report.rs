//! Binding budgets derived from the same composed WGSL and registry as pipelines.
//!
//! This report covers `COMPUTE_STAGES`. The separately constructed BSSRDF probe
//! budget is included when the scene uses subsurface materials.

use crate::util::error::PbrtError;

use super::shader::{compose_source_with_features, resolve_resource_bindings};
use super::stage::{ComputeStageId, ComputeStageSpec, WAVEFRONT_BIND_GROUP_COUNT};
use super::stages::{BindingClass, BindingSpec, RequiredLimits, ResourceId};

#[derive(Debug)]
pub struct StageBindingReport {
    pub stage_id: ComputeStageId,
    pub entry_point: &'static str,
    pub label: &'static str,
    pub bindings: Vec<BindingSpec>,
    pub required: RequiredLimits,
    pub acceleration_structures: u32,
}

#[derive(Debug)]
pub struct BindingReport {
    pub stages: Vec<StageBindingReport>,
    /// Per-field maxima of shader declarations, before layout requirements.
    pub required: RequiredLimits,
    pub noise_enabled: bool,
    pub nanovdb_enabled: bool,
}

/// Signed headroom: negative values denote an overage, including at u32 limits.
#[derive(Debug, Eq, PartialEq)]
pub struct BindingHeadroom {
    pub storage: i64,
    pub uniform: i64,
    pub combined: i64,
    pub bind_groups: i64,
}

impl BindingHeadroom {
    pub fn new(required: RequiredLimits, available: &wgpu::Limits) -> Self {
        Self {
            storage: i64::from(available.max_storage_buffers_per_shader_stage)
                - i64::from(required.storage_buffers_per_shader_stage),
            uniform: i64::from(available.max_uniform_buffers_per_shader_stage)
                - i64::from(required.uniform_buffers_per_shader_stage),
            combined: i64::from(available.max_buffers_and_acceleration_structures_per_shader_stage)
                - i64::from(required.buffers_and_acceleration_structures_per_shader_stage),
            bind_groups: i64::from(available.max_bind_groups) - i64::from(required.bind_groups),
        }
    }

    pub fn fits(&self) -> bool {
        self.storage >= 0 && self.uniform >= 0 && self.combined >= 0 && self.bind_groups >= 0
    }
}

impl BindingReport {
    pub fn binding_array_usage(
        &self,
        geometry_page_count: u32,
        texture_image_count: u32,
        texture_sampler_count: u32,
        subsurface_enabled: bool,
    ) -> Result<(u32, u32, u32), PbrtError> {
        let mut max_elements = 0u32;
        let mut max_samplers = 0u32;
        let mut max_metal_slots = 0u32;
        for stage in &self.stages {
            let mut elements = 0u32;
            let mut samplers = 0u32;
            let mut arrays = 0u32;
            for binding in &stage.bindings {
                let count = match binding.resource {
                    ResourceId::Vertex | ResourceId::Index => geometry_page_count,
                    ResourceId::TextureImageArray => texture_image_count,
                    ResourceId::TextureSamplerArray => {
                        samplers = texture_sampler_count;
                        texture_sampler_count
                    }
                    _ => continue,
                };
                elements = elements.checked_add(count).ok_or_else(|| {
                    PbrtError::error("WebGPU binding array element count overflowed.")
                })?;
                arrays = arrays
                    .checked_add(1)
                    .ok_or_else(|| PbrtError::error("WebGPU binding array count overflowed."))?;
            }
            max_elements = max_elements.max(elements);
            max_samplers = max_samplers.max(samplers);
            max_metal_slots = max_metal_slots.max(
                stage
                    .required
                    .buffers_and_acceleration_structures_per_shader_stage
                    .checked_add(arrays)
                    .ok_or_else(|| {
                        PbrtError::error("WebGPU Metal buffer slot count overflowed.")
                    })?,
            );
        }
        if subsurface_enabled {
            let probe_elements = geometry_page_count.checked_mul(2).ok_or_else(|| {
                PbrtError::error("WebGPU BSSRDF probe binding array element count overflowed.")
            })?;
            max_elements = max_elements.max(probe_elements);
            max_metal_slots = max_metal_slots.max(10);
        }
        Ok((max_elements, max_samplers, max_metal_slots))
    }

    pub fn for_stages(
        canonical_bindings: &[BindingSpec],
        stages: &[ComputeStageSpec],
        noise_enabled: bool,
        nanovdb_enabled: bool,
    ) -> Result<Self, PbrtError> {
        let mut required = RequiredLimits::default();
        let stages = stages
            .iter()
            .map(|stage| {
                let source =
                    compose_source_with_features(stage.source, noise_enabled, nanovdb_enabled);
                let bindings =
                    resolve_resource_bindings(canonical_bindings, stage.entry_point, &source)?;
                let stage_required = RequiredLimits::from_bindings(&bindings)?;
                required = required.maximum(stage_required);
                let acceleration_structures = bindings
                    .iter()
                    .filter(|binding| binding.class == BindingClass::AccelerationStructure)
                    .count() as u32;
                Ok(StageBindingReport {
                    stage_id: stage.id,
                    entry_point: stage.entry_point,
                    label: stage.label,
                    bindings,
                    required: stage_required,
                    acceleration_structures,
                })
            })
            .collect::<Result<Vec<_>, PbrtError>>()?;
        Ok(Self {
            stages,
            required,
            noise_enabled,
            nanovdb_enabled,
        })
    }

    /// The deployed layouts reserve group 1 even for stages without textures.
    pub fn device_required(&self) -> RequiredLimits {
        RequiredLimits {
            bind_groups: self.required.bind_groups.max(WAVEFRONT_BIND_GROUP_COUNT),
            ..self.required
        }
    }

    /// Stable, developer-facing text. When requested, omit stages that fit;
    /// layout-wide requirements remain visible even if all declarations fit.
    pub fn format(&self, available: Option<&wgpu::Limits>, only_over_budget: bool) -> String {
        let mut output = format!(
            "WebGPU binding report: scope=wavefront, noise={}, nanovdb={}\n",
            self.noise_enabled, self.nanovdb_enabled,
        );
        let required = self.device_required();
        output.push_str(&format!(
            "requested: storage={} uniform={} combined={} bind_groups={} (layout minimum={})\n",
            required.storage_buffers_per_shader_stage,
            required.uniform_buffers_per_shader_stage,
            required.buffers_and_acceleration_structures_per_shader_stage,
            required.bind_groups,
            WAVEFRONT_BIND_GROUP_COUNT,
        ));
        if let Some(limits) = available {
            output.push_str(&format!(
                "limits: storage={} uniform={} combined={} bind_groups={}\n",
                limits.max_storage_buffers_per_shader_stage,
                limits.max_uniform_buffers_per_shader_stage,
                limits.max_buffers_and_acceleration_structures_per_shader_stage,
                limits.max_bind_groups,
            ));
            let headroom = BindingHeadroom::new(required, limits);
            output.push_str(&format!(
                "request headroom: storage={} uniform={} combined={} bind_groups={}\n",
                headroom.storage, headroom.uniform, headroom.combined, headroom.bind_groups,
            ));
        } else {
            output.push_str("limits: unavailable (no adapter selected)\n");
        }
        let mut stages = self.stages.iter().collect::<Vec<_>>();
        stages.sort_by(|left, right| {
            right
                .required
                .buffers_and_acceleration_structures_per_shader_stage
                .cmp(
                    &left
                        .required
                        .buffers_and_acceleration_structures_per_shader_stage,
                )
                .then_with(|| left.entry_point.cmp(right.entry_point))
        });
        for stage in stages {
            let headroom = available.map(|limits| BindingHeadroom::new(stage.required, limits));
            if only_over_budget && headroom.as_ref().is_some_and(BindingHeadroom::fits) {
                continue;
            }
            output.push_str(&format!(
                "stage={} storage={} uniform={} as={} combined={} declared_bind_groups={}\n",
                stage.entry_point,
                stage.required.storage_buffers_per_shader_stage,
                stage.required.uniform_buffers_per_shader_stage,
                stage.acceleration_structures,
                stage
                    .required
                    .buffers_and_acceleration_structures_per_shader_stage,
                stage.required.bind_groups,
            ));
            if let Some(headroom) = headroom {
                output.push_str(&format!(
                    "  headroom: storage={} uniform={} combined={} bind_groups={}\n",
                    headroom.storage, headroom.uniform, headroom.combined, headroom.bind_groups,
                ));
            }
            for (category, buffer_resources) in [("buffers", true), ("textures/samplers", false)] {
                output.push_str(&format!("  {category}:\n"));
                for binding in &stage.bindings {
                    let is_buffer = matches!(
                        binding.class,
                        BindingClass::Storage
                            | BindingClass::Uniform
                            | BindingClass::AccelerationStructure
                    );
                    if is_buffer == buffer_resources {
                        output.push_str(&format!(
                            "    group={} binding={} resource={:?} class={:?} access={:?}\n",
                            binding.group,
                            binding.binding,
                            binding.resource,
                            binding.class,
                            binding.access,
                        ));
                    }
                }
            }
        }
        output
    }
}
