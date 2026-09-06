//! Stage resource contracts shared by pipeline construction and shader composition.
//!
//! This module deliberately contains no `wgpu` handles.  A stage describes the
//! resources it needs; the resource adapter turns that description into a
//! bind-group layout after device capabilities have been checked.

use crate::util::error::PbrtError;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ResourceId {
    CameraParams,
    SampleParams,
    DepthParams,
    LightSamplingParams,
    FilmParams,
    DebugParams,
    RenderError,
    Tlas,
    Vertex,
    Index,
    Geometry,
    Instance,
    CurrentRay,
    NextRay,
    ShadowQueue,
    HitRecord,
    Surface,
    PixelSampleState,
    RaySamples,
    ShadingContext,
    LightRecord,
    PointLight,
    AreaLight,
    LightBvh,
    LightBvhHeader,
    LightBvhNode,
    LightLeaf,
    PrimitiveDistributionMap,
    TriangleDistribution,
    LightCandidate,
    ConstantBxdf,
    Film,
    MaterialTable,
    MaterialRecord,
    MaterialAttribute,
    ScalarAttribute,
    SpectrumAttribute,
    TextureAttribute,
    ScatteringModel,
    ScatteringNode,
    ScatteringChild,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingClass {
    Uniform,
    Storage,
    AccelerationStructure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

impl Access {
    pub fn permits_write(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindingSpec {
    pub group: u32,
    pub binding: u32,
    pub resource: ResourceId,
    pub class: BindingClass,
    pub access: Access,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageId {
    BeginSample,
    GenerateCameraRays,
    ResetDepthQueues,
    GenerateRaySamples,
    TraceClosest,
    BuildSurface,
    HandleEscaped,
    HandleEmission,
    BuildShadingContext,
    SampleDirectLight,
    ScatterDiffuse,
    ScatterDielectric,
    ScatterThinDielectric,
    ScatterLayered,
    TraceShadow,
    DebugSurface,
    AccumulateFilm,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageSpec {
    pub id: StageId,
    pub entry_point: &'static str,
    pub bindings: &'static [BindingSpec],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RequiredLimits {
    pub storage_buffers_per_shader_stage: u32,
    pub uniform_buffers_per_shader_stage: u32,
    pub bind_groups: u32,
}

/// Storage bindings currently present in the shared WebGPU bind group.
///
/// StageSpec negotiation remains intentionally independent while the fixed
/// layout is being migrated. The device request must still cover every entry
/// in that layout, including resources used only by some composed stages.

/// Canonical binding registry for the currently deployed wavefront layout.
///
/// The registry is the migration seam between the old single-group shader ABI
/// and the stage-specific layouts. Pipeline construction must consume this list
/// instead of duplicating binding numbers.
pub fn canonical_wavefront_bindings() -> Vec<BindingSpec> {
    let mut bindings = Vec::with_capacity(26);
    let mut push = |binding, resource, class, access| {
        bindings.push(BindingSpec {
            group: 0,
            binding,
            resource,
            class,
            access,
        });
    };
    push(
        0,
        ResourceId::CameraParams,
        BindingClass::Uniform,
        Access::Read,
    );
    push(
        1,
        ResourceId::SampleParams,
        BindingClass::Uniform,
        Access::Read,
    );
    push(
        2,
        ResourceId::Tlas,
        BindingClass::AccelerationStructure,
        Access::Read,
    );
    for (binding, resource) in [
        (3, ResourceId::Vertex),
        (4, ResourceId::Index),
        (5, ResourceId::Geometry),
        (6, ResourceId::Instance),
    ] {
        push(binding, resource, BindingClass::Storage, Access::Read);
    }
    push(
        8,
        ResourceId::Surface,
        BindingClass::Storage,
        Access::ReadWrite,
    );
    push(
        9,
        ResourceId::Film,
        BindingClass::Storage,
        Access::ReadWrite,
    );
    push(
        10,
        ResourceId::RaySamples,
        BindingClass::Storage,
        Access::ReadWrite,
    );
    push(
        11,
        ResourceId::MaterialTable,
        BindingClass::Uniform,
        Access::Read,
    );
    push(
        12,
        ResourceId::LightSamplingParams,
        BindingClass::Uniform,
        Access::Read,
    );
    for (binding, resource) in [
        (13, ResourceId::MaterialRecord),
        (14, ResourceId::MaterialAttribute),
        (15, ResourceId::ScalarAttribute),
        (16, ResourceId::ScatteringModel),
        (17, ResourceId::ScatteringNode),
        (18, ResourceId::ScatteringChild),
        (19, ResourceId::SpectrumAttribute),
        (20, ResourceId::LightRecord),
        (21, ResourceId::PointLight),
        (22, ResourceId::AreaLight),
        (23, ResourceId::TriangleDistribution),
        (24, ResourceId::LightBvhHeader),
        (25, ResourceId::LightBvhNode),
        (26, ResourceId::LightLeaf),
    ] {
        push(binding, resource, BindingClass::Storage, Access::Read);
    }
    bindings
}

pub const FIXED_LAYOUT_STORAGE_BUFFERS_PER_SHADER_STAGE: u32 = 24;
pub const FIXED_LAYOUT_UNIFORM_BUFFERS_PER_SHADER_STAGE: u32 = 4;

impl RequiredLimits {
    pub fn from_stages(stages: &[StageSpec]) -> Result<Self, PbrtError> {
        let mut required = Self::default();
        for stage in stages {
            let mut storage = 0;
            let mut uniform = 0;
            let mut max_group = 0;
            for binding in stage.bindings {
                max_group = max_group.max(binding.group);
                match binding.class {
                    BindingClass::Storage => storage += 1,
                    BindingClass::Uniform => uniform += 1,
                    BindingClass::AccelerationStructure => {}
                }
            }
            required.storage_buffers_per_shader_stage =
                required.storage_buffers_per_shader_stage.max(storage);
            required.uniform_buffers_per_shader_stage =
                required.uniform_buffers_per_shader_stage.max(uniform);
            required.bind_groups = required.bind_groups.max(max_group + 1);
            validate_stage(stage)?;
        }
        Ok(required)
    }
}

fn validate_stage(stage: &StageSpec) -> Result<(), PbrtError> {
    for (index, left) in stage.bindings.iter().enumerate() {
        for right in &stage.bindings[index + 1..] {
            if left.group == right.group && left.binding == right.binding {
                return Err(PbrtError::error(&format!(
                    "Stage {:?} declares duplicate binding group {} binding {}.",
                    stage.id, left.group, left.binding
                )));
            }
            if left.resource == right.resource && left.class != right.class {
                return Err(PbrtError::error(&format!(
                    "Stage {:?} binds resource {:?} with conflicting classes.",
                    stage.id, left.resource
                )));
            }
        }
    }
    Ok(())
}

const SAMPLE_DIRECT_LIGHT_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        group: 0,
        binding: 0,
        resource: ResourceId::LightSamplingParams,
        class: BindingClass::Uniform,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 0,
        resource: ResourceId::CurrentRay,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 1,
        resource: ResourceId::ShadingContext,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 2,
        resource: ResourceId::RaySamples,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 3,
        resource: ResourceId::LightRecord,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 4,
        resource: ResourceId::PointLight,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 5,
        resource: ResourceId::AreaLight,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 6,
        resource: ResourceId::LightBvh,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 7,
        resource: ResourceId::LightLeaf,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 8,
        resource: ResourceId::TriangleDistribution,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 9,
        resource: ResourceId::PrimitiveDistributionMap,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 10,
        resource: ResourceId::Vertex,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 11,
        resource: ResourceId::Index,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 12,
        resource: ResourceId::Geometry,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 13,
        resource: ResourceId::Instance,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 2,
        binding: 0,
        resource: ResourceId::LightCandidate,
        class: BindingClass::Storage,
        access: Access::Write,
    },
];

const BEGIN_SAMPLE_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        group: 0,
        binding: 0,
        resource: ResourceId::SampleParams,
        class: BindingClass::Uniform,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 0,
        resource: ResourceId::PixelSampleState,
        class: BindingClass::Storage,
        access: Access::Write,
    },
    BindingSpec {
        group: 1,
        binding: 1,
        resource: ResourceId::CurrentRay,
        class: BindingClass::Storage,
        access: Access::ReadWrite,
    },
    BindingSpec {
        group: 1,
        binding: 2,
        resource: ResourceId::NextRay,
        class: BindingClass::Storage,
        access: Access::ReadWrite,
    },
];

const TRACE_CLOSEST_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        group: 0,
        binding: 0,
        resource: ResourceId::Tlas,
        class: BindingClass::AccelerationStructure,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 0,
        resource: ResourceId::CurrentRay,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 1,
        resource: ResourceId::HitRecord,
        class: BindingClass::Storage,
        access: Access::Write,
    },
];

const BUILD_SURFACE_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        group: 0,
        binding: 0,
        resource: ResourceId::DepthParams,
        class: BindingClass::Uniform,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 0,
        resource: ResourceId::CurrentRay,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 1,
        resource: ResourceId::HitRecord,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 2,
        resource: ResourceId::Vertex,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 3,
        resource: ResourceId::Index,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 4,
        resource: ResourceId::Geometry,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 5,
        resource: ResourceId::Instance,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 6,
        resource: ResourceId::Surface,
        class: BindingClass::Storage,
        access: Access::ReadWrite,
    },
];

const SCATTER_BINDINGS: &[BindingSpec] = &[
    BindingSpec {
        group: 0,
        binding: 0,
        resource: ResourceId::SampleParams,
        class: BindingClass::Uniform,
        access: Access::Read,
    },
    BindingSpec {
        group: 0,
        binding: 1,
        resource: ResourceId::DepthParams,
        class: BindingClass::Uniform,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 0,
        resource: ResourceId::CurrentRay,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 1,
        resource: ResourceId::Surface,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 2,
        resource: ResourceId::ShadingContext,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 3,
        resource: ResourceId::RaySamples,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 4,
        resource: ResourceId::LightCandidate,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 1,
        binding: 5,
        resource: ResourceId::ConstantBxdf,
        class: BindingClass::Storage,
        access: Access::Read,
    },
    BindingSpec {
        group: 2,
        binding: 0,
        resource: ResourceId::NextRay,
        class: BindingClass::Storage,
        access: Access::ReadWrite,
    },
    BindingSpec {
        group: 2,
        binding: 1,
        resource: ResourceId::ShadowQueue,
        class: BindingClass::Storage,
        access: Access::ReadWrite,
    },
    BindingSpec {
        group: 2,
        binding: 2,
        resource: ResourceId::RenderError,
        class: BindingClass::Storage,
        access: Access::ReadWrite,
    },
];

pub fn initial_stage_specs() -> Vec<StageSpec> {
    vec![
        StageSpec {
            id: StageId::BeginSample,
            entry_point: "begin_sample",
            bindings: BEGIN_SAMPLE_BINDINGS,
        },
        StageSpec {
            id: StageId::TraceClosest,
            entry_point: "trace_closest",
            bindings: TRACE_CLOSEST_BINDINGS,
        },
        StageSpec {
            id: StageId::BuildSurface,
            entry_point: "build_surface",
            bindings: BUILD_SURFACE_BINDINGS,
        },
        StageSpec {
            id: StageId::SampleDirectLight,
            entry_point: "sample_direct_light",
            bindings: SAMPLE_DIRECT_LIGHT_BINDINGS,
        },
        StageSpec {
            id: StageId::ScatterDiffuse,
            entry_point: "scatter_diffuse",
            bindings: SCATTER_BINDINGS,
        },
    ]
}

/// Returns the complete contract registry used to size the device and validate
/// every wavefront variant.  `initial_stage_specs` remains the small bootstrap
/// set used by focused contract tests.
pub fn all_stage_specs() -> Vec<StageSpec> {
    vec![
        StageSpec {
            id: StageId::BeginSample,
            entry_point: "begin_sample",
            bindings: BEGIN_SAMPLE_BINDINGS,
        },
        StageSpec {
            id: StageId::GenerateCameraRays,
            entry_point: "generate_primary_rays",
            bindings: BEGIN_SAMPLE_BINDINGS,
        },
        StageSpec {
            id: StageId::ResetDepthQueues,
            entry_point: "reset_classification_queues",
            bindings: BEGIN_SAMPLE_BINDINGS,
        },
        StageSpec {
            id: StageId::GenerateRaySamples,
            entry_point: "prepare_sample",
            bindings: BEGIN_SAMPLE_BINDINGS,
        },
        StageSpec {
            id: StageId::TraceClosest,
            entry_point: "trace_closest",
            bindings: TRACE_CLOSEST_BINDINGS,
        },
        StageSpec {
            id: StageId::BuildSurface,
            entry_point: "build_surface",
            bindings: BUILD_SURFACE_BINDINGS,
        },
        StageSpec {
            id: StageId::HandleEscaped,
            entry_point: "handle_escaped",
            bindings: SCATTER_BINDINGS,
        },
        StageSpec {
            id: StageId::HandleEmission,
            entry_point: "handle_emissive",
            bindings: SCATTER_BINDINGS,
        },
        StageSpec {
            id: StageId::BuildShadingContext,
            entry_point: "build_shading_context",
            bindings: BUILD_SURFACE_BINDINGS,
        },
        StageSpec {
            id: StageId::SampleDirectLight,
            entry_point: "sample_direct_light",
            bindings: SAMPLE_DIRECT_LIGHT_BINDINGS,
        },
        StageSpec {
            id: StageId::ScatterDiffuse,
            entry_point: "scatter_diffuse",
            bindings: SCATTER_BINDINGS,
        },
        StageSpec {
            id: StageId::ScatterDielectric,
            entry_point: "scatter_dielectric",
            bindings: SCATTER_BINDINGS,
        },
        StageSpec {
            id: StageId::ScatterThinDielectric,
            entry_point: "scatter_thin_dielectric",
            bindings: SCATTER_BINDINGS,
        },
        StageSpec {
            id: StageId::ScatterLayered,
            entry_point: "scatter_layered",
            bindings: SCATTER_BINDINGS,
        },
        StageSpec {
            id: StageId::TraceShadow,
            entry_point: "trace_shadow",
            bindings: TRACE_CLOSEST_BINDINGS,
        },
        StageSpec {
            id: StageId::DebugSurface,
            entry_point: "debug_surface",
            bindings: BUILD_SURFACE_BINDINGS,
        },
        StageSpec {
            id: StageId::AccumulateFilm,
            entry_point: "accumulate_sample",
            bindings: SCATTER_BINDINGS,
        },
    ]
}
