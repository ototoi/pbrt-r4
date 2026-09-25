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
    RenderError,
    Tlas,
    Vertex,
    Index,
    Geometry,
    Instance,
    QueueCounters,
    CurrentRay,
    NextRay,
    ShadowQueue,
    MaterialRayQueue,
    AttributesEvalWorkItems,
    HitAreaRayQueue,
    EscapedRayQueue,
    Surface,
    PixelSampleState,
    LightRecord,
    LightSamplingModel,
    LightPosition,
    LightBvhHeader,
    LightBvhNode,
    LightLeaf,
    TriangleDistribution,
    PortalLightCandidate,
    DirectLightSample,
    DirectEvalQueue,
    ScatterDiffuseQueue,
    ScatterDiffuseTransmissionQueue,
    ScatterConductorQueue,
    ScatterDielectricQueue,
    ScatterThinDielectricQueue,
    ScatterMeasuredQueue,
    ScatterCoatedQueue,
    QueueDispatchArgs,
    Film,
    MaterialTable,
    AttributeRef,
    ScalarAttribute,
    SpectrumAttribute,
    TextureRoot,
    TextureNode,
    TextureChild,
    TextureImageArray,
    TextureSamplerArray,
    RgbSpectrumTable,
    NoiseTable,
    TextureEvalResult,
    MaterialRoot,
    MaterialNode,
    MeasuredBsdf,
    MeasuredTable,
    SamplerParams,
    SamplerTable,
    PortalInfiniteLight,
    PortalDistribution,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingClass {
    Uniform,
    Storage,
    AccelerationStructure,
    SampledTexture,
    IntegerTexture,
    Sampler,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RequiredLimits {
    pub storage_buffers_per_shader_stage: u32,
    pub uniform_buffers_per_shader_stage: u32,
    pub buffers_and_acceleration_structures_per_shader_stage: u32,
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
    let mut bindings = Vec::with_capacity(44);
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
        43,
        ResourceId::NoiseTable,
        BindingClass::IntegerTexture,
        Access::Read,
    );
    push(
        44,
        ResourceId::TextureEvalResult,
        BindingClass::Storage,
        Access::ReadWrite,
    );
    push(
        45,
        ResourceId::MaterialRoot,
        BindingClass::Storage,
        Access::Read,
    );
    push(
        46,
        ResourceId::MaterialNode,
        BindingClass::Storage,
        Access::Read,
    );
    push(
        47,
        ResourceId::MeasuredBsdf,
        BindingClass::Storage,
        Access::Read,
    );
    push(
        48,
        ResourceId::MeasuredTable,
        BindingClass::Storage,
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
        7,
        ResourceId::FilmParams,
        BindingClass::Uniform,
        Access::Read,
    );
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
        ResourceId::QueueCounters,
        BindingClass::Storage,
        Access::ReadWrite,
    );
    push(
        11,
        ResourceId::RenderError,
        BindingClass::Storage,
        Access::ReadWrite,
    );
    push(
        12,
        ResourceId::PixelSampleState,
        BindingClass::Storage,
        Access::ReadWrite,
    );
    for (binding, resource) in [
        (13, ResourceId::CurrentRay),
        (14, ResourceId::NextRay),
        (15, ResourceId::ShadowQueue),
        (16, ResourceId::MaterialRayQueue),
        (37, ResourceId::AttributesEvalWorkItems),
        (17, ResourceId::HitAreaRayQueue),
        (18, ResourceId::EscapedRayQueue),
    ] {
        push(binding, resource, BindingClass::Storage, Access::ReadWrite);
    }
    push(
        60,
        ResourceId::QueueDispatchArgs,
        BindingClass::Storage,
        Access::ReadWrite,
    );
    push(
        19,
        ResourceId::MaterialTable,
        BindingClass::Uniform,
        Access::Read,
    );
    push(
        20,
        ResourceId::LightSamplingParams,
        BindingClass::Uniform,
        Access::Read,
    );
    push(
        21,
        ResourceId::SamplerParams,
        BindingClass::Uniform,
        Access::Read,
    );
    push(
        49,
        ResourceId::SamplerTable,
        BindingClass::IntegerTexture,
        Access::Read,
    );
    push(
        24,
        ResourceId::PortalInfiniteLight,
        BindingClass::Storage,
        Access::Read,
    );
    push(
        25,
        ResourceId::PortalDistribution,
        BindingClass::Storage,
        Access::Read,
    );
    push(
        50,
        ResourceId::PortalLightCandidate,
        BindingClass::Storage,
        Access::ReadWrite,
    );
    push(
        51,
        ResourceId::DirectLightSample,
        BindingClass::Storage,
        Access::ReadWrite,
    );
    for (binding, resource) in [
        (52, ResourceId::DirectEvalQueue),
        (53, ResourceId::ScatterDiffuseQueue),
        (54, ResourceId::ScatterDiffuseTransmissionQueue),
        (55, ResourceId::ScatterConductorQueue),
        (56, ResourceId::ScatterDielectricQueue),
        (57, ResourceId::ScatterThinDielectricQueue),
        (58, ResourceId::ScatterMeasuredQueue),
        (59, ResourceId::ScatterCoatedQueue),
    ] {
        push(binding, resource, BindingClass::Storage, Access::ReadWrite);
    }
    for (binding, resource) in [
        (22, ResourceId::AttributeRef),
        (23, ResourceId::ScalarAttribute),
        (28, ResourceId::LightRecord),
        (29, ResourceId::LightSamplingModel),
        (30, ResourceId::TriangleDistribution),
        (31, ResourceId::LightBvhHeader),
        (32, ResourceId::LightBvhNode),
        (33, ResourceId::LightLeaf),
        (34, ResourceId::SpectrumAttribute),
        (35, ResourceId::TextureNode),
        (38, ResourceId::TextureChild),
        (36, ResourceId::LightPosition),
        (41, ResourceId::RgbSpectrumTable),
        (42, ResourceId::TextureRoot),
    ] {
        push(binding, resource, BindingClass::Storage, Access::Read);
    }
    push(
        0,
        ResourceId::TextureImageArray,
        BindingClass::SampledTexture,
        Access::Read,
    );
    push(
        1,
        ResourceId::TextureSamplerArray,
        BindingClass::Sampler,
        Access::Read,
    );
    for binding in bindings.iter_mut() {
        if matches!(
            binding.resource,
            ResourceId::TextureImageArray | ResourceId::TextureSamplerArray
        ) {
            binding.group = 1;
        }
    }
    bindings
}

impl RequiredLimits {
    pub fn from_bindings(bindings: &[BindingSpec]) -> Result<Self, PbrtError> {
        let mut required = Self::default();
        for (index, left) in bindings.iter().enumerate() {
            for right in &bindings[index + 1..] {
                if left.group == right.group && left.binding == right.binding {
                    return Err(PbrtError::error(&format!(
                        "Duplicate binding group {} binding {} in layout registry.",
                        left.group, left.binding
                    )));
                }
            }
            required.bind_groups = required.bind_groups.max(left.group + 1);
            match left.class {
                BindingClass::Storage => required.storage_buffers_per_shader_stage += 1,
                BindingClass::Uniform => required.uniform_buffers_per_shader_stage += 1,
                BindingClass::AccelerationStructure => {}
                BindingClass::SampledTexture
                | BindingClass::IntegerTexture
                | BindingClass::Sampler => {}
            }
            if matches!(
                left.class,
                BindingClass::Storage | BindingClass::Uniform | BindingClass::AccelerationStructure
            ) {
                required.buffers_and_acceleration_structures_per_shader_stage += 1;
            }
        }
        Ok(required)
    }
}
