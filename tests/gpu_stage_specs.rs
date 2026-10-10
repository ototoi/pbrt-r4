use std::collections::HashSet;

use pbrt_r4::gpu::webgpu::shader::{compose_source_with_noise, resource_bindings};
use pbrt_r4::gpu::webgpu::stage::COMPUTE_STAGES;
use pbrt_r4::gpu::webgpu::stages::{
    canonical_wavefront_bindings, Access, BindingClass, BindingSpec, RequiredLimits, ResourceId,
};

#[test]
fn duplicate_bindings_are_rejected_before_device_creation() {
    let bindings = [
        BindingSpec {
            group: 0,
            binding: 0,
            resource: ResourceId::SampleParams,
            class: BindingClass::Uniform,
            access: Access::Read,
        },
        BindingSpec {
            group: 0,
            binding: 0,
            resource: ResourceId::DepthParams,
            class: BindingClass::Uniform,
            access: Access::Read,
        },
    ];
    let result = RequiredLimits::from_bindings(&bindings);
    assert!(result.is_err());
}

#[test]
fn canonical_wavefront_layout_has_unique_bindings_and_named_resources() {
    let bindings = canonical_wavefront_bindings();
    assert_eq!(bindings.len(), 74);
    for (index, left) in bindings.iter().enumerate() {
        for right in &bindings[index + 1..] {
            assert!(!(left.group == right.group && left.binding == right.binding));
        }
    }
    assert_eq!(
        bindings.iter().find(|b| b.binding == 21).unwrap().resource,
        ResourceId::SamplerParams
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 19).unwrap().resource,
        ResourceId::MaterialTable
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 22).unwrap().resource,
        ResourceId::AttributeRef
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 46).unwrap().resource,
        ResourceId::MaterialNode
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 47).unwrap().resource,
        ResourceId::MeasuredBsdf
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 48).unwrap().resource,
        ResourceId::MeasuredTable
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 10).unwrap().resource,
        ResourceId::QueueCounters
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 13).unwrap().resource,
        ResourceId::CurrentRay
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 72).unwrap().resource,
        ResourceId::ImageInfiniteSampling
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 73).unwrap().resource,
        ResourceId::ImageInfiniteDistribution
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 74).unwrap().resource,
        ResourceId::ImageInfiniteRowCdf
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 75).unwrap().resource,
        ResourceId::UniformGridMedium
    );
    assert_eq!(
        bindings.iter().find(|b| b.binding == 76).unwrap().resource,
        ResourceId::VolumeData
    );
}

#[test]
fn canonical_layout_drives_required_limits() {
    let limits = RequiredLimits::from_bindings(&canonical_wavefront_bindings()).unwrap();
    assert_eq!(limits.storage_buffers_per_shader_stage, 63);
    assert_eq!(limits.uniform_buffers_per_shader_stage, 6);
    assert_eq!(
        limits.buffers_and_acceleration_structures_per_shader_stage,
        70
    );
    assert_eq!(limits.bind_groups, 2);
}

#[test]
fn compute_stage_specs_are_unique_and_reference_their_entry_points() {
    let mut ids = HashSet::new();
    let mut labels = HashSet::new();
    let mut entry_points = HashSet::new();

    for stage in COMPUTE_STAGES {
        assert!(ids.insert(stage.id), "duplicate stage ID: {:?}", stage.id);
        assert!(
            labels.insert(stage.label),
            "duplicate stage label: {}",
            stage.label
        );
        assert!(
            entry_points.insert(stage.entry_point),
            "duplicate stage entry point: {}",
            stage.entry_point
        );

        for noise_enabled in [false, true] {
            let source = compose_source_with_noise(stage.source, noise_enabled);
            assert!(
                source.contains(&format!("fn {}(", stage.entry_point)),
                "stage {} does not declare entry point {}",
                stage.label,
                stage.entry_point
            );

            let used_bindings = resource_bindings(&source);
            for binding in &used_bindings {
                assert!(
                    canonical_wavefront_bindings()
                        .iter()
                        .any(|spec| (spec.group, spec.binding) == *binding),
                    "stage {} uses undeclared binding {binding:?}",
                    stage.label
                );
            }
        }
    }

    assert!(!COMPUTE_STAGES.is_empty());
}
