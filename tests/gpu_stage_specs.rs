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
    assert_eq!(bindings.len(), 58);
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
}

#[test]
fn canonical_layout_drives_required_limits() {
    let limits = RequiredLimits::from_bindings(&canonical_wavefront_bindings()).unwrap();
    assert_eq!(limits.storage_buffers_per_shader_stage, 47);
    assert_eq!(limits.uniform_buffers_per_shader_stage, 6);
    assert_eq!(
        limits.buffers_and_acceleration_structures_per_shader_stage,
        54
    );
    assert_eq!(limits.bind_groups, 2);
}
