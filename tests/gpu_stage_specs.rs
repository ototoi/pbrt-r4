use pbrt_r4::gpu::webgpu::stages::{
    all_stage_specs, canonical_wavefront_bindings, initial_stage_specs, Access, BindingClass,
    BindingSpec, RequiredLimits, ResourceId, StageId, StageSpec,
};

#[test]
fn initial_stage_specs_request_their_actual_storage_dependencies() {
    let specs = initial_stage_specs();
    let limits = RequiredLimits::from_stages(&specs).unwrap();
    assert_eq!(limits.storage_buffers_per_shader_stage, 14);
    assert_eq!(limits.uniform_buffers_per_shader_stage, 2);
    assert_eq!(limits.bind_groups, 3);
}

#[test]
fn all_stage_specs_cover_the_wavefront_registry() {
    let specs = all_stage_specs();
    assert_eq!(specs.len(), 18);
    let limits = RequiredLimits::from_stages(&specs).unwrap();
    assert_eq!(limits.storage_buffers_per_shader_stage, 14);
    assert_eq!(limits.uniform_buffers_per_shader_stage, 2);
}

#[test]
fn portal_distribution_resources_are_scoped_to_sampling_and_escaped_stages() {
    let specs = all_stage_specs();
    let direct = specs
        .iter()
        .find(|spec| spec.id == StageId::SamplePortalDirect)
        .unwrap();
    let escaped = specs
        .iter()
        .find(|spec| spec.id == StageId::HandleEscaped)
        .unwrap();
    for (resource, direct_binding, escaped_binding) in [
        (ResourceId::PortalInfiniteLight, 2, 6),
        (ResourceId::PortalDistribution, 3, 7),
    ] {
        assert!(direct.bindings.iter().any(|binding| {
            binding.group == 1 && binding.binding == direct_binding && binding.resource == resource
        }));
        assert!(escaped.bindings.iter().any(|binding| {
            binding.group == 1 && binding.binding == escaped_binding && binding.resource == resource
        }));
        assert!(specs
            .iter()
            .filter(|spec| !matches!(
                spec.id,
                StageId::SamplePortalDirect | StageId::HandleEscaped
            ))
            .all(|spec| spec
                .bindings
                .iter()
                .all(|binding| binding.resource != resource)));
    }
}

#[test]
fn duplicate_bindings_are_rejected_before_device_creation() {
    let bindings = Box::leak(Box::new([
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
    ]));
    let result = RequiredLimits::from_stages(&[StageSpec {
        id: StageId::BeginSample,
        entry_point: "begin_sample",
        bindings,
    }]);
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
