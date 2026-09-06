use pbrt_r4::gpu::webgpu::stages::{
    initial_stage_specs, Access, BindingClass, BindingSpec, RequiredLimits, ResourceId, StageId,
    StageSpec,
};

#[test]
fn initial_stage_specs_request_their_actual_storage_dependencies() {
    let specs = initial_stage_specs();
    let limits = RequiredLimits::from_stages(&specs).unwrap();
    assert_eq!(limits.storage_buffers_per_shader_stage, 15);
    assert_eq!(limits.uniform_buffers_per_shader_stage, 2);
    assert_eq!(limits.bind_groups, 3);
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
