#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::webgpu::binding_report::{BindingHeadroom, BindingReport};
use pbrt_r4::gpu::webgpu::bssrdf::probe_shader_source;
use pbrt_r4::gpu::webgpu::context::validate_binding_limits;
use pbrt_r4::gpu::webgpu::shader::{
    compose_source_with_features, required_limits_for_sources_with_nanovdb,
    resolve_resource_bindings, specialize_source,
};
use pbrt_r4::gpu::webgpu::stage::{ComputeStageId, ComputeStageSpec, COMPUTE_STAGES};
use pbrt_r4::gpu::webgpu::stages::{
    canonical_wavefront_bindings, Access, BindingClass, BindingSpec, RequiredLimits, ResourceId,
};

fn mac_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffers_per_shader_stage: 29,
        max_uniform_buffers_per_shader_stage: 29,
        max_buffers_and_acceleration_structures_per_shader_stage: 29,
        ..wgpu::Limits::default()
    }
}

fn naga_bindings(source: &str) -> Vec<(u32, u32)> {
    let module = wgpu::naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    let mut bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, global)| global.binding.as_ref().map(|b| (b.group, b.binding)))
        .collect::<Vec<_>>();
    bindings.sort_unstable();
    bindings
}

#[test]
fn reports_match_pipeline_declarations_for_all_feature_combinations() {
    let registry = canonical_wavefront_bindings();
    for noise in [false, true] {
        for nanovdb in [false, true] {
            let report =
                BindingReport::for_stages(&registry, COMPUTE_STAGES, noise, nanovdb).unwrap();
            assert_eq!(report.stages.len(), COMPUTE_STAGES.len());
            let mut maximum = RequiredLimits::default();
            for (stage, entry) in COMPUTE_STAGES.iter().zip(&report.stages) {
                assert_eq!(entry.stage_id, stage.id);
                assert_eq!(entry.entry_point, stage.entry_point);
                let source = compose_source_with_features(stage.source, noise, nanovdb);
                let pairs = entry
                    .bindings
                    .iter()
                    .map(|b| (b.group, b.binding))
                    .collect::<Vec<_>>();
                // Use the WGSL parser as an independent check on declaration extraction.
                assert_eq!(pairs, naga_bindings(&source), "{}", stage.entry_point);
                let specialized = specialize_source(source, 3, 2, 17);
                assert_eq!(pairs, naga_bindings(&specialized));
                assert_eq!(
                    entry.bindings,
                    resolve_resource_bindings(&registry, stage.entry_point, &specialized).unwrap()
                );
                assert_eq!(
                    entry.required,
                    RequiredLimits::from_bindings(&entry.bindings).unwrap()
                );
                maximum = maximum.maximum(entry.required);
            }
            assert_eq!(report.required, maximum);
            assert_eq!(
                report.device_required().bind_groups,
                report.required.bind_groups.max(2)
            );
            if noise {
                let sources = COMPUTE_STAGES
                    .iter()
                    .map(|stage| stage.source)
                    .collect::<Vec<_>>();
                assert_eq!(
                    report.required,
                    required_limits_for_sources_with_nanovdb(&registry, &sources, nanovdb).unwrap()
                );
            }
        }
    }
}

#[test]
fn over_budget_stages_are_explained_without_a_gpu() {
    let report =
        BindingReport::for_stages(&canonical_wavefront_bindings(), COMPUTE_STAGES, true, false)
            .unwrap();
    let medium = report
        .stages
        .iter()
        .find(|stage| stage.stage_id == ComputeStageId::ScatterMedium)
        .unwrap();
    let direct = report
        .stages
        .iter()
        .find(|stage| stage.stage_id == ComputeStageId::SampleDirectLight)
        .unwrap();
    assert_eq!(medium.required.storage_buffers_per_shader_stage, 30);
    assert_eq!(medium.required.uniform_buffers_per_shader_stage, 4);
    assert_eq!(
        medium
            .required
            .buffers_and_acceleration_structures_per_shader_stage,
        34
    );
    assert_eq!(
        direct
            .required
            .buffers_and_acceleration_structures_per_shader_stage,
        32
    );
    assert_eq!(
        BindingHeadroom::new(medium.required, &mac_limits()).combined,
        -5
    );
    let text = report.format(Some(&mac_limits()), true);
    assert!(text.contains("scope=wavefront, noise=true, nanovdb=false"));
    assert!(text.contains("stage=scatter_medium storage=30 uniform=4 as=0 combined=34"));
    assert!(text.contains("stage=sample_direct_light storage=30 uniform=2 as=0 combined=32"));
    assert!(text.contains("headroom: storage=-1 uniform=25 combined=-5"));
    assert!(!text.contains("resource=MaterialRoot class=Storage"));
    assert!(!text.contains("resource=MaterialNode class=Storage"));
    assert!(text.contains("resource=TextureImageArray class=SampledTexture"));
    assert!(!text.contains("stage=prepare_sample "));
    println!("{text}");
}

#[test]
fn maxima_are_independent_and_layout_groups_are_not_shader_resources() {
    let storage_heavy = RequiredLimits {
        storage_buffers_per_shader_stage: 8,
        uniform_buffers_per_shader_stage: 1,
        buffers_and_acceleration_structures_per_shader_stage: 9,
        bind_groups: 1,
    };
    let uniform_heavy = RequiredLimits {
        storage_buffers_per_shader_stage: 1,
        uniform_buffers_per_shader_stage: 7,
        buffers_and_acceleration_structures_per_shader_stage: 8,
        bind_groups: 2,
    };
    let maximum = storage_heavy.maximum(uniform_heavy);
    assert_eq!(maximum.storage_buffers_per_shader_stage, 8);
    assert_eq!(maximum.uniform_buffers_per_shader_stage, 7);
    assert_eq!(
        maximum.buffers_and_acceleration_structures_per_shader_stage,
        9
    );
    assert_eq!(maximum.bind_groups, 2);
    let report = BindingReport::for_stages(
        &canonical_wavefront_bindings(),
        &COMPUTE_STAGES[2..3],
        false,
        false,
    )
    .unwrap();
    assert_eq!(report.required.bind_groups, 1);
    assert_eq!(report.device_required().bind_groups, 2);
    assert!(report.stages[0].bindings.iter().all(|b| b.group == 0));
    let limits = wgpu::Limits {
        max_bind_groups: 1,
        ..mac_limits()
    };
    assert!(!BindingHeadroom::new(report.device_required(), &limits).fits());
    assert!(report
        .format(Some(&limits), true)
        .contains("bind_groups=-1"));
}

#[test]
fn acceleration_structures_count_toward_combined_but_textures_do_not() {
    let bindings = [
        (BindingClass::Storage, ResourceId::Vertex),
        (BindingClass::Uniform, ResourceId::CameraParams),
        (BindingClass::AccelerationStructure, ResourceId::Tlas),
        (BindingClass::SampledTexture, ResourceId::TextureImageArray),
        (BindingClass::IntegerTexture, ResourceId::NoiseTable),
        (BindingClass::Sampler, ResourceId::TextureSamplerArray),
    ]
    .into_iter()
    .enumerate()
    .map(|(binding, (class, resource))| BindingSpec {
        group: u32::from(binding >= 3),
        binding: binding as u32,
        class,
        resource,
        access: Access::Read,
    })
    .collect::<Vec<_>>();
    let required = RequiredLimits::from_bindings(&bindings).unwrap();
    assert_eq!(required.storage_buffers_per_shader_stage, 1);
    assert_eq!(required.uniform_buffers_per_shader_stage, 1);
    assert_eq!(
        required.buffers_and_acceleration_structures_per_shader_stage,
        3
    );
    assert_eq!(required.bind_groups, 2);
}

#[test]
fn limit_boundaries_and_signed_headroom_are_exact() {
    for (combined, fits, headroom) in [(29, true, 0), (30, false, -1), (36, false, -7)] {
        let required = RequiredLimits {
            buffers_and_acceleration_structures_per_shader_stage: combined,
            ..RequiredLimits::default()
        };
        let result = BindingHeadroom::new(required, &mac_limits());
        assert_eq!(result.fits(), fits);
        assert_eq!(result.combined, headroom);
    }
    for required in [
        RequiredLimits {
            storage_buffers_per_shader_stage: 30,
            ..RequiredLimits::default()
        },
        RequiredLimits {
            uniform_buffers_per_shader_stage: 30,
            ..RequiredLimits::default()
        },
        RequiredLimits {
            bind_groups: u32::MAX,
            ..RequiredLimits::default()
        },
    ] {
        assert!(!BindingHeadroom::new(required, &mac_limits()).fits());
    }
    let required = RequiredLimits {
        storage_buffers_per_shader_stage: u32::MAX,
        ..RequiredLimits::default()
    };
    assert_eq!(
        BindingHeadroom::new(required, &mac_limits()).storage,
        29 - i64::from(u32::MAX)
    );
}

#[test]
fn unknown_and_duplicate_registry_bindings_include_stage_context() {
    let registry = canonical_wavefront_bindings();
    let source = "@group(1) @binding(999) var<storage, read> unknown: array<u32>;";
    let error = resolve_resource_bindings(&registry, "test_unknown", source)
        .unwrap_err()
        .to_string();
    assert!(error.contains("test_unknown uses unregistered group 1 binding 999"));
    let mut duplicate = registry.clone();
    duplicate.push(registry[0]);
    let error = BindingReport::for_stages(&duplicate, &COMPUTE_STAGES[..1], false, false)
        .unwrap_err()
        .to_string();
    assert!(error.contains(COMPUTE_STAGES[0].entry_point));
    assert!(error.contains("Duplicate binding group 0 binding 0"));
}

#[test]
fn output_order_is_stable_and_missing_adapter_limits_are_explicit() {
    let stages = [
        ComputeStageSpec {
            id: ComputeStageId::PrepareSample,
            label: "z",
            entry_point: "z",
            source: "@compute @workgroup_size(1) fn z() {}",
        },
        ComputeStageSpec {
            id: ComputeStageId::ResetShadowQueue,
            label: "a",
            entry_point: "a",
            source: "@compute @workgroup_size(1) fn a() {}",
        },
    ];
    let report =
        BindingReport::for_stages(&canonical_wavefront_bindings(), &stages, false, false).unwrap();
    let text = report.format(None, false);
    assert!(text.contains("limits: unavailable (no adapter selected)"));
    assert!(text.find("stage=a ").unwrap() < text.find("stage=z ").unwrap());
    assert_eq!(report.stages[0].entry_point, "z");
    assert_eq!(text, report.format(None, false));
}

#[test]
fn separate_probe_pipeline_remains_below_wavefront_requirements() {
    // Its auto layout is derived from this WGSL, not the wavefront registry.
    let source = probe_shader_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
    let mut storage = 0;
    let mut acceleration = 0;
    for (_, global) in module
        .global_variables
        .iter()
        .filter(|(_, global)| global.binding.is_some())
    {
        if matches!(global.space, wgpu::naga::AddressSpace::Storage { .. }) {
            storage += 1;
        }
        if matches!(
            module.types[global.ty].inner,
            wgpu::naga::TypeInner::AccelerationStructure { .. }
        ) {
            acceleration += 1;
        }
        assert_eq!(global.binding.as_ref().unwrap().group, 0);
    }
    assert_eq!((storage, acceleration), (9, 1));
    for noise in [false, true] {
        for nanovdb in [false, true] {
            let required = BindingReport::for_stages(
                &canonical_wavefront_bindings(),
                COMPUTE_STAGES,
                noise,
                nanovdb,
            )
            .unwrap()
            .device_required();
            assert!(required.storage_buffers_per_shader_stage >= storage);
            assert!(
                required.buffers_and_acceleration_structures_per_shader_stage
                    >= storage + acceleration
            );
            assert!(required.bind_groups >= 1);
        }
    }
}

#[test]
fn limit_failure_explains_all_over_budget_stages_without_debug_logging() {
    let report =
        BindingReport::for_stages(&canonical_wavefront_bindings(), COMPUTE_STAGES, true, false)
            .unwrap();
    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 29,
        max_uniform_buffers_per_shader_stage: 29,
        max_buffers_and_acceleration_structures_per_shader_stage: 29,
        ..wgpu::Limits::default()
    };
    let error = validate_binding_limits(report.device_required(), &limits, Some(&report))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("requested storage=30, uniform=4, buffers_and_acceleration_structures=34")
    );
    assert!(error.contains("stage=scatter_medium"));
    assert!(error.contains("stage=sample_direct_light"));
    assert!(error.contains("resource=Instance"));
    assert!(error.contains("combined=-5"));
    assert!(!error.contains("stage=prepare_sample "));
}

#[test]
fn legacy_context_limit_check_preserves_the_summary_without_a_report() {
    let required = RequiredLimits {
        storage_buffers_per_shader_stage: 30,
        ..RequiredLimits::default()
    };
    let limits = wgpu::Limits {
        max_storage_buffers_per_shader_stage: 29,
        ..wgpu::Limits::default()
    };
    let error = validate_binding_limits(required, &limits, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("requested storage=30"));
    assert!(error.contains("available storage=29"));
    assert!(!error.contains("stage="));
    assert!(validate_binding_limits(RequiredLimits::default(), &limits, None).is_ok());
}
