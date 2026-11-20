use std::collections::HashSet;

use crate::gpu::flat::MaterialKind;

use super::super::stage::ComputeStageId;

const SCATTER_STAGES: [ComputeStageId; 8] = [
    ComputeStageId::ScatterDiffuse,
    ComputeStageId::ScatterDiffuseTransmission,
    ComputeStageId::ScatterConductor,
    ComputeStageId::ScatterDielectric,
    ComputeStageId::ScatterThinDielectric,
    ComputeStageId::ScatterMeasured,
    ComputeStageId::ScatterCoated,
    ComputeStageId::ScatterHair,
];

/// Mirrors the routing in classify_surface_scatter.wgsl.
fn scatter_stage(kind: MaterialKind) -> Option<ComputeStageId> {
    match kind {
        MaterialKind::Diffuse => Some(ComputeStageId::ScatterDiffuse),
        MaterialKind::DiffuseTransmission => Some(ComputeStageId::ScatterDiffuseTransmission),
        MaterialKind::ConductorEtaK | MaterialKind::ConductorReflectance => {
            Some(ComputeStageId::ScatterConductor)
        }
        MaterialKind::Dielectric | MaterialKind::Subsurface => {
            Some(ComputeStageId::ScatterDielectric)
        }
        MaterialKind::ThinDielectric => Some(ComputeStageId::ScatterThinDielectric),
        MaterialKind::Measured => Some(ComputeStageId::ScatterMeasured),
        MaterialKind::CoatedDiffuse | MaterialKind::CoatedConductor => {
            Some(ComputeStageId::ScatterCoated)
        }
        MaterialKind::Hair => Some(ComputeStageId::ScatterHair),
        MaterialKind::Mix => None,
    }
}

/// Stages whose queues can never receive work for this scene.
pub(super) fn idle_stages(
    material_kinds: impl IntoIterator<Item = MaterialKind>,
    have_media: bool,
    debug_material: bool,
) -> HashSet<ComputeStageId> {
    if debug_material {
        return HashSet::new();
    }
    let used: HashSet<_> = material_kinds
        .into_iter()
        .filter_map(scatter_stage)
        .collect();
    let mut idle: HashSet<_> = SCATTER_STAGES
        .into_iter()
        .filter(|stage| !used.contains(stage))
        .collect();
    if !have_media {
        idle.insert(ComputeStageId::ScatterMedium);
        idle.insert(ComputeStageId::SelectMediumLight);
    }
    idle
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_KINDS: [MaterialKind; 12] = [
        MaterialKind::Diffuse,
        MaterialKind::Dielectric,
        MaterialKind::ThinDielectric,
        MaterialKind::DiffuseTransmission,
        MaterialKind::ConductorEtaK,
        MaterialKind::ConductorReflectance,
        MaterialKind::Mix,
        MaterialKind::CoatedDiffuse,
        MaterialKind::CoatedConductor,
        MaterialKind::Measured,
        MaterialKind::Subsurface,
        MaterialKind::Hair,
    ];

    fn wgsl_name(kind: MaterialKind) -> &'static str {
        match kind {
            MaterialKind::Diffuse => "MATERIAL_KIND_DIFFUSE",
            MaterialKind::Dielectric => "MATERIAL_KIND_DIELECTRIC",
            MaterialKind::ThinDielectric => "MATERIAL_KIND_THIN_DIELECTRIC",
            MaterialKind::DiffuseTransmission => "MATERIAL_KIND_DIFFUSE_TRANSMISSION",
            MaterialKind::ConductorEtaK => "MATERIAL_KIND_CONDUCTOR_ETA_K",
            MaterialKind::ConductorReflectance => "MATERIAL_KIND_CONDUCTOR_REFLECTANCE",
            MaterialKind::Mix => "MATERIAL_KIND_MIX",
            MaterialKind::CoatedDiffuse => "MATERIAL_KIND_COATED_DIFFUSE",
            MaterialKind::CoatedConductor => "MATERIAL_KIND_COATED_CONDUCTOR",
            MaterialKind::Measured => "MATERIAL_KIND_MEASURED",
            MaterialKind::Subsurface => "MATERIAL_KIND_SUBSURFACE",
            MaterialKind::Hair => "MATERIAL_KIND_HAIR",
        }
    }

    fn append_name(stage: ComputeStageId) -> &'static str {
        match stage {
            ComputeStageId::ScatterDiffuse => "append_scatter_diffuse",
            ComputeStageId::ScatterDiffuseTransmission => "append_scatter_diffuse_transmission",
            ComputeStageId::ScatterConductor => "append_scatter_conductor",
            ComputeStageId::ScatterDielectric => "append_scatter_dielectric",
            ComputeStageId::ScatterThinDielectric => "append_scatter_thin_dielectric",
            ComputeStageId::ScatterMeasured => "append_scatter_measured",
            ComputeStageId::ScatterCoated => "append_scatter_coated",
            ComputeStageId::ScatterHair => "append_scatter_hair",
            other => panic!("not a scatter stage: {other:?}"),
        }
    }

    /// `case A, B: { ... append_scatter_x(...)` pairs from the classifier.
    fn classifier_routes() -> Vec<(String, String)> {
        let source = include_str!("../shaders/classify_surface_scatter.wgsl");
        let mut routes = Vec::new();
        for block in source.split("case ").skip(1) {
            let (labels, body) = block.split_once(':').unwrap();
            let body = body.split("case ").next().unwrap();
            if let Some(start) = body.find("append_scatter_") {
                let call = &body[start..];
                let name = &call[..call.find('(').unwrap()];
                for label in labels.split(',') {
                    routes.push((label.trim().to_string(), name.to_string()));
                }
            }
        }
        routes
    }

    #[test]
    fn mapping_matches_the_wgsl_classifier() {
        let routes = classifier_routes();
        for kind in ALL_KINDS {
            let routed: Vec<_> = routes
                .iter()
                .filter(|(label, _)| label == wgsl_name(kind))
                .map(|(_, append)| append.as_str())
                .collect();
            match scatter_stage(kind) {
                Some(stage) => assert_eq!(routed, vec![append_name(stage)], "{kind:?}"),
                None => assert!(routed.is_empty(), "{kind:?} is routed to {routed:?}"),
            }
        }
    }

    #[test]
    fn unused_material_stages_are_idle() {
        let idle = idle_stages([MaterialKind::Diffuse, MaterialKind::Mix], true, false);
        let expected: HashSet<_> = SCATTER_STAGES
            .into_iter()
            .filter(|stage| *stage != ComputeStageId::ScatterDiffuse)
            .collect();
        assert_eq!(idle, expected);
    }

    #[test]
    fn every_kind_keeps_its_stage() {
        let idle = idle_stages(ALL_KINDS, true, false);
        assert!(idle.is_empty(), "{idle:?}");
    }

    #[test]
    fn media_stages_are_idle_without_media() {
        let idle = idle_stages(ALL_KINDS, false, false);
        assert_eq!(
            idle,
            HashSet::from([
                ComputeStageId::ScatterMedium,
                ComputeStageId::SelectMediumLight
            ])
        );
    }

    #[test]
    fn debug_material_keeps_every_stage() {
        assert!(idle_stages([], false, true).is_empty());
    }
}
