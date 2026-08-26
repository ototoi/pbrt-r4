use super::super::fragment::{Fragment, FragmentId};
use super::super::{ShaderStage, ShaderStageId};
use super::ShaderRecipe;

pub fn build_wavefront() -> ShaderRecipe {
    ShaderRecipe {
        label: "pbrt-r4 WebGPU wavefront shader",
        fragments: vec![
            Fragment {
                id: FragmentId::Abi,
                path: "shaders/common/abi.wgsl",
                source: include_str!("../../shaders/common/abi.wgsl"),
                dependencies: Vec::new(),
            },
            Fragment {
                id: FragmentId::Transform,
                path: "shaders/common/transform.wgsl",
                source: include_str!("../../shaders/common/transform.wgsl"),
                dependencies: vec![FragmentId::Abi],
            },
            Fragment {
                id: FragmentId::Sampling,
                path: "shaders/common/sampling.wgsl",
                source: include_str!("../../shaders/common/sampling.wgsl"),
                dependencies: vec![FragmentId::Abi],
            },
            Fragment {
                id: FragmentId::Texture,
                path: "shaders/common/texture.wgsl",
                source: include_str!("../../shaders/common/texture.wgsl"),
                dependencies: vec![FragmentId::Abi],
            },
            Fragment {
                id: FragmentId::AreaGeometry,
                path: "shaders/common/area_geometry.wgsl",
                source: include_str!("../../shaders/common/area_geometry.wgsl"),
                dependencies: vec![FragmentId::Transform, FragmentId::WavefrontAbi],
            },
            Fragment {
                id: FragmentId::WavefrontAbi,
                path: "shaders/wavefront/abi.wgsl",
                source: include_str!("../../shaders/wavefront/abi.wgsl"),
                dependencies: vec![FragmentId::Abi],
            },
            Fragment {
                id: FragmentId::WavefrontCamera,
                path: "shaders/wavefront/camera.wgsl",
                source: include_str!("../../shaders/wavefront/camera.wgsl"),
                dependencies: vec![
                    FragmentId::Transform,
                    FragmentId::Sampling,
                    FragmentId::WavefrontAbi,
                ],
            },
            Fragment {
                id: FragmentId::WavefrontIntersection,
                path: "shaders/wavefront/intersection.wgsl",
                source: include_str!("../../shaders/wavefront/intersection.wgsl"),
                dependencies: vec![FragmentId::WavefrontAbi],
            },
            Fragment {
                id: FragmentId::WavefrontEscaped,
                path: "shaders/wavefront/escaped.wgsl",
                source: include_str!("../../shaders/wavefront/escaped.wgsl"),
                dependencies: vec![FragmentId::WavefrontAbi],
            },
            Fragment {
                id: FragmentId::WavefrontSurface,
                path: "shaders/wavefront/surface.wgsl",
                source: include_str!("../../shaders/wavefront/surface.wgsl"),
                dependencies: vec![FragmentId::WavefrontAbi],
            },
            Fragment {
                id: FragmentId::WavefrontEmissive,
                path: "shaders/wavefront/emissive.wgsl",
                source: include_str!("../../shaders/wavefront/emissive.wgsl"),
                dependencies: vec![FragmentId::AreaGeometry, FragmentId::WavefrontAbi],
            },
            Fragment {
                id: FragmentId::WavefrontMaterial,
                path: "shaders/wavefront/material.wgsl",
                source: include_str!("../../shaders/wavefront/material.wgsl"),
                dependencies: vec![FragmentId::Texture, FragmentId::WavefrontAbi],
            },
            Fragment {
                id: FragmentId::WavefrontDirectLighting,
                path: "shaders/wavefront/direct_lighting.wgsl",
                source: include_str!("../../shaders/wavefront/direct_lighting.wgsl"),
                dependencies: vec![
                    FragmentId::Sampling,
                    FragmentId::AreaGeometry,
                    FragmentId::WavefrontAbi,
                ],
            },
            Fragment {
                id: FragmentId::WavefrontShadow,
                path: "shaders/wavefront/shadow.wgsl",
                source: include_str!("../../shaders/wavefront/shadow.wgsl"),
                dependencies: vec![FragmentId::WavefrontIntersection],
            },
            Fragment {
                id: FragmentId::WavefrontIndirect,
                path: "shaders/wavefront/indirect.wgsl",
                source: include_str!("../../shaders/wavefront/indirect.wgsl"),
                dependencies: vec![FragmentId::Sampling, FragmentId::WavefrontAbi],
            },
            Fragment {
                id: FragmentId::WavefrontFilm,
                path: "shaders/wavefront/film.wgsl",
                source: include_str!("../../shaders/wavefront/film.wgsl"),
                dependencies: vec![FragmentId::WavefrontAbi],
            },
        ],
        roots: vec![
            FragmentId::WavefrontCamera,
            FragmentId::WavefrontIntersection,
            FragmentId::WavefrontEscaped,
            FragmentId::WavefrontSurface,
            FragmentId::WavefrontEmissive,
            FragmentId::WavefrontMaterial,
            FragmentId::WavefrontDirectLighting,
            FragmentId::WavefrontShadow,
            FragmentId::WavefrontIndirect,
            FragmentId::WavefrontFilm,
        ],
        stages: vec![
            ShaderStage {
                id: ShaderStageId::PrepareCameraRays,
                entry_point: "prepare_camera_rays",
            },
            ShaderStage {
                id: ShaderStageId::GenerateCameraRays,
                entry_point: "generate_camera_rays",
            },
            ShaderStage {
                id: ShaderStageId::IntersectClosest,
                entry_point: "intersect_closest",
            },
            ShaderStage {
                id: ShaderStageId::HandleEscapedRays,
                entry_point: "handle_escaped_rays",
            },
            ShaderStage {
                id: ShaderStageId::EvaluateSurfaceInteraction,
                entry_point: "evaluate_surface_interaction",
            },
            ShaderStage {
                id: ShaderStageId::HandleEmissiveIntersection,
                entry_point: "handle_emissive_intersection",
            },
            ShaderStage {
                id: ShaderStageId::EvaluateMaterial,
                entry_point: "evaluate_material",
            },
            ShaderStage {
                id: ShaderStageId::SampleDirectLighting,
                entry_point: "sample_direct_lighting",
            },
            ShaderStage {
                id: ShaderStageId::IntersectShadow,
                entry_point: "intersect_shadow",
            },
            ShaderStage {
                id: ShaderStageId::SampleIndirectBxdf,
                entry_point: "sample_indirect_bxdf",
            },
            ShaderStage {
                id: ShaderStageId::UpdateFilm,
                entry_point: "update_film",
            },
        ],
    }
}
