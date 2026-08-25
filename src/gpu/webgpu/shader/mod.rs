mod fragment;

use super::device::AccelerationMode;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderSet {
    pub source: String,
    pub label: &'static str,
    pub entry_point: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShaderBuildError {
    Fragment(String),
}

impl std::fmt::Display for ShaderBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fragment(error) => {
                write!(formatter, "shader fragment composition failed: {error}")
            }
        }
    }
}

impl std::error::Error for ShaderBuildError {}

pub fn build_shader_set(mode: AccelerationMode) -> Result<ShaderSet, ShaderBuildError> {
    let (traversal, label) = match mode {
        AccelerationMode::HardwareRayQuery => (
            fragment::FragmentId::RayQueryTraversal,
            "pbrt-r4 WebGPU hardware ray query shader",
        ),
        AccelerationMode::SoftwareBvh => (
            fragment::FragmentId::SoftwareBvhTraversal,
            "pbrt-r4 WebGPU software BVH shader",
        ),
    };
    let common_ids = vec![
        fragment::FragmentId::Abi,
        fragment::FragmentId::Transform,
        fragment::FragmentId::Texture,
        fragment::FragmentId::Geometry,
        fragment::FragmentId::Sampling,
        fragment::FragmentId::AreaGeometry,
        fragment::FragmentId::AreaSampling,
        fragment::FragmentId::Emission,
    ];
    let fragments = [
        fragment::Fragment {
            id: fragment::FragmentId::Abi,
            path: "shaders/common/abi.wgsl",
            source: include_str!("../shaders/common/abi.wgsl"),
            dependencies: Vec::new(),
        },
        fragment::Fragment {
            id: fragment::FragmentId::Transform,
            path: "shaders/common/transform.wgsl",
            source: include_str!("../shaders/common/transform.wgsl"),
            dependencies: vec![fragment::FragmentId::Abi],
        },
        fragment::Fragment {
            id: fragment::FragmentId::Texture,
            path: "shaders/common/texture.wgsl",
            source: include_str!("../shaders/common/texture.wgsl"),
            dependencies: vec![fragment::FragmentId::Transform],
        },
        fragment::Fragment {
            id: fragment::FragmentId::Geometry,
            path: "shaders/common/geometry.wgsl",
            source: include_str!("../shaders/common/geometry.wgsl"),
            dependencies: vec![fragment::FragmentId::Texture],
        },
        fragment::Fragment {
            id: fragment::FragmentId::Sampling,
            path: "shaders/common/sampling.wgsl",
            source: include_str!("../shaders/common/sampling.wgsl"),
            dependencies: vec![fragment::FragmentId::Abi],
        },
        fragment::Fragment {
            id: fragment::FragmentId::AreaGeometry,
            path: "shaders/common/area_geometry.wgsl",
            source: include_str!("../shaders/common/area_geometry.wgsl"),
            dependencies: vec![fragment::FragmentId::Geometry],
        },
        fragment::Fragment {
            id: fragment::FragmentId::AreaSampling,
            path: "shaders/common/area_sampling.wgsl",
            source: include_str!("../shaders/common/area_sampling.wgsl"),
            dependencies: vec![fragment::FragmentId::AreaGeometry],
        },
        fragment::Fragment {
            id: fragment::FragmentId::Emission,
            path: "shaders/common/emission.wgsl",
            source: include_str!("../shaders/common/emission.wgsl"),
            dependencies: vec![fragment::FragmentId::AreaSampling],
        },
        fragment::Fragment {
            id: fragment::FragmentId::HardwareBindings,
            path: "shaders/intersection/ray_query.wgsl:bindings",
            source: "@group(0) @binding(2)\nvar acceleration: acceleration_structure;\n",
            dependencies: common_ids.clone(),
        },
        fragment::Fragment {
            id: fragment::FragmentId::RayQueryTraversal,
            path: "shaders/intersection/ray_query.wgsl",
            source: include_str!("../shaders/intersection/ray_query.wgsl"),
            dependencies: vec![fragment::FragmentId::HardwareBindings],
        },
        fragment::Fragment {
            id: fragment::FragmentId::SoftwareBvhTraversal,
            path: "shaders/intersection/software_bvh.wgsl",
            source: include_str!("../shaders/intersection/software_bvh.wgsl"),
            dependencies: common_ids,
        },
        fragment::Fragment {
            id: fragment::FragmentId::EntryMain,
            path: "shaders/entry/main.wgsl",
            source: include_str!("../shaders/entry/main.wgsl"),
            dependencies: vec![traversal],
        },
    ];
    let composed = fragment::compose(&fragments, &[fragment::FragmentId::EntryMain])
        .map_err(|error| ShaderBuildError::Fragment(format!("{error:?}")))?;
    let source = match mode {
        AccelerationMode::HardwareRayQuery => format!("enable wgpu_ray_query;\n\n{composed}"),
        AccelerationMode::SoftwareBvh => composed,
    };
    Ok(ShaderSet {
        source,
        label,
        entry_point: "main",
    })
}
