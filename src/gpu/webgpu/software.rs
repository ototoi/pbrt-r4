use super::device::DeviceContext;
use super::error::{BackendError, PlanError};
use super::geometry::{
    index_bytes, light_bytes, material_bytes, primitive_bytes, texture_bytes, transform_bytes,
    vertex_bytes, ScenePlan,
};
use glam::Vec3;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

const LEAF_MAX_PRIMITIVES: usize = 4;
const MAX_BVH_STACK_ENTRIES: u32 = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct BvhNodePlan {
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub first: u32,
    pub count: u32,
    pub flags: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BvhPrimitivePlan {
    pub primitive: u32,
    pub triangle: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SoftwareBvhPlan {
    pub nodes: Vec<BvhNodePlan>,
    pub primitives: Vec<BvhPrimitivePlan>,
}

#[derive(Clone, Debug)]
struct TriangleInfo {
    primitive: u32,
    triangle: u32,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    centroid: [f32; 3],
}

impl SoftwareBvhPlan {
    pub fn from_scene(plan: &ScenePlan) -> Result<Self, BackendError> {
        let mut triangles = Vec::new();
        for (primitive_index, primitive) in plan.primitives.iter().enumerate() {
            let transform = plan
                .transforms
                .get(primitive_index)
                .ok_or(BackendError::Plan(PlanError::InvalidReference {
                    resource: "transform table",
                    index: primitive_index as u32,
                }))?;
            let triangle_count = usize::try_from(primitive.triangle_count).map_err(|_| {
                BackendError::Plan(PlanError::LimitExceeded {
                    resource: "triangle count",
                    value: u32::MAX,
                    maximum: u32::MAX,
                })
            })?;
            for triangle in 0..triangle_count {
                let index_offset = primitive.first_index as usize + triangle * 3;
                if index_offset + 2 >= plan.indices.len() {
                    return Err(BackendError::Plan(PlanError::InvalidReference {
                        resource: "index buffer",
                        index: index_offset as u32,
                    }));
                }
                let vertex_indices = [
                    plan.indices[index_offset] as usize + primitive.first_vertex as usize,
                    plan.indices[index_offset + 1] as usize + primitive.first_vertex as usize,
                    plan.indices[index_offset + 2] as usize + primitive.first_vertex as usize,
                ];
                let positions = vertex_indices
                    .into_iter()
                    .map(|index| {
                        plan.vertices.get(index).copied().ok_or(BackendError::Plan(
                            PlanError::InvalidReference {
                                resource: "vertex buffer",
                                index: index as u32,
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let positions = positions
                    .into_iter()
                    .map(|vertex| transform_point(transform.render_from_object, vertex.position))
                    .collect::<Vec<_>>();
                let mut bounds_min = positions[0];
                let mut bounds_max = positions[0];
                for position in &positions[1..] {
                    for axis in 0..3 {
                        bounds_min[axis] = bounds_min[axis].min(position[axis]);
                        bounds_max[axis] = bounds_max[axis].max(position[axis]);
                    }
                }
                triangles.push(TriangleInfo {
                    primitive: primitive_index as u32,
                    triangle: triangle as u32,
                    bounds_min,
                    bounds_max,
                    centroid: Vec3::from_array(bounds_min)
                        .lerp(Vec3::from_array(bounds_max), 0.5)
                        .to_array(),
                });
            }
        }
        if triangles.is_empty() {
            return Err(BackendError::Plan(PlanError::EmptyScene));
        }

        let mut nodes = Vec::new();
        let mut primitives = Vec::with_capacity(triangles.len());
        build_tree(&mut triangles, &mut nodes, &mut primitives)?;
        Ok(Self { nodes, primitives })
    }
}

fn build_tree(
    triangles: &mut [TriangleInfo],
    nodes: &mut Vec<BvhNodePlan>,
    primitives: &mut Vec<BvhPrimitivePlan>,
) -> Result<(), BackendError> {
    nodes.push(BvhNodePlan {
        bounds_min: [f32::INFINITY; 3],
        bounds_max: [f32::NEG_INFINITY; 3],
        first: 0,
        count: 0,
        flags: 0,
    });
    let mut work = vec![(0usize, triangles.len(), 0usize, 0u32)];
    while let Some((start, end, node_index, depth)) = work.pop() {
        let subset = &mut triangles[start..end];
        let (bounds_min, bounds_max) = bounds(subset);
        if subset.len() <= LEAF_MAX_PRIMITIVES {
            let first = primitives.len();
            primitives.extend(subset.iter().map(|triangle| BvhPrimitivePlan {
                primitive: triangle.primitive,
                triangle: triangle.triangle,
            }));
            nodes[node_index] = BvhNodePlan {
                bounds_min,
                bounds_max,
                first: first as u32,
                count: subset.len() as u32,
                flags: 1,
            };
            continue;
        }

        if depth + 2 >= MAX_BVH_STACK_ENTRIES {
            return Err(BackendError::Plan(PlanError::LimitExceeded {
                resource: "bvh_stack_depth",
                value: depth + 2,
                maximum: MAX_BVH_STACK_ENTRIES,
            }));
        }

        let extent = [
            bounds_max[0] - bounds_min[0],
            bounds_max[1] - bounds_min[1],
            bounds_max[2] - bounds_min[2],
        ];
        let axis = (0..3)
            .max_by(|left, right| extent[*left].total_cmp(&extent[*right]))
            .unwrap_or(0);
        subset.sort_by(|left, right| left.centroid[axis].total_cmp(&right.centroid[axis]));
        let middle = start + subset.len() / 2;
        let left_index = nodes.len();
        nodes.push(BvhNodePlan {
            bounds_min: [f32::INFINITY; 3],
            bounds_max: [f32::NEG_INFINITY; 3],
            first: 0,
            count: 0,
            flags: 0,
        });
        nodes.push(BvhNodePlan {
            bounds_min: [f32::INFINITY; 3],
            bounds_max: [f32::NEG_INFINITY; 3],
            first: 0,
            count: 0,
            flags: 0,
        });
        nodes[node_index] = BvhNodePlan {
            bounds_min,
            bounds_max,
            first: left_index as u32,
            count: 2,
            flags: 0,
        };
        work.push((middle, end, left_index + 1, depth + 1));
        work.push((start, middle, left_index, depth + 1));
    }
    Ok(())
}

fn bounds(triangles: &[TriangleInfo]) -> ([f32; 3], [f32; 3]) {
    let mut bounds_min = [f32::INFINITY; 3];
    let mut bounds_max = [f32::NEG_INFINITY; 3];
    for triangle in triangles {
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(triangle.bounds_min[axis]);
            bounds_max[axis] = bounds_max[axis].max(triangle.bounds_max[axis]);
        }
    }
    (bounds_min, bounds_max)
}

fn transform_point(matrix: [[f32; 4]; 4], point: [f32; 3]) -> [f32; 3] {
    let value = [point[0], point[1], point[2], 1.0];
    let result = [
        matrix[0][0] * value[0]
            + matrix[0][1] * value[1]
            + matrix[0][2] * value[2]
            + matrix[0][3] * value[3],
        matrix[1][0] * value[0]
            + matrix[1][1] * value[1]
            + matrix[1][2] * value[2]
            + matrix[1][3] * value[3],
        matrix[2][0] * value[0]
            + matrix[2][1] * value[1]
            + matrix[2][2] * value[2]
            + matrix[2][3] * value[3],
    ];
    let w = matrix[3][0] * value[0]
        + matrix[3][1] * value[1]
        + matrix[3][2] * value[2]
        + matrix[3][3] * value[3];
    if w != 0.0 && w != 1.0 {
        [result[0] / w, result[1] / w, result[2] / w]
    } else {
        result
    }
}

fn node_bytes(plan: &SoftwareBvhPlan) -> Vec<u8> {
    plan.nodes
        .iter()
        .flat_map(|node| {
            [
                [
                    node.bounds_min[0],
                    node.bounds_min[1],
                    node.bounds_min[2],
                    0.0,
                ],
                [
                    node.bounds_max[0],
                    node.bounds_max[1],
                    node.bounds_max[2],
                    0.0,
                ],
            ]
            .into_iter()
            .flatten()
            .flat_map(f32::to_ne_bytes)
            .chain(
                [node.first, node.count, node.flags, 0]
                    .into_iter()
                    .flat_map(u32::to_ne_bytes),
            )
        })
        .collect()
}

fn primitive_ref_bytes(plan: &SoftwareBvhPlan) -> Vec<u8> {
    plan.primitives
        .iter()
        .flat_map(|primitive| {
            [primitive.primitive, primitive.triangle]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
        })
        .collect()
}

pub struct SoftwareAcceleration {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub primitive_buffer: wgpu::Buffer,
    pub material_buffer: wgpu::Buffer,
    pub transform_buffer: wgpu::Buffer,
    pub light_buffer: wgpu::Buffer,
    pub bvh_buffer: wgpu::Buffer,
    pub bvh_primitive_offset: u32,
    pub bvh_node_offset: u32,
}

impl SoftwareAcceleration {
    pub fn create(context: &DeviceContext, plan: &ScenePlan) -> Result<Self, BackendError> {
        let bvh = SoftwareBvhPlan::from_scene(plan)?;
        let mut bvh_bytes = texture_bytes(plan);
        let bvh_node_offset = u32::try_from(bvh_bytes.len() / 4).map_err(|_| {
            BackendError::Plan(PlanError::LimitExceeded {
                resource: "software BVH buffer",
                value: u32::MAX,
                maximum: u32::MAX,
            })
        })?;
        bvh_bytes.extend(node_bytes(&bvh));
        let bvh_primitive_offset = u32::try_from(bvh_bytes.len() / 4).map_err(|_| {
            BackendError::Plan(PlanError::LimitExceeded {
                resource: "software BVH buffer",
                value: u32::MAX,
                maximum: u32::MAX,
            })
        })?;
        bvh_bytes.extend(primitive_ref_bytes(&bvh));
        let storage = |label, contents| {
            context.device.create_buffer_init(&BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        Ok(Self {
            vertex_buffer: storage("pbrt-r4 WebGPU software vertex buffer", &vertex_bytes(plan)),
            index_buffer: storage("pbrt-r4 WebGPU software index buffer", &index_bytes(plan)),
            primitive_buffer: storage(
                "pbrt-r4 WebGPU software primitive table",
                &primitive_bytes(plan),
            ),
            material_buffer: storage(
                "pbrt-r4 WebGPU software material table",
                &material_bytes(plan),
            ),
            transform_buffer: storage(
                "pbrt-r4 WebGPU software transform table",
                &transform_bytes(plan),
            ),
            light_buffer: storage("pbrt-r4 WebGPU software light table", &light_bytes(plan)),
            bvh_buffer: storage("pbrt-r4 WebGPU software BVH", &bvh_bytes),
            bvh_primitive_offset,
            bvh_node_offset,
        })
    }
}
