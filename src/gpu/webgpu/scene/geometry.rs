use crate::gpu::flat;
use crate::gpu::flat::SourceShape;
use crate::util::error::PbrtError;

use super::super::abi::{
    Geometry, Vertex, INTERSECTION_NORMAL_KIND_QUADRIC, INTERSECTION_NORMAL_KIND_TRIANGLE,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryPlacement {
    pub page: u32,
    pub vertex_offset: u32,
    pub index_offset: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeometryPagePlan {
    pub vertex_capacity: u32,
    pub index_capacity: u32,
    pub pages: Vec<GeometryPageRange>,
    pub placements: Vec<GeometryPlacement>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryPageRange {
    pub first_geometry: usize,
    pub geometry_count: usize,
    pub vertex_count: u32,
    pub index_count: u32,
}

impl GeometryPagePlan {
    pub fn new(
        geometries: &[flat::Geometry],
        vertex_capacity: u32,
        index_capacity: u32,
    ) -> Result<Self, PbrtError> {
        if vertex_capacity == 0 || index_capacity == 0 {
            return Err(PbrtError::error(
                "WebGPU geometry page capacities must be positive.",
            ));
        }
        let mut pages = Vec::new();
        let mut placements = Vec::with_capacity(geometries.len());
        let mut current = GeometryPageRange {
            first_geometry: 0,
            geometry_count: 0,
            vertex_count: 0,
            index_count: 0,
        };
        for (geometry_id, geometry) in geometries.iter().enumerate() {
            if geometry.vertex_count == 0
                || geometry.index_count == 0
                || geometry.index_count % 3 != 0
            {
                return Err(PbrtError::error(&format!(
                    "WebGPU geometry {geometry_id} must contain vertices and a non-empty multiple of three indices."
                )));
            }
            if geometry.vertex_count > vertex_capacity || geometry.index_count > index_capacity {
                let required_vertex_bytes =
                    u64::from(geometry.vertex_count) * std::mem::size_of::<Vertex>() as u64;
                let required_index_bytes =
                    u64::from(geometry.index_count) * std::mem::size_of::<u32>() as u64;
                let vertex_capacity_bytes =
                    u64::from(vertex_capacity) * std::mem::size_of::<Vertex>() as u64;
                let index_capacity_bytes =
                    u64::from(index_capacity) * std::mem::size_of::<u32>() as u64;
                return Err(PbrtError::error(&format!(
                    "WebGPU geometry {geometry_id} exceeds one geometry page: vertices need {} elements/{required_vertex_bytes} bytes (capacity {vertex_capacity}/{vertex_capacity_bytes} bytes), indices need {} elements/{required_index_bytes} bytes (capacity {index_capacity}/{index_capacity_bytes} bytes).",
                    geometry.vertex_count, geometry.index_count
                )));
            }
            if current.geometry_count > 0
                && (u64::from(current.vertex_count) + u64::from(geometry.vertex_count)
                    > u64::from(vertex_capacity)
                    || u64::from(current.index_count) + u64::from(geometry.index_count)
                        > u64::from(index_capacity))
            {
                pages.push(current);
                current = GeometryPageRange {
                    first_geometry: geometry_id,
                    geometry_count: 0,
                    vertex_count: 0,
                    index_count: 0,
                };
            }
            placements.push(GeometryPlacement {
                page: u32::try_from(pages.len())
                    .map_err(|_| PbrtError::error("WebGPU geometry page count exceeds u32."))?,
                vertex_offset: current.vertex_count,
                index_offset: current.index_count,
            });
            current.geometry_count += 1;
            current.vertex_count += geometry.vertex_count;
            current.index_count += geometry.index_count;
        }
        if current.geometry_count > 0 {
            pages.push(current);
        }
        if pages.is_empty() {
            return Err(PbrtError::error(
                "WebGPU rendering requires at least one geometry page.",
            ));
        }
        Ok(Self {
            vertex_capacity,
            index_capacity,
            pages,
            placements,
        })
    }
}

pub fn convert_geometry_page(
    flat: &flat::Scene,
    plan: &GeometryPagePlan,
    page_index: usize,
) -> Result<(Vec<Vertex>, Vec<u32>, Vec<Geometry>), PbrtError> {
    let page = plan.pages.get(page_index).ok_or_else(|| {
        PbrtError::error(&format!(
            "WebGPU geometry page {page_index} is out of range."
        ))
    })?;
    let vertex_capacity = usize::try_from(page.vertex_count)
        .map_err(|_| PbrtError::error("WebGPU page vertex count does not fit usize."))?;
    let index_capacity = usize::try_from(page.index_count)
        .map_err(|_| PbrtError::error("WebGPU page index count does not fit usize."))?;
    let mut vertices = Vec::with_capacity(vertex_capacity);
    let mut indices = Vec::with_capacity(index_capacity);
    let mut gpu_geometries = Vec::with_capacity(page.geometry_count);
    let end_geometry = page
        .first_geometry
        .checked_add(page.geometry_count)
        .ok_or_else(|| PbrtError::error("WebGPU page geometry range overflowed."))?;
    for geometry_id in page.first_geometry..end_geometry {
        let source = flat.geometry.geometries.get(geometry_id).ok_or_else(|| {
            PbrtError::error(&format!("WebGPU geometry {geometry_id} is out of range."))
        })?;
        let vertex_end = source
            .first_vertex
            .checked_add(source.vertex_count)
            .ok_or_else(|| {
                PbrtError::error(&format!(
                    "Flat geometry {geometry_id} vertex range overflowed."
                ))
            })?;
        let index_end = source
            .first_index
            .checked_add(source.index_count)
            .ok_or_else(|| {
                PbrtError::error(&format!(
                    "Flat geometry {geometry_id} index range overflowed."
                ))
            })?;
        let source_vertices = flat
            .geometry
            .vertices
            .get(source.first_vertex as usize..vertex_end as usize)
            .ok_or_else(|| {
                PbrtError::error(&format!(
                    "Flat geometry {geometry_id} vertex range is out of bounds."
                ))
            })?;
        let source_indices = flat
            .geometry
            .indices
            .get(source.first_index as usize..index_end as usize)
            .ok_or_else(|| {
                PbrtError::error(&format!(
                    "Flat geometry {geometry_id} index range is out of bounds."
                ))
            })?;
        let vertex_offset = vertices.len();
        let index_offset = indices.len();
        let placement = *plan.placements.get(geometry_id).ok_or_else(|| {
            PbrtError::error(&format!(
                "WebGPU geometry {geometry_id} has no page placement."
            ))
        })?;
        if placement.page as usize != page_index
            || placement.vertex_offset as usize != vertex_offset
            || placement.index_offset as usize != index_offset
        {
            return Err(PbrtError::error(&format!(
                "WebGPU geometry {geometry_id} page plan is inconsistent."
            )));
        }
        for vertex in source_vertices {
            if !vertex.position.iter().all(|value| value.is_finite())
                || !vertex.uv.iter().all(|value| value.is_finite())
                || !vertex.normal.iter().all(|value| value.is_finite())
                || !vertex.tangent.iter().all(|value| value.is_finite())
            {
                return Err(PbrtError::error(&format!(
                    "Flat geometry {geometry_id} contains a non-finite vertex attribute."
                )));
            }
            vertices.push(Vertex {
                position: [
                    vertex.position[0],
                    vertex.position[1],
                    vertex.position[2],
                    1.0,
                ],
                normal: [vertex.normal[0], vertex.normal[1], vertex.normal[2], 0.0],
                tangent: [vertex.tangent[0], vertex.tangent[1], vertex.tangent[2], 0.0],
                uv: vertex.uv,
                padding: [0; 2],
            });
        }
        for &absolute_index in source_indices {
            if absolute_index < source.first_vertex || absolute_index >= vertex_end {
                return Err(PbrtError::error(&format!(
                    "Flat geometry {geometry_id} contains an index outside its vertex range."
                )));
            }
            indices.push(absolute_index - source.first_vertex);
        }
        for triangle in indices[index_offset..].chunks_exact(3) {
            let p0 = vertices[vertex_offset + triangle[0] as usize].position;
            let p1 = vertices[vertex_offset + triangle[1] as usize].position;
            let p2 = vertices[vertex_offset + triangle[2] as usize].position;
            let edge0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let edge1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                edge0[1] * edge1[2] - edge0[2] * edge1[1],
                edge0[2] * edge1[0] - edge0[0] * edge1[2],
                edge0[0] * edge1[1] - edge0[1] * edge1[0],
            ];
            let norm_squared = cross.iter().map(|value| value * value).sum::<f32>();
            if !cross.iter().all(|value| value.is_finite())
                || !norm_squared.is_finite()
                || norm_squared == 0.0
            {
                return Err(PbrtError::error(&format!(
                    "Flat geometry {geometry_id} contains a zero-area or non-finite triangle."
                )));
            }
        }
        gpu_geometries.push(Geometry {
            vertex_offset: placement.vertex_offset,
            vertex_count: source.vertex_count,
            index_offset: placement.index_offset,
            index_count: source.index_count,
            intersection_normal_kind: match source.source_shape {
                SourceShape::Disk | SourceShape::Sphere => INTERSECTION_NORMAL_KIND_QUADRIC,
                _ => INTERSECTION_NORMAL_KIND_TRIANGLE,
            },
            page: placement.page,
            padding: [0; 2],
        });
    }
    Ok((vertices, indices, gpu_geometries))
}

pub fn validate_geometry_page_plan(
    flat: &flat::Scene,
    plan: &GeometryPagePlan,
) -> Result<(), PbrtError> {
    if plan.placements.len() != flat.geometry.geometries.len()
        || plan.pages.is_empty()
        || plan.pages.iter().any(|page| {
            page.vertex_count > plan.vertex_capacity || page.index_count > plan.index_capacity
        })
    {
        return Err(PbrtError::error(
            "WebGPU geometry page plan does not match the scene or its capacities.",
        ));
    }
    validate_geometry_page_coverage(&plan.pages, flat.geometry.geometries.len())?;
    for (page_index, page) in plan.pages.iter().enumerate() {
        let end = page
            .first_geometry
            .checked_add(page.geometry_count)
            .ok_or_else(|| PbrtError::error("WebGPU page geometry range overflowed."))?;
        if end > flat.geometry.geometries.len() {
            return Err(PbrtError::error(&format!(
                "WebGPU geometry page {page_index} references geometry {end} beyond the scene."
            )));
        }
        let mut vertex_offset = 0u32;
        let mut index_offset = 0u32;
        for geometry_id in page.first_geometry..end {
            let geometry = &flat.geometry.geometries[geometry_id];
            let placement = plan.placements[geometry_id];
            if placement.page as usize != page_index
                || placement.vertex_offset != vertex_offset
                || placement.index_offset != index_offset
            {
                return Err(PbrtError::error(&format!(
                    "WebGPU geometry {geometry_id} page plan is inconsistent."
                )));
            }
            let vertex_end = geometry
                .first_vertex
                .checked_add(geometry.vertex_count)
                .ok_or_else(|| {
                    PbrtError::error(&format!(
                        "Flat geometry {geometry_id} vertex range overflowed."
                    ))
                })?;
            let index_end = geometry
                .first_index
                .checked_add(geometry.index_count)
                .ok_or_else(|| {
                    PbrtError::error(&format!(
                        "Flat geometry {geometry_id} index range overflowed."
                    ))
                })?;
            let vertices = flat
                .geometry
                .vertices
                .get(geometry.first_vertex as usize..vertex_end as usize)
                .ok_or_else(|| {
                    PbrtError::error(&format!(
                        "Flat geometry {geometry_id} vertex range is out of bounds."
                    ))
                })?;
            let indices = flat
                .geometry
                .indices
                .get(geometry.first_index as usize..index_end as usize)
                .ok_or_else(|| {
                    PbrtError::error(&format!(
                        "Flat geometry {geometry_id} index range is out of bounds."
                    ))
                })?;
            if indices
                .iter()
                .any(|index| *index < geometry.first_vertex || *index >= vertex_end)
            {
                return Err(PbrtError::error(&format!(
                    "Flat geometry {geometry_id} contains an index outside its vertex range."
                )));
            }
            if vertices.iter().any(|vertex| {
                !vertex.position.iter().all(|value| value.is_finite())
                    || !vertex.uv.iter().all(|value| value.is_finite())
                    || !vertex.normal.iter().all(|value| value.is_finite())
                    || !vertex.tangent.iter().all(|value| value.is_finite())
            }) {
                return Err(PbrtError::error(&format!(
                    "Flat geometry {geometry_id} contains a non-finite vertex attribute."
                )));
            }
            vertex_offset = vertex_offset
                .checked_add(geometry.vertex_count)
                .ok_or_else(|| PbrtError::error("WebGPU page vertex range overflowed."))?;
            index_offset = index_offset
                .checked_add(geometry.index_count)
                .ok_or_else(|| PbrtError::error("WebGPU page index range overflowed."))?;
        }
        if vertex_offset != page.vertex_count || index_offset != page.index_count {
            return Err(PbrtError::error(&format!(
                "WebGPU geometry page {page_index} counts do not match its geometry ranges."
            )));
        }
    }
    Ok(())
}

fn validate_geometry_page_coverage(
    pages: &[GeometryPageRange],
    geometry_count: usize,
) -> Result<(), PbrtError> {
    let mut next_geometry = 0usize;
    for (page_index, page) in pages.iter().enumerate() {
        if page.first_geometry != next_geometry || page.geometry_count == 0 {
            return Err(PbrtError::error(&format!(
                "WebGPU geometry page {page_index} does not continue the preceding geometry range."
            )));
        }
        next_geometry = page
            .first_geometry
            .checked_add(page.geometry_count)
            .ok_or_else(|| PbrtError::error("WebGPU page geometry range overflowed."))?;
        if next_geometry > geometry_count {
            return Err(PbrtError::error(&format!(
                "WebGPU geometry page {page_index} references geometry {next_geometry} beyond the scene."
            )));
        }
    }
    if next_geometry != geometry_count {
        return Err(PbrtError::error(&format!(
            "WebGPU geometry page plan covers {next_geometry} of {geometry_count} geometries."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_coverage_rejects_gaps_and_trailing_omissions() {
        let ranges = [
            GeometryPageRange {
                first_geometry: 0,
                geometry_count: 1,
                vertex_count: 1,
                index_count: 3,
            },
            GeometryPageRange {
                first_geometry: 2,
                geometry_count: 1,
                vertex_count: 1,
                index_count: 3,
            },
        ];
        assert!(validate_geometry_page_coverage(&ranges, 3).is_err());
        assert!(validate_geometry_page_coverage(&ranges[..1], 3).is_err());
    }
}
