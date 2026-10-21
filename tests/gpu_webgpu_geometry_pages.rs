#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::flat::{Geometry, SourceShape};
use pbrt_r4::gpu::webgpu::scene::geometry::GeometryPagePlan;

fn geometry(vertex_count: u32, index_count: u32) -> Geometry {
    Geometry {
        source_shape: SourceShape::TriangleMesh,
        vertex_count,
        index_count,
        ..Default::default()
    }
}

#[test]
fn geometry_pages_pack_in_registration_order_and_keep_both_ranges_together() {
    let geometries = vec![
        geometry(4, 6),
        geometry(3, 3),
        geometry(4, 6),
        geometry(3, 3),
        geometry(4, 6),
    ];
    let plan = GeometryPagePlan::new(&geometries, 7, 9).unwrap();

    assert_eq!(plan.pages.len(), 3);
    assert_eq!(
        plan.pages,
        vec![
            pbrt_r4::gpu::webgpu::scene::geometry::GeometryPageRange {
                first_geometry: 0,
                geometry_count: 2,
                vertex_count: 7,
                index_count: 9,
            },
            pbrt_r4::gpu::webgpu::scene::geometry::GeometryPageRange {
                first_geometry: 2,
                geometry_count: 2,
                vertex_count: 7,
                index_count: 9,
            },
            pbrt_r4::gpu::webgpu::scene::geometry::GeometryPageRange {
                first_geometry: 4,
                geometry_count: 1,
                vertex_count: 4,
                index_count: 6,
            },
        ]
    );
    assert_eq!(plan.placements[0].page, 0);
    assert_eq!(plan.placements[1].page, 0);
    assert_eq!(plan.placements[1].vertex_offset, 4);
    assert_eq!(plan.placements[1].index_offset, 6);
    assert_eq!(plan.placements[2].page, 1);
    assert_eq!(plan.placements[4].page, 2);
}

#[test]
fn geometry_page_starts_when_either_capacity_is_full() {
    let geometries = vec![geometry(2, 6), geometry(2, 3), geometry(2, 3)];
    let plan = GeometryPagePlan::new(&geometries, 8, 9).unwrap();
    assert_eq!(plan.pages.len(), 2);
    assert_eq!(plan.placements[0].page, 0);
    assert_eq!(plan.placements[1].page, 0);
    assert_eq!(plan.placements[2].page, 1);
}

#[test]
fn geometry_larger_than_a_page_reports_the_geometry_and_capacity() {
    let error = GeometryPagePlan::new(&[geometry(11, 3)], 10, 10).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("geometry 0"), "{message}");
    assert!(message.contains("capacity 10"), "{message}");
}

#[test]
fn geometry_pages_reject_empty_invalid_capacity_and_non_triangle_indices() {
    assert!(GeometryPagePlan::new(&[], 1, 1).is_err());
    assert!(GeometryPagePlan::new(&[geometry(3, 3)], 0, 3).is_err());
    assert!(GeometryPagePlan::new(&[geometry(3, 4)], 3, 4).is_err());
}

#[test]
fn page_plan_handles_large_aggregate_ranges_without_allocating_mesh_payloads() {
    let geometries = vec![geometry(u32::MAX, 3), geometry(u32::MAX, 3), geometry(4, 3)];
    let plan = GeometryPagePlan::new(&geometries, u32::MAX, 6).unwrap();
    assert_eq!(plan.pages.len(), 3);
    assert_eq!(plan.placements[2].page, 2);
}
