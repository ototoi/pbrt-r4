#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::flat::flatten_node;
use pbrt_r4::gpu::flat::SourceShape;
use pbrt_r4::gpu::node::{
    prepare_triangle_meshes, tessellate_shapes, Component, CurvesShape, Node, Shape,
    ShapeComponent, TriangleMeshShape, DEFAULT_CURVE_DICE_SLICES, DEFAULT_CURVE_DICE_VERTICES,
};
use pbrt_r4::paramdict::ParameterDictionary;
use pbrt_r4::parser::parse_string;
use pbrt_r4::parser::scene_builder::SceneBuilder;
use pbrt_r4::shapes::curve::curve_geometry_from_params;
use pbrt_r4::util::base::Float;

const PREFIX: &str = r#"
Integrator "path"
Sampler "independent" "integer pixelsamples" [1]
Film "rgb" "integer xresolution" [1] "integer yresolution" [1]
Camera "perspective"
WorldBegin
"#;
const SUFFIX: &str = "WorldEnd\n";

fn curve_params(curve_type: &str) -> ParameterDictionary {
    let mut params = ParameterDictionary::new();
    params.add_string("type", curve_type);
    params.add_float("width", 0.2);
    params.add_point(
        "P",
        &[
            0.0, 0.0, 0.0, //
            0.3, 0.0, 0.0, //
            0.7, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
        ],
    );
    if curve_type == "ribbon" {
        params.add_point("N", &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    }
    params
}

fn curves_node(curve_type: &str) -> Node {
    let mut node = Node::new("curve-node");
    node.add_component(Component::Shape(ShapeComponent {
        shape: Shape::Curves(Box::new(CurvesShape {
            params: vec![curve_params(curve_type)],
        })),
        reverse_orientation: false,
    }));
    node
}

fn tessellated_mesh(curve_type: &str) -> TriangleMeshShape {
    let mut node = curves_node(curve_type);
    tessellate_shapes(&mut node).expect("curve dicing should succeed");
    let Some(Component::Shape(component)) = node.components.first() else {
        panic!("diced node lost its shape component");
    };
    let Shape::TriangleMesh(mesh) = &component.shape else {
        panic!("curve should be tessellated to a triangle mesh");
    };
    (**mesh).clone()
}

#[test]
fn curve_dicing_matches_pbrt_v4_vertex_and_index_counts() {
    // pbrt-v4 diceCurveToBLP(shape, 5 /* nseg */, 5 /* nvert */) produces
    // (nDiceV + 1) * (nDiceU + 1) tube vertices and nDiceU * nDiceV * 4
    // bilinear patches; 2 * (nDiceU + 1) ribbon vertices and nDiceU patches.
    let tube_vertices = (DEFAULT_CURVE_DICE_VERTICES + 1) * (DEFAULT_CURVE_DICE_SLICES + 1);
    let tube_indices = DEFAULT_CURVE_DICE_SLICES * DEFAULT_CURVE_DICE_VERTICES * 6;
    let ribbon_vertices = 2 * (DEFAULT_CURVE_DICE_SLICES + 1);
    let ribbon_indices = DEFAULT_CURVE_DICE_SLICES * 6;

    for curve_type in ["flat", "cylinder"] {
        let mesh = tessellated_mesh(curve_type);
        assert_eq!(mesh.positions.len(), tube_vertices, "{curve_type}");
        assert_eq!(mesh.indices.len(), tube_indices, "{curve_type}");
        assert_eq!(mesh.source_shape, "curves");
    }
    let ribbon = tessellated_mesh("ribbon");
    assert_eq!(ribbon.positions.len(), ribbon_vertices);
    assert_eq!(ribbon.indices.len(), ribbon_indices);
}

#[test]
fn curve_dicing_uv_v_spans_the_width_for_the_hair_model() {
    // The hair BCSDF computes h = -1 + 2 * uv.y; every diced curve must keep
    // uv.y spanning [0, 1] across the curve width.
    for curve_type in ["flat", "cylinder", "ribbon"] {
        let mesh = tessellated_mesh(curve_type);
        assert!(mesh
            .uvs
            .as_ref()
            .unwrap()
            .iter()
            .all(|uv| uv.0[0] >= 0.0 && uv.0[0] <= 1.0 && uv.0[1] >= 0.0 && uv.0[1] <= 1.0));
    }

    // Tube seam: the first and last vertices of a slice repeat the position
    // while carrying uv.y = 0 and 1, as in pbrt-v4 (up to the sin(2*pi)
    // round-off).
    let tube = tessellated_mesh("cylinder");
    let ring = DEFAULT_CURVE_DICE_VERTICES + 1;
    for slice in 1..=DEFAULT_CURVE_DICE_SLICES {
        let first = slice * ring;
        let last = first + DEFAULT_CURVE_DICE_VERTICES;
        for axis in 0..3 {
            assert!(
                (tube.positions[first].0[axis] - tube.positions[last].0[axis]).abs() < 1e-5,
                "seam vertices should repeat the position"
            );
        }
        assert_eq!(tube.uvs.as_ref().unwrap()[first].0[1], 0.0);
        assert_eq!(tube.uvs.as_ref().unwrap()[last].0[1], 1.0);
    }

    // Ribbon: v = 1 lies on the +dpdv/2 side of the curve, so h keeps the
    // same orientation as the CPU curve intersection.
    let ribbon = tessellated_mesh("ribbon");
    for slice in 0..=DEFAULT_CURVE_DICE_SLICES {
        let v0 = slice * 2;
        let v1 = v0 + 1;
        let (p0, p1) = (ribbon.positions[v0].0, ribbon.positions[v1].0);
        assert!(
            p1[1] > p0[1],
            "ribbon uv v=1 vertex should sit on the +width side"
        );
    }
}

#[test]
fn diced_curve_normals_agree_with_triangle_winding() {
    // The baked shading normals must stay on the same side as the geometric
    // triangle normals so the WebGPU triangle pipeline sees a consistent
    // surface orientation.
    for curve_type in ["flat", "cylinder", "ribbon"] {
        let mesh = tessellated_mesh(curve_type);
        let normals = mesh.normals.as_ref().unwrap();
        for triangle in mesh.indices.chunks_exact(3) {
            let p0 = mesh.positions[triangle[0] as usize].0;
            let p1 = mesh.positions[triangle[1] as usize].0;
            let p2 = mesh.positions[triangle[2] as usize].0;
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let geometric = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            // The winding is chosen against triangles that actually have
            // area; zero-area triangles are removed later by
            // remove_invalid_triangles.
            if geometric.iter().map(|v| v * v).sum::<f32>() == 0.0 {
                continue;
            }
            for &corner in triangle {
                let normal = normals[corner as usize].0;
                let dot =
                    geometric[0] * normal[0] + geometric[1] * normal[1] + geometric[2] * normal[2];
                assert!(
                    dot > 0.0,
                    "{curve_type}: geometric normal opposes baked normal"
                );
            }
        }
    }
}

#[test]
fn diced_curve_positions_follow_the_bezier_centerline() {
    // The curve centerline is recovered from each slice: the ribbon slice
    // midpoint is on the Bezier curve, and the tube slice's distinct ring
    // vertices average to the Bezier point (the vertices around a full
    // circle sum to the center, like pbrt-v4's duplicated seam vertex).
    let ribbon = tessellated_mesh("ribbon");
    for (slice, expected) in [
        (0usize, [0.0, 0.0, 0.0]),
        (DEFAULT_CURVE_DICE_SLICES, [1.0, 0.0, 0.0]),
    ] {
        let (a, b) = (
            ribbon.positions[slice * 2].0,
            ribbon.positions[slice * 2 + 1].0,
        );
        let center = [
            (a[0] + b[0]) / 2.0,
            (a[1] + b[1]) / 2.0,
            (a[2] + b[2]) / 2.0,
        ];
        for axis in 0..3 {
            assert!(
                (center[axis] - expected[axis]).abs() < 1e-5,
                "ribbon centerline at slice {slice}"
            );
        }
    }

    for curve_type in ["flat", "cylinder"] {
        let mesh = tessellated_mesh(curve_type);
        let ring = DEFAULT_CURVE_DICE_VERTICES + 1;
        for (slice, expected) in [
            (0usize, [0.0, 0.0, 0.0]),
            (DEFAULT_CURVE_DICE_SLICES, [1.0, 0.0, 0.0]),
        ] {
            let start = slice * ring;
            let mut center = [0.0f32; 3];
            for vertex in &mesh.positions[start..start + DEFAULT_CURVE_DICE_VERTICES] {
                for axis in 0..3 {
                    center[axis] += vertex.0[axis];
                }
            }
            for axis in 0..3 {
                center[axis] /= DEFAULT_CURVE_DICE_VERTICES as f32;
                assert!(
                    (center[axis] - expected[axis]).abs() < 1e-3,
                    "{curve_type} centerline at slice {slice}: {center:?}"
                );
            }
        }
    }
}

#[test]
fn curve_geometry_parses_all_degree_and_basis_combinations() {
    let cases = [
        ("bezier", 2, 3, 1),
        ("bezier", 2, 5, 2),
        ("bezier", 3, 4, 1),
        ("bezier", 3, 7, 2),
        ("bspline", 2, 3, 1),
        ("bspline", 2, 4, 2),
        ("bspline", 3, 4, 1),
        ("bspline", 3, 5, 2),
    ];
    for (basis, degree, n_control_points, n_segments) in cases {
        let mut params = ParameterDictionary::new();
        params.add_string("type", "flat");
        params.add_int("degree", degree);
        if basis != "bezier" {
            params.add_string("basis", basis);
        }
        let mut points = Vec::new();
        for i in 0..n_control_points {
            points.push(i as Float);
            points.push(0.0 as Float);
            points.push(0.0 as Float);
        }
        params.add_point("P", &points);
        let geometry = curve_geometry_from_params(&params)
            .unwrap_or_else(|error| panic!("{basis}/{degree}: {error}"));
        assert_eq!(geometry.n_segments, n_segments, "{basis}/{degree}");
        assert_eq!(geometry.seg_cp_bezier.len(), n_segments, "{basis}/{degree}");
        assert!(geometry.normals.is_none(), "{basis}/{degree}");
    }
}

#[test]
fn curve_geometry_rejects_invalid_parameters() {
    let mut params = ParameterDictionary::new();
    params.add_int("degree", 4);
    params.add_point("P", &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(curve_geometry_from_params(&params)
        .unwrap_err()
        .to_string()
        .contains("only degree 2 and 3 curves are supported"));

    let mut params = ParameterDictionary::new();
    params.add_string("basis", "invalid_basis");
    params.add_point("P", &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(curve_geometry_from_params(&params)
        .unwrap_err()
        .to_string()
        .contains("only \"bezier\" and \"bspline\" are supported"));

    let mut params = ParameterDictionary::new();
    params.add_string("type", "ribbon");
    params.add_point(
        "P",
        &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0],
    );
    assert!(curve_geometry_from_params(&params)
        .unwrap_err()
        .to_string()
        .contains("Must provide normals \"N\" at curve endpoints with ribbon curves."));

    let mut params = ParameterDictionary::new();
    params.add_string("type", "ribbon");
    params.add_point(
        "P",
        &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0],
    );
    params.add_point("N", &[0.0, 0.0, 1.0]);
    assert!(curve_geometry_from_params(&params)
        .unwrap_err()
        .to_string()
        .contains("Invalid number of normals 1"));
}

#[test]
fn curve_geometry_ignores_normals_for_non_ribbon_types() {
    // pbrt-v4 warns and drops N for flat/cylinder curves (D9: ignore).
    let mut params = ParameterDictionary::new();
    params.add_string("type", "cylinder");
    params.add_point(
        "P",
        &[
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.3, //
            0.0, 0.0, 0.7, //
            0.0, 0.0, 1.0, //
        ],
    );
    params.add_point("N", &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    let geometry = curve_geometry_from_params(&params).expect("N should be ignored, not fatal");
    assert!(geometry.normals.is_none());
}

fn flat_curves_scene(extra: &str) -> pbrt_r4::gpu::flat::Scene {
    let mut builder = SceneBuilder::new();
    parse_string(&format!("{PREFIX}{extra}{SUFFIX}"), &mut builder).expect("scene should parse");
    let root = builder.build_gpu_ir_node().expect("GPU IR should build");
    {
        let mut node = root.write().expect("node lock");
        tessellate_shapes(&mut node).expect("curves should dice");
        pbrt_r4::gpu::node::prepare_triangle_meshes(&mut node).expect("meshes should prepare");
    }
    let flat = flatten_node(root).expect("flat scene should build");
    flat
}

#[test]
fn consecutive_curve_statements_dice_into_one_grouped_geometry() {
    let scene = flat_curves_scene(
        r#"Shape "curve" "string type" "cylinder" "float width" [0.2]
    "point3 P" [0 0 0 0.3 0 0 0.7 0 0 1 0 0]
Shape "curve" "string type" "cylinder" "float width" [0.2]
    "point3 P" [0 0 1 0.3 0 1 0.7 0 1 1 0 1]
"#,
    );
    assert_eq!(scene.geometry.geometries.len(), 1);
    assert_eq!(
        scene.geometry.geometries[0].source_shape,
        SourceShape::Curves
    );
    let tube_vertices = (DEFAULT_CURVE_DICE_VERTICES + 1) * (DEFAULT_CURVE_DICE_SLICES + 1);
    assert_eq!(scene.geometry.vertices.len(), 2 * tube_vertices);
    assert_eq!(
        scene.geometry.indices.len(),
        2 * DEFAULT_CURVE_DICE_SLICES * DEFAULT_CURVE_DICE_VERTICES * 6
    );
}

#[test]
fn gpu_builder_rejects_curve_alpha_and_shadowalpha() {
    for parameter in [
        "\"float alpha\" [0.5]",
        "\"float alpha\" [1.0]",
        "\"float shadowalpha\" [0.5]",
    ] {
        let mut builder = SceneBuilder::new();
        parse_string(
            &format!(
                "{PREFIX}Shape \"curve\" \"string type\" \"flat\" \"point3 P\" [0 0 0 0.3 0 0 0.7 0 0 1 0 0] {parameter}\n{SUFFIX}"
            ),
            &mut builder,
        )
        .expect("scene should parse");
        let error = match builder.build_gpu_ir_node() {
            Ok(_) => panic!("curve alpha should be rejected on WebGPU"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("WebGPU does not support curve \"alpha\"/\"shadowalpha\""));
    }
}

#[test]
fn gpu_builder_reports_curve_validation_errors_at_tessellation() {
    // A ribbon without N parses and builds Node IR; the shared curve
    // validation error must surface when the curve is diced.
    let mut builder = SceneBuilder::new();
    parse_string(
        &format!(
            "{PREFIX}Shape \"curve\" \"string type\" \"ribbon\" \"point3 P\" [0 0 0 0.3 0 0 0.7 0 0 1 0 0]\n{SUFFIX}"
        ),
        &mut builder,
    )
    .expect("scene should parse");
    let root = builder.build_gpu_ir_node().expect("GPU IR should build");
    let mut node = root.write().expect("node lock");
    let error = tessellate_shapes(&mut node).expect_err("ribbon without N must fail");
    assert!(error
        .to_string()
        .contains("Must provide normals \"N\" at curve endpoints with ribbon curves."));
}

#[test]
fn tapered_curve_tips_keep_finite_tube_normals() {
    // A curve tapered to width 0 collapses its last ring onto the
    // centerline. v4 derives tube normals as Normalize(position - center),
    // which is NaN there; v4's degenerate quads are never hit by rays, but
    // the r4 mesh validator rejects any non-finite normal for the whole
    // node, so the dicer normalizes the unscaled radial direction instead.
    let mut params = curve_params("cylinder");
    params.add_float("width0", 0.2);
    params.add_float("width1", 0.0);
    let mut node = Node::new("tapered-curve-node");
    node.add_component(Component::Shape(ShapeComponent {
        shape: Shape::Curves(Box::new(CurvesShape {
            params: vec![params],
        })),
        reverse_orientation: false,
    }));
    tessellate_shapes(&mut node).expect("tapered curve dicing should succeed");
    prepare_triangle_meshes(&mut node).expect("tapered curve mesh should validate");
    let Some(Component::Shape(component)) = node.components.first() else {
        panic!("diced node lost its shape component");
    };
    let Shape::TriangleMesh(mesh) = &component.shape else {
        panic!("curve should be tessellated to a triangle mesh");
    };
    let normals = mesh.normals.as_ref().expect("diced tube normals");
    assert!(normals
        .iter()
        .all(|normal| normal.0.iter().all(|value| value.is_finite())));
}

#[test]
fn tapered_flat_curve_tips_keep_finite_tube_normals_too() {
    // The flat type dices through the same tube code path.
    let mut params = curve_params("flat");
    params.add_float("width0", 0.2);
    params.add_float("width1", 0.0);
    let mut node = Node::new("tapered-flat-curve-node");
    node.add_component(Component::Shape(ShapeComponent {
        shape: Shape::Curves(Box::new(CurvesShape {
            params: vec![params],
        })),
        reverse_orientation: false,
    }));
    tessellate_shapes(&mut node).expect("tapered flat dicing should succeed");
    prepare_triangle_meshes(&mut node).expect("tapered flat mesh should validate");
    let Some(Component::Shape(component)) = node.components.first() else {
        panic!("diced node lost its shape component");
    };
    let Shape::TriangleMesh(mesh) = &component.shape else {
        panic!("curve should be tessellated to a triangle mesh");
    };
    let normals = mesh.normals.as_ref().expect("diced tube normals");
    assert!(normals
        .iter()
        .all(|normal| normal.0.iter().all(|value| value.is_finite())));
}
