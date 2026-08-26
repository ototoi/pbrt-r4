use super::{
    Bounds2, Bounds3, Float, FloatTextureId, GeometryId, LightId, MaterialId, MinMaxNodeId,
    Normal3, Point2, Point3, TransformId, Vector3,
};

#[derive(Clone, Debug, PartialEq)]
pub struct TriangleMesh {
    pub positions: Vec<Point3>,
    pub indices: Vec<[super::Index; 3]>,
    pub normals: Option<Vec<Normal3>>,
    pub tangents: Option<Vec<Vector3>>,
    pub uvs: Option<Vec<Point2>>,
    pub face_indices: Option<Vec<super::Index>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BilinearPatchMesh {
    pub positions: Vec<Point3>,
    pub indices: Vec<[super::Index; 4]>,
    pub normals: Option<Vec<Normal3>>,
    pub uvs: Option<Vec<Point2>>,
    pub face_indices: Option<Vec<super::Index>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CurveType {
    Flat,
    Cylinder,
    Ribbon,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurveSegment {
    pub control_points: [Point3; 4],
    pub widths: [Float; 2],
    pub endpoint_normals: Option<[Normal3; 2]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CurveMesh {
    pub curve_type: CurveType,
    pub curves: Vec<CurveSegment>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Quadric {
    Sphere {
        radius: Float,
        z_min: Float,
        z_max: Float,
        phi_max_radians: Float,
    },
    Cylinder {
        radius: Float,
        z_min: Float,
        z_max: Float,
        phi_max_radians: Float,
    },
    Disk {
        height: Float,
        radius: Float,
        inner_radius: Float,
        phi_max_radians: Float,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct MinMaxNode {
    pub parameter_bounds: Bounds2,
    pub displacement_min: Float,
    pub displacement_max: Float,
    pub children: Option<[MinMaxNodeId; 4]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DisplacedTriangleMesh {
    pub base_mesh: GeometryId,
    pub displacement: FloatTextureId,
    pub displacement_scale: Float,
    pub displacement_offset: Float,
    pub edge_length: Float,
    pub min_max_nodes: Box<[MinMaxNode]>,
    pub triangle_roots: Box<[MinMaxNodeId]>,
    pub displaced_bounds_object: Box<[Bounds3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Geometry {
    TriangleMesh(TriangleMesh),
    BilinearPatchMesh(BilinearPatchMesh),
    CurveMesh(CurveMesh),
    Quadric(Quadric),
    DisplacedTriangleMesh(DisplacedTriangleMesh),
}

#[derive(Clone, Debug, PartialEq)]
pub enum AreaLightBinding {
    None,
    Uniform(LightId),
    PerElement(Vec<LightId>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Primitive {
    pub geometry: GeometryId,
    pub transform: TransformId,
    pub material: Option<MaterialId>,
    pub alpha: Option<FloatTextureId>,
    pub area_light: AreaLightBinding,
    pub reverse_orientation: bool,
}
