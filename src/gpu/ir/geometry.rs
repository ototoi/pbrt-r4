use super::{
    FloatTextureId, GeometryId, GpuBounds2, GpuBounds3, GpuFloat, GpuNormal3, GpuPoint2, GpuPoint3,
    GpuVector3, LightId, MaterialId, MinMaxNodeId, TransformId,
};

#[derive(Clone, Debug, PartialEq)]
pub struct GpuTriangleMesh {
    pub positions: Vec<GpuPoint3>,
    pub indices: Vec<[super::GpuIndex; 3]>,
    pub normals: Option<Vec<GpuNormal3>>,
    pub tangents: Option<Vec<GpuVector3>>,
    pub uvs: Option<Vec<GpuPoint2>>,
    pub face_indices: Option<Vec<super::GpuIndex>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuBilinearPatchMesh {
    pub positions: Vec<GpuPoint3>,
    pub indices: Vec<[super::GpuIndex; 4]>,
    pub normals: Option<Vec<GpuNormal3>>,
    pub uvs: Option<Vec<GpuPoint2>>,
    pub face_indices: Option<Vec<super::GpuIndex>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCurveType {
    Flat,
    Cylinder,
    Ribbon,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuCurveSegment {
    pub control_points: [GpuPoint3; 4],
    pub widths: [GpuFloat; 2],
    pub endpoint_normals: Option<[GpuNormal3; 2]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuCurveMesh {
    pub curve_type: GpuCurveType,
    pub curves: Vec<GpuCurveSegment>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuQuadric {
    Sphere {
        radius: GpuFloat,
        z_min: GpuFloat,
        z_max: GpuFloat,
        phi_max_radians: GpuFloat,
    },
    Cylinder {
        radius: GpuFloat,
        z_min: GpuFloat,
        z_max: GpuFloat,
        phi_max_radians: GpuFloat,
    },
    Disk {
        height: GpuFloat,
        radius: GpuFloat,
        inner_radius: GpuFloat,
        phi_max_radians: GpuFloat,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuMinMaxNode {
    pub parameter_bounds: GpuBounds2,
    pub displacement_min: GpuFloat,
    pub displacement_max: GpuFloat,
    pub children: Option<[MinMaxNodeId; 4]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuDisplacedTriangleMesh {
    pub base_mesh: GeometryId,
    pub displacement: FloatTextureId,
    pub displacement_scale: GpuFloat,
    pub displacement_offset: GpuFloat,
    pub edge_length: GpuFloat,
    pub min_max_nodes: Box<[GpuMinMaxNode]>,
    pub triangle_roots: Box<[MinMaxNodeId]>,
    pub displaced_bounds_object: Box<[GpuBounds3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuGeometry {
    TriangleMesh(GpuTriangleMesh),
    BilinearPatchMesh(GpuBilinearPatchMesh),
    CurveMesh(GpuCurveMesh),
    Quadric(GpuQuadric),
    DisplacedTriangleMesh(GpuDisplacedTriangleMesh),
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuAreaLightBinding {
    None,
    Uniform(LightId),
    PerElement(Vec<LightId>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuPrimitive {
    pub geometry: GeometryId,
    pub transform: TransformId,
    pub material: Option<MaterialId>,
    pub alpha: Option<FloatTextureId>,
    pub shadow_alpha: Option<FloatTextureId>,
    pub area_light: GpuAreaLightBinding,
    pub reverse_orientation: bool,
}
