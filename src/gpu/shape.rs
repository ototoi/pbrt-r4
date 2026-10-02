#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SourceShape {
    #[default]
    TriangleMesh,
    PlyMesh,
    LoopSubdiv,
    Disk,
    Sphere,
    Cylinder,
    Cone,
    Paraboloid,
    HeightField,
    BilinearMesh,
    Hyperboloid,
    Nurbs,
}
