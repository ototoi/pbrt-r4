//! Minimal semantic GPU IR used by the initial backend contract.
//!
//! This is intentionally not a device ABI. It contains no `wgpu` handles,
//! raw pointers, shader bindings, or CPU trait objects. Geometry, materials,
//! and textures will be added in later IR phases.

pub type GpuFloat = f32;
pub type GpuIndex = u32;

macro_rules! typed_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub struct $name(pub GpuIndex);
    };
}

typed_id!(TransformId);
typed_id!(GeometryId);
typed_id!(PrimitiveId);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuPoint2(pub [GpuFloat; 2]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuPoint3(pub [GpuFloat; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuVector3(pub [GpuFloat; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuNormal3(pub [GpuFloat; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMatrix4x4(pub [[GpuFloat; 4]; 4]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuStaticTransform {
    pub render_from_object: GpuMatrix4x4,
    pub object_from_render: GpuMatrix4x4,
    pub swaps_handedness: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuTransform {
    Static(GpuStaticTransform),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBounds3 {
    pub min: GpuPoint3,
    pub max: GpuPoint3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuIrVersion {
    pub major: u16,
    pub minor: u16,
}

pub const CURRENT_IR_VERSION: GpuIrVersion = GpuIrVersion { major: 1, minor: 0 };

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBounds2i {
    pub min: [GpuIndex; 2],
    pub max: [GpuIndex; 2],
}

impl GpuBounds2i {
    pub fn area(self) -> Option<u64> {
        let width = u64::from(self.max[0]).checked_sub(u64::from(self.min[0]))?;
        let height = u64::from(self.max[1]).checked_sub(u64::from(self.min[1]))?;
        (width > 0 && height > 0).then(|| width.checked_mul(height))?
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRenderConfig {
    pub pixel_bounds: GpuBounds2i,
    pub sample_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuTriangleMesh {
    pub positions: Vec<GpuPoint3>,
    pub indices: Vec<[GpuIndex; 3]>,
    pub normals: Option<Vec<GpuNormal3>>,
    pub tangents: Option<Vec<GpuVector3>>,
    pub uvs: Option<Vec<GpuPoint2>>,
    pub face_indices: Option<Vec<GpuIndex>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuGeometry {
    TriangleMesh(GpuTriangleMesh),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPrimitive {
    pub geometry: GeometryId,
    pub transform: TransformId,
    pub reverse_orientation: bool,
}

impl Default for GpuRenderConfig {
    fn default() -> Self {
        Self {
            pixel_bounds: GpuBounds2i {
                min: [0, 0],
                max: [1, 1],
            },
            sample_count: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuSceneData {
    pub transforms: Vec<GpuTransform>,
    pub geometry: Vec<GpuGeometry>,
    pub primitives: Vec<GpuPrimitive>,
    pub render: GpuRenderConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuSceneDraft {
    pub version: GpuIrVersion,
    pub data: GpuSceneData,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuSceneIr {
    version: GpuIrVersion,
    data: GpuSceneData,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSceneView<'a> {
    pub version: &'a GpuIrVersion,
    pub transforms: &'a [GpuTransform],
    pub geometry: &'a [GpuGeometry],
    pub primitives: &'a [GpuPrimitive],
    pub render: &'a GpuRenderConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuIrValidationError {
    UnsupportedMajorVersion {
        found: GpuIrVersion,
        expected_major: u16,
    },
    InvalidPixelBounds,
    InvalidSampleCount,
    EmptyTriangleMesh {
        geometry: GeometryId,
    },
    TriangleIndexOutOfBounds {
        geometry: GeometryId,
        index: GpuIndex,
    },
    DegenerateTriangle {
        geometry: GeometryId,
        triangle: GpuIndex,
    },
    AttributeLengthMismatch {
        geometry: GeometryId,
    },
    InvalidGeometryReference {
        primitive: PrimitiveId,
        geometry: GeometryId,
    },
    InvalidTransformReference {
        primitive: PrimitiveId,
        transform: TransformId,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuIrValidationErrors {
    issues: Box<[GpuIrValidationError]>,
}

impl GpuIrValidationErrors {
    pub fn issues(&self) -> &[GpuIrValidationError] {
        &self.issues
    }
}

impl GpuSceneDraft {
    pub fn finish(self) -> Result<GpuSceneIr, GpuIrValidationErrors> {
        let mut issues = Vec::new();
        if self.version.major != CURRENT_IR_VERSION.major {
            issues.push(GpuIrValidationError::UnsupportedMajorVersion {
                found: self.version,
                expected_major: CURRENT_IR_VERSION.major,
            });
        }
        if self.data.render.pixel_bounds.area().is_none() {
            issues.push(GpuIrValidationError::InvalidPixelBounds);
        }
        if self.data.render.sample_count == 0 {
            issues.push(GpuIrValidationError::InvalidSampleCount);
        }
        for (geometry_index, geometry) in self.data.geometry.iter().enumerate() {
            let geometry_id = GeometryId(geometry_index as GpuIndex);
            match geometry {
                GpuGeometry::TriangleMesh(mesh) => {
                    validate_triangle_mesh(geometry_id, mesh, &mut issues)
                }
            }
        }
        for (primitive_index, primitive) in self.data.primitives.iter().enumerate() {
            let primitive_id = PrimitiveId(primitive_index as GpuIndex);
            if usize::try_from(primitive.geometry.0)
                .ok()
                .and_then(|index| self.data.geometry.get(index))
                .is_none()
            {
                issues.push(GpuIrValidationError::InvalidGeometryReference {
                    primitive: primitive_id,
                    geometry: primitive.geometry,
                });
            }
            if usize::try_from(primitive.transform.0)
                .ok()
                .and_then(|index| self.data.transforms.get(index))
                .is_none()
            {
                issues.push(GpuIrValidationError::InvalidTransformReference {
                    primitive: primitive_id,
                    transform: primitive.transform,
                });
            }
        }
        if issues.is_empty() {
            Ok(GpuSceneIr {
                version: self.version,
                data: self.data,
            })
        } else {
            Err(GpuIrValidationErrors {
                issues: issues.into_boxed_slice(),
            })
        }
    }
}

fn validate_triangle_mesh(
    geometry: GeometryId,
    mesh: &GpuTriangleMesh,
    issues: &mut Vec<GpuIrValidationError>,
) {
    if mesh.positions.is_empty() || mesh.indices.is_empty() {
        issues.push(GpuIrValidationError::EmptyTriangleMesh { geometry });
        return;
    }
    let position_count = mesh.positions.len();
    for (triangle_index, triangle) in mesh.indices.iter().enumerate() {
        if triangle
            .iter()
            .any(|index| usize::try_from(*index).map_or(true, |index| index >= position_count))
        {
            let index = triangle
                .iter()
                .copied()
                .find(|index| usize::try_from(*index).map_or(true, |index| index >= position_count))
                .unwrap_or_default();
            issues.push(GpuIrValidationError::TriangleIndexOutOfBounds { geometry, index });
        }
        if triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[2] == triangle[0] {
            issues.push(GpuIrValidationError::DegenerateTriangle {
                geometry,
                triangle: triangle_index as GpuIndex,
            });
        }
    }
    let expected = position_count;
    if mesh.normals.as_ref().is_some_and(|v| v.len() != expected)
        || mesh.tangents.as_ref().is_some_and(|v| v.len() != expected)
        || mesh.uvs.as_ref().is_some_and(|v| v.len() != expected)
        || mesh
            .face_indices
            .as_ref()
            .is_some_and(|v| v.len() != mesh.indices.len())
    {
        issues.push(GpuIrValidationError::AttributeLengthMismatch { geometry });
    }
}

impl GpuSceneIr {
    pub fn view(&self) -> GpuSceneView<'_> {
        GpuSceneView {
            version: &self.version,
            transforms: &self.data.transforms,
            geometry: &self.data.geometry,
            primitives: &self.data.primitives,
            render: &self.data.render,
        }
    }
}
