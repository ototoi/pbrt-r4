pub type Float = f32;
pub type Index = u32;

macro_rules! typed_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub Index);
    };
}

typed_id!(TransformId);
typed_id!(GeometryId);
typed_id!(PrimitiveId);
typed_id!(MaterialId);
typed_id!(SpectrumId);
typed_id!(LightId);
typed_id!(FloatTextureId);
typed_id!(SpectrumTextureId);
typed_id!(TextureMappingId);
typed_id!(ImageId);
typed_id!(InstanceDefinitionId);
typed_id!(InstanceId);
typed_id!(MinMaxNodeId);
typed_id!(SourceId);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point2(pub [Float; 2]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point3(pub [Float; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector2(pub [Float; 2]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector3(pub [Float; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normal3(pub [Float; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix4x4(pub [[Float; 4]; 4]);

impl Matrix4x4 {
    pub fn identity() -> Self {
        Self([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix3x3(pub [[Float; 3]; 3]);

impl Matrix3x3 {
    pub fn identity() -> Self {
        Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticTransform {
    pub render_from_object: Matrix4x4,
    pub object_from_render: Matrix4x4,
    pub swaps_handedness: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transform {
    Static(StaticTransform),
    Animated(AnimatedTransform),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimatedTransform {
    pub start: StaticTransform,
    pub end: StaticTransform,
    pub start_time: Float,
    pub end_time: Float,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds3 {
    pub min: Point3,
    pub max: Point3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds2 {
    pub min: Point2,
    pub max: Point2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Range {
    pub offset: Index,
    pub count: Index,
}
