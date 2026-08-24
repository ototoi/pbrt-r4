pub type GpuFloat = f32;
pub type GpuIndex = u32;

macro_rules! typed_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub GpuIndex);
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
pub struct GpuPoint2(pub [GpuFloat; 2]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuPoint3(pub [GpuFloat; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuVector2(pub [GpuFloat; 2]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuVector3(pub [GpuFloat; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuNormal3(pub [GpuFloat; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMatrix4x4(pub [[GpuFloat; 4]; 4]);

impl GpuMatrix4x4 {
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
pub struct GpuMatrix3x3(pub [[GpuFloat; 3]; 3]);

impl GpuMatrix3x3 {
    pub fn identity() -> Self {
        Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuStaticTransform {
    pub render_from_object: GpuMatrix4x4,
    pub object_from_render: GpuMatrix4x4,
    pub swaps_handedness: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuTransform {
    Static(GpuStaticTransform),
    Animated(GpuAnimatedTransform),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuAnimatedTransform {
    pub start: GpuStaticTransform,
    pub end: GpuStaticTransform,
    pub start_time: GpuFloat,
    pub end_time: GpuFloat,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBounds3 {
    pub min: GpuPoint3,
    pub max: GpuPoint3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuBounds2 {
    pub min: GpuPoint2,
    pub max: GpuPoint2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRange {
    pub offset: GpuIndex,
    pub count: GpuIndex,
}
