use super::super::ir::{GeometryId, TransformId};
use super::device::AccelerationMode;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendError {
    InvalidPrepareOptions { reason: &'static str },
    AdapterRequest(String),
    DeviceRequest(String),
    MissingRayQueryFeature,
    Plan(PlanError),
    InvalidRenderRequest(super::super::ir::GpuRenderRequestError),
    UnsupportedRenderRequest { reason: &'static str },
    Readback(String),
    UnsupportedAccelerationMode(AccelerationMode),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanError {
    EmptyScene,
    UnsupportedGeometry {
        geometry: GeometryId,
    },
    EmptyGeometry {
        geometry: GeometryId,
    },
    UnsupportedTransform {
        transform: TransformId,
    },
    InvalidTransform {
        primitive: u32,
    },
    UnsupportedInstances,
    InstanceCycle {
        instance: u32,
    },
    UnsupportedAreaLight {
        primitive: u32,
    },
    InvalidAreaLightBinding {
        primitive: u32,
        expected: u32,
        actual: u32,
    },
    InvalidReference {
        resource: &'static str,
        index: u32,
    },
    LimitExceeded {
        resource: &'static str,
        value: u32,
        maximum: u32,
    },
    UnsupportedMaterial {
        primitive: u32,
    },
    UnsupportedAlphaMask {
        primitive: u32,
    },
    UnsupportedTexture {
        texture: u32,
    },
    UnsupportedLight {
        light: u32,
    },
    UnsupportedLightConfiguration,
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPrepareOptions { reason } => {
                write!(formatter, "invalid WebGPU prepare options: {reason}")
            }
            Self::AdapterRequest(message) => write!(formatter, "adapter request failed: {message}"),
            Self::DeviceRequest(message) => write!(formatter, "device request failed: {message}"),
            Self::MissingRayQueryFeature => {
                write!(formatter, "adapter does not support experimental ray query")
            }
            Self::Plan(error) => error.fmt(formatter),
            Self::InvalidRenderRequest(error) => {
                write!(formatter, "invalid render request: {error:?}")
            }
            Self::UnsupportedRenderRequest { reason } => {
                write!(formatter, "unsupported render request: {reason}")
            }
            Self::Readback(message) => write!(formatter, "GPU readback failed: {message}"),
            Self::UnsupportedAccelerationMode(mode) => {
                write!(formatter, "unsupported acceleration mode: {mode:?}")
            }
        }
    }
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyScene => write!(formatter, "scene contains no world primitives"),
            Self::UnsupportedGeometry { geometry } => {
                write!(
                    formatter,
                    "unsupported geometry for hardware ray query: {geometry:?}"
                )
            }
            Self::EmptyGeometry { geometry } => {
                write!(formatter, "empty triangle geometry: {geometry:?}")
            }
            Self::UnsupportedTransform { transform } => {
                write!(
                    formatter,
                    "animated transform is not supported: {transform:?}"
                )
            }
            Self::InvalidTransform { primitive } => {
                write!(formatter, "singular transform for primitive {primitive}")
            }
            Self::UnsupportedInstances => {
                write!(
                    formatter,
                    "instance definitions are not supported by the initial plan"
                )
            }
            Self::InstanceCycle { instance } => {
                write!(formatter, "instance cycle detected at instance {instance}")
            }
            Self::UnsupportedAreaLight { primitive } => {
                write!(
                    formatter,
                    "area lights are unsupported for primitive {primitive}"
                )
            }
            Self::InvalidAreaLightBinding {
                primitive,
                expected,
                actual,
            } => write!(
                formatter,
                "primitive {primitive} has {actual} area lights for {expected} elements"
            ),
            Self::InvalidReference { resource, index } => {
                write!(formatter, "invalid {resource} reference: {index}")
            }
            Self::LimitExceeded {
                resource,
                value,
                maximum,
            } => write!(
                formatter,
                "{resource} value {value} exceeds maximum {maximum}"
            ),
            Self::UnsupportedMaterial { primitive } => {
                write!(formatter, "unsupported material for primitive {primitive}")
            }
            Self::UnsupportedAlphaMask { primitive } => {
                write!(
                    formatter,
                    "alpha masking is not yet supported for primitive {primitive}"
                )
            }
            Self::UnsupportedTexture { texture } => {
                write!(formatter, "unsupported texture {texture}")
            }
            Self::UnsupportedLight { light } => {
                write!(formatter, "unsupported light {light}")
            }
            Self::UnsupportedLightConfiguration => {
                write!(
                    formatter,
                    "the wavefront renderer supports only point, diffuse area, and uniform infinite lights"
                )
            }
        }
    }
}

impl std::error::Error for BackendError {}
