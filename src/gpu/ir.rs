//! Minimal semantic GPU IR used by the initial backend contract.
//!
//! This is intentionally not a device ABI. It contains no `wgpu` handles,
//! raw pointers, shader bindings, or CPU trait objects. Geometry, materials,
//! and textures will be added in later IR phases.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuIrVersion {
    pub major: u16,
    pub minor: u16,
}

pub const CURRENT_IR_VERSION: GpuIrVersion = GpuIrVersion { major: 1, minor: 0 };

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuBounds2i {
    pub min: [i32; 2],
    pub max: [i32; 2],
}

impl GpuBounds2i {
    pub fn area(self) -> Option<u64> {
        let width = i64::from(self.max[0]) - i64::from(self.min[0]);
        let height = i64::from(self.max[1]) - i64::from(self.min[1]);
        (width > 0 && height > 0).then(|| (width as u64) * (height as u64))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRenderConfig {
    pub pixel_bounds: GpuBounds2i,
    pub sample_count: u32,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuSceneData {
    pub render: GpuRenderConfig,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuSceneDraft {
    pub version: GpuIrVersion,
    pub data: GpuSceneData,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuSceneIr {
    version: GpuIrVersion,
    data: GpuSceneData,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuSceneView<'a> {
    pub version: &'a GpuIrVersion,
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

impl GpuSceneIr {
    pub fn view(&self) -> GpuSceneView<'_> {
        GpuSceneView {
            version: &self.version,
            render: &self.data.render,
        }
    }
}
