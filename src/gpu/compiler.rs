//! Host-side construction boundary for GPU IR.

use super::ir::{
    GeometryId, GpuBounds2i, GpuFloat, GpuGeometry, GpuIndex, GpuIrValidationErrors, GpuMatrix4x4,
    GpuNormal3, GpuPoint2, GpuPoint3, GpuPrimitive, GpuRenderConfig, GpuSceneData, GpuSceneDraft,
    GpuSceneIr, GpuSceneView, GpuStaticTransform, GpuTransform, GpuTriangleMesh, GpuVector3,
    TransformId, CURRENT_IR_VERSION,
};
use crate::paramdict::ParameterDictionary;
use crate::parser::scene_builder::{RenderFromObject, SceneBuilder, ShapeSceneEntity};
use crate::util::base::Float;
use crate::util::transform::{Matrix4x4, Transform};
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuSourceLocation {
    pub filename: String,
    pub line: u32,
    pub column: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuCompileError {
    UnsupportedShape {
        name: String,
        source: GpuSourceLocation,
    },
    UnsupportedSceneFeature {
        feature: &'static str,
        source: GpuSourceLocation,
    },
    MissingParameter {
        parameter: &'static str,
        source: GpuSourceLocation,
    },
    InvalidParameter {
        parameter: &'static str,
        detail: String,
        source: GpuSourceLocation,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuSceneBuildError {
    Compile(GpuCompileError),
    Validation(GpuIrValidationErrors),
}

impl From<GpuCompileError> for GpuSceneBuildError {
    fn from(error: GpuCompileError) -> Self {
        Self::Compile(error)
    }
}

impl From<GpuIrValidationErrors> for GpuSceneBuildError {
    fn from(errors: GpuIrValidationErrors) -> Self {
        Self::Validation(errors)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuCompiledScene {
    ir: Arc<GpuSceneIr>,
}

impl GpuCompiledScene {
    pub fn new(ir: GpuSceneIr) -> Self {
        Self { ir: Arc::new(ir) }
    }

    pub fn scene(&self) -> &GpuSceneIr {
        &self.ir
    }

    pub fn view(&self) -> GpuSceneView<'_> {
        self.ir.view()
    }
}

impl SceneBuilder {
    /// Compiles the currently supported subset of scene entities into the
    /// backend-independent GPU IR. Unsupported semantics are reported rather
    /// than silently omitted or delegated to the CPU renderer.
    pub fn build_gpu_ir(&self) -> Result<GpuCompiledScene, GpuSceneBuildError> {
        if !self.instance_definitions.is_empty() || !self.instance_uses.is_empty() {
            return Err(GpuSceneBuildError::Compile(
                GpuCompileError::UnsupportedSceneFeature {
                    feature: "instances",
                    source: empty_source(),
                },
            ));
        }
        let mut transforms = Vec::new();
        let mut geometry = Vec::new();
        let mut primitives = Vec::new();

        for shape in &self.shapes {
            compile_shape(shape, &mut transforms, &mut geometry, &mut primitives)?;
        }
        if !self.animated_shapes.is_empty() {
            let shape = &self.animated_shapes[0];
            return Err(GpuSceneBuildError::Compile(unsupported_feature(
                shape,
                "animated transforms",
            )));
        }

        let render = render_config(self)?;
        let draft = GpuSceneDraft {
            version: CURRENT_IR_VERSION,
            data: GpuSceneData {
                transforms,
                geometry,
                primitives,
                render,
            },
        };
        let ir = draft.finish()?;
        Ok(GpuCompiledScene::new(ir))
    }
}

fn compile_shape(
    shape: &ShapeSceneEntity,
    transforms: &mut Vec<GpuTransform>,
    geometry: &mut Vec<GpuGeometry>,
    primitives: &mut Vec<GpuPrimitive>,
) -> Result<(), GpuCompileError> {
    let source = source_location(shape);
    if shape.base.name != "trianglemesh" {
        return Err(GpuCompileError::UnsupportedShape {
            name: shape.base.name.clone(),
            source,
        });
    }
    if !shape.child_params.is_empty() {
        return Err(unsupported_feature(shape, "grouped child shapes"));
    }
    if shape.area_light_index.is_some()
        || !shape.medium_interface.is_empty()
        || shape.material_name.is_some()
        || shape.material_index != usize::MAX
        || !shape.material_is_default
    {
        return Err(unsupported_feature(
            shape,
            "material, area light, or medium binding",
        ));
    }
    if shape.instance_name.is_some() {
        return Err(unsupported_feature(shape, "instances"));
    }
    let RenderFromObject::Static(transform) = &shape.render_from_object else {
        return Err(unsupported_feature(shape, "animated transforms"));
    };

    let transform_id = TransformId(transforms.len() as GpuIndex);
    transforms.push(GpuTransform::Static(static_transform(transform, &source)?));
    let mesh = triangle_mesh(&shape.base.params, &source)?;
    let geometry_id = GeometryId(geometry.len() as GpuIndex);
    geometry.push(GpuGeometry::TriangleMesh(mesh));
    primitives.push(GpuPrimitive {
        geometry: geometry_id,
        transform: transform_id,
        reverse_orientation: shape.reverse_orientation,
    });
    Ok(())
}

fn triangle_mesh(
    params: &ParameterDictionary,
    source: &GpuSourceLocation,
) -> Result<GpuTriangleMesh, GpuCompileError> {
    let positions = params.get_points("P");
    if positions.is_empty() {
        return Err(GpuCompileError::MissingParameter {
            parameter: "P",
            source: source.clone(),
        });
    }
    if positions.len() % 3 != 0 {
        return Err(invalid_parameter(
            "P",
            "position count is not divisible by 3",
            source,
        ));
    }
    let positions = positions
        .chunks_exact(3)
        .map(|p| {
            Ok(GpuPoint3([
                to_gpu_float(p[0], source)?,
                to_gpu_float(p[1], source)?,
                to_gpu_float(p[2], source)?,
            ]))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let raw_indices = params.get_ints("indices");
    if raw_indices.is_empty() {
        return Err(GpuCompileError::MissingParameter {
            parameter: "indices",
            source: source.clone(),
        });
    }
    if raw_indices.len() % 3 != 0 {
        return Err(invalid_parameter(
            "indices",
            "index count is not divisible by 3",
            source,
        ));
    }
    let mut indices = Vec::with_capacity(raw_indices.len() / 3);
    for triangle in raw_indices.chunks_exact(3) {
        let mut converted = [0; 3];
        for (dst, index) in converted.iter_mut().zip(triangle) {
            *dst = GpuIndex::try_from(*index)
                .map_err(|_| invalid_parameter("indices", "negative index", source))?;
        }
        indices.push(converted);
    }

    let normals = optional_vec3(params.get_points("N"), "N", source)?;
    let tangents = optional_vec3(params.get_points("S"), "S", source)?.map(|values| {
        values
            .into_iter()
            .map(|normal| GpuVector3(normal.0))
            .collect()
    });
    let uvs = optional_vec2(params.get_points("uv"), "uv", source)?;
    Ok(GpuTriangleMesh {
        positions,
        indices,
        normals,
        tangents,
        uvs,
        face_indices: None,
    })
}

fn optional_vec2(
    values: Vec<Float>,
    parameter: &'static str,
    source: &GpuSourceLocation,
) -> Result<Option<Vec<GpuPoint2>>, GpuCompileError> {
    if values.is_empty() {
        return Ok(None);
    }
    if values.len() % 2 != 0 {
        return Err(invalid_parameter(
            parameter,
            "value count is not divisible by 2",
            source,
        ));
    }
    values
        .chunks_exact(2)
        .map(|value| {
            Ok(GpuPoint2([
                to_gpu_float(value[0], source)?,
                to_gpu_float(value[1], source)?,
            ]))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn optional_vec3(
    values: Vec<Float>,
    parameter: &'static str,
    source: &GpuSourceLocation,
) -> Result<Option<Vec<GpuNormal3>>, GpuCompileError> {
    if values.is_empty() {
        return Ok(None);
    }
    if values.len() % 3 != 0 {
        return Err(invalid_parameter(
            parameter,
            "value count is not divisible by 3",
            source,
        ));
    }
    values
        .chunks_exact(3)
        .map(|value| {
            Ok(GpuNormal3([
                to_gpu_float(value[0], source)?,
                to_gpu_float(value[1], source)?,
                to_gpu_float(value[2], source)?,
            ]))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn static_transform(
    transform: &Transform,
    source: &GpuSourceLocation,
) -> Result<GpuStaticTransform, GpuCompileError> {
    Ok(GpuStaticTransform {
        render_from_object: matrix(transform.m, source)?,
        object_from_render: matrix(transform.minv, source)?,
        swaps_handedness: transform.swaps_handedness(),
    })
}

fn matrix(matrix: Matrix4x4, source: &GpuSourceLocation) -> Result<GpuMatrix4x4, GpuCompileError> {
    let mut result = [[0.0; 4]; 4];
    for (row, values) in result.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = to_gpu_float(matrix.m[row * 4 + column], source)?;
        }
    }
    Ok(GpuMatrix4x4(result))
}

fn render_config(builder: &SceneBuilder) -> Result<GpuRenderConfig, GpuCompileError> {
    let xresolution = builder.film_params.get_one_int("xresolution", 1280);
    let yresolution = builder.film_params.get_one_int("yresolution", 720);
    if xresolution <= 0 || yresolution <= 0 {
        return Err(invalid_parameter(
            "xresolution/yresolution",
            "resolution must be positive",
            &empty_source(),
        ));
    }

    let mut min = [0_i32, 0_i32];
    let mut max = [xresolution, yresolution];
    if let Some(pixelbounds) = builder.film_params.get_ints_ref("pixelbounds") {
        if pixelbounds.len() != 4 {
            return Err(invalid_parameter(
                "pixelbounds",
                "value count must be 4",
                &empty_source(),
            ));
        }
        min = [pixelbounds[0], pixelbounds[2]];
        max = [pixelbounds[1], pixelbounds[3]];
        min[0] = min[0].max(0);
        min[1] = min[1].max(0);
        max[0] = max[0].min(xresolution);
        max[1] = max[1].min(yresolution);
    }
    if let Some(cropwindow) = builder.film_params.get_floats_ref("cropwindow") {
        if cropwindow.len() != 4 {
            return Err(invalid_parameter(
                "cropwindow",
                "value count must be 4",
                &empty_source(),
            ));
        }
        let x0 = cropwindow[0].min(cropwindow[1]).clamp(0.0, 1.0);
        let y0 = cropwindow[2].min(cropwindow[3]).clamp(0.0, 1.0);
        let x1 = cropwindow[0].max(cropwindow[1]).clamp(0.0, 1.0);
        let y1 = cropwindow[2].max(cropwindow[3]).clamp(0.0, 1.0);
        min = [
            (xresolution as Float * x0).ceil() as i32,
            (yresolution as Float * y0).ceil() as i32,
        ];
        max = [
            (xresolution as Float * x1).ceil() as i32,
            (yresolution as Float * y1).ceil() as i32,
        ];
    }
    if max[0] <= min[0] || max[1] <= min[1] {
        return Err(invalid_parameter(
            "pixelbounds/cropwindow",
            "pixel bounds must have positive area",
            &empty_source(),
        ));
    }
    let sample_default = if builder.sampler_name == "independent" {
        4
    } else {
        16
    };
    let sample_count = builder
        .sampler_params
        .get_one_int("pixelsamples", sample_default);
    if sample_count <= 0 {
        return Err(invalid_parameter(
            "pixelsamples",
            "sample count must be positive",
            &empty_source(),
        ));
    }
    Ok(GpuRenderConfig {
        pixel_bounds: GpuBounds2i {
            min: [
                u32::try_from(min[0]).unwrap_or(0),
                u32::try_from(min[1]).unwrap_or(0),
            ],
            max: [
                u32::try_from(max[0]).unwrap_or(0),
                u32::try_from(max[1]).unwrap_or(0),
            ],
        },
        sample_count: u32::try_from(sample_count).map_err(|_| {
            invalid_parameter(
                "pixelsamples",
                "sample count does not fit u32",
                &empty_source(),
            )
        })?,
    })
}

fn empty_source() -> GpuSourceLocation {
    GpuSourceLocation {
        filename: String::new(),
        line: 0,
        column: 0,
    }
}

fn to_gpu_float(value: Float, source: &GpuSourceLocation) -> Result<GpuFloat, GpuCompileError> {
    let value = value as f32;
    value.is_finite().then_some(value).ok_or_else(|| {
        invalid_parameter(
            "numeric value",
            "value cannot be represented as finite f32",
            source,
        )
    })
}

fn source_location(shape: &ShapeSceneEntity) -> GpuSourceLocation {
    GpuSourceLocation {
        filename: shape.base.loc.filename.clone(),
        line: shape.base.loc.line,
        column: shape.base.loc.column,
    }
}

fn unsupported_feature(shape: &ShapeSceneEntity, feature: &'static str) -> GpuCompileError {
    GpuCompileError::UnsupportedSceneFeature {
        feature,
        source: source_location(shape),
    }
}

fn invalid_parameter(
    parameter: &'static str,
    detail: &str,
    source: &GpuSourceLocation,
) -> GpuCompileError {
    GpuCompileError::InvalidParameter {
        parameter,
        detail: detail.to_owned(),
        source: source.clone(),
    }
}
