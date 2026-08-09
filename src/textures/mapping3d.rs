use crate::paramdict::ParameterDictionary;
use crate::textures::TextureEvalContext;
use crate::util::base::*;
use crate::util::transform::*;

// Enum-based TextureMapping3D
#[derive(Clone)]
pub enum TextureMapping3D {
    PointTransform(PointTransformMapping),
}

impl TextureMapping3D {
    pub fn map(&self, ctx: &TextureEvalContext) -> (Point3f, Vector3f, Vector3f) {
        match self {
            TextureMapping3D::PointTransform(m) => m.map(ctx),
        }
    }

    pub fn create(_parameters: &ParameterDictionary, render_from_texture: &Transform) -> Self {
        let texture_from_render = render_from_texture.inverse();
        TextureMapping3D::PointTransform(PointTransformMapping::new(&texture_from_render))
    }
}

#[derive(Clone)]
pub struct PointTransformMapping {
    texture_from_render: Transform,
}

impl PointTransformMapping {
    pub fn new(texture_from_render: &Transform) -> Self {
        PointTransformMapping {
            texture_from_render: *texture_from_render,
        }
    }

    pub fn map(&self, ctx: &TextureEvalContext) -> (Point3f, Vector3f, Vector3f) {
        let p = self.texture_from_render.transform_point(&ctx.p);
        let dpdx = self.texture_from_render.transform_vector(&ctx.dpdx);
        let dpdy = self.texture_from_render.transform_vector(&ctx.dpdy);
        return (p, dpdx, dpdy);
    }
}
