use crate::interaction::SurfaceInteraction;
use crate::shapes::{Normal3f, Vector3f};
use crate::textures::TextureEvalContext;

#[derive(Copy, Clone, Debug, Default)]
pub struct MaterialEvalContext {
    pub texture_ctx: TextureEvalContext,
    pub wo: Vector3f,
    pub ns: Normal3f,
    pub dpdus: Vector3f,
}

impl MaterialEvalContext {
    pub fn new(
        texture_ctx: TextureEvalContext,
        wo: Vector3f,
        ns: Normal3f,
        dpdus: Vector3f,
    ) -> Self {
        Self {
            texture_ctx,
            wo,
            ns,
            dpdus,
        }
    }

    pub fn from_surface_interaction(si: &SurfaceInteraction) -> Self {
        Self::new(
            TextureEvalContext::from(si),
            si.wo,
            si.shading.n,
            si.shading.dpdu,
        )
    }

    pub fn texture_context(&self) -> &TextureEvalContext {
        &self.texture_ctx
    }
}

impl From<&SurfaceInteraction> for MaterialEvalContext {
    fn from(si: &SurfaceInteraction) -> Self {
        Self::from_surface_interaction(si)
    }
}

impl From<MaterialEvalContext> for TextureEvalContext {
    fn from(ctx: MaterialEvalContext) -> Self {
        ctx.texture_ctx
    }
}
