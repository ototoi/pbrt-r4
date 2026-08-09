use crate::interaction::SurfaceInteraction;
use crate::shapes::{Normal3f, Point2f, Point3f, Vector3f};
use crate::util::base::Float;

#[derive(Copy, Clone, Debug, Default)]
pub struct TextureEvalContext {
    pub p: Point3f,
    pub dpdx: Vector3f,
    pub dpdy: Vector3f,
    pub n: Normal3f,
    pub uv: Point2f,
    pub dudx: Float,
    pub dudy: Float,
    pub dvdx: Float,
    pub dvdy: Float,
    pub face_index: u32,
}

impl TextureEvalContext {
    pub fn new(
        p: Point3f,
        dpdx: Vector3f,
        dpdy: Vector3f,
        n: Normal3f,
        uv: Point2f,
        dudx: Float,
        dudy: Float,
        dvdx: Float,
        dvdy: Float,
        face_index: u32,
    ) -> Self {
        Self {
            p,
            dpdx,
            dpdy,
            n,
            uv,
            dudx,
            dudy,
            dvdx,
            dvdy,
            face_index,
        }
    }

    pub fn from_surface_interaction(si: &SurfaceInteraction) -> Self {
        Self::new(
            si.p,
            si.dpdx,
            si.dpdy,
            si.n,
            si.uv,
            si.dudx,
            si.dudy,
            si.dvdx,
            si.dvdy,
            si.face_index,
        )
    }
}

impl From<&SurfaceInteraction> for TextureEvalContext {
    fn from(si: &SurfaceInteraction) -> Self {
        Self::from_surface_interaction(si)
    }
}
