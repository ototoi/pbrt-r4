use crate::interaction::SurfaceInteraction;
use crate::paramdict::TextureParameterDictionary;
use crate::textures::{FloatTexture, TextureEvalContext};
use crate::util::base::*;
use crate::util::error::PbrtError;
use crate::util::imageio::{read_raw_image_gamma_correct, RawImage};
use crate::util::sampling::gram_schmidt;
use crate::util::vecmath::Frame;

use std::sync::Arc;

pub struct NormalMap {
    raw: RawImage,
}

impl NormalMap {
    pub fn read(filename: &str) -> Result<Self, PbrtError> {
        let raw = read_raw_image_gamma_correct(filename, false)?;
        if raw.channels < 3 {
            return Err(PbrtError::error(&format!(
                "{}: normal map image must contain R, G, and B channels",
                filename
            )));
        }
        if raw.resolution.x <= 0 || raw.resolution.y <= 0 {
            return Err(PbrtError::error(&format!(
                "{}: normal map image has invalid resolution {:?}",
                filename, raw.resolution
            )));
        }
        Ok(Self { raw })
    }

    fn bilerp_channel(&self, uv: Point2f, channel: usize) -> Float {
        let width = self.raw.resolution.x;
        let height = self.raw.resolution.y;
        let x = uv.x * width as Float - 0.5;
        let y = uv.y * height as Float - 0.5;
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let dx = x - xi as Float;
        let dy = y - yi as Float;

        let v00 = self.get_channel_repeat(xi, yi, channel);
        let v10 = self.get_channel_repeat(xi + 1, yi, channel);
        let v01 = self.get_channel_repeat(xi, yi + 1, channel);
        let v11 = self.get_channel_repeat(xi + 1, yi + 1, channel);
        (1.0 - dx) * (1.0 - dy) * v00
            + dx * (1.0 - dy) * v10
            + (1.0 - dx) * dy * v01
            + dx * dy * v11
    }

    fn get_channel_repeat(&self, x: i32, y: i32, channel: usize) -> Float {
        let width = self.raw.resolution.x;
        let height = self.raw.resolution.y;
        let xx = repeat_index(x, width);
        let yy = repeat_index(y, height);
        let pixel_offset = (yy * width + xx) as usize;
        self.raw.channel(pixel_offset, channel)
    }
}

#[derive(Clone, Copy)]
pub struct NormalBumpShadingContext {
    pub n: Normal3f,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
}

#[derive(Clone, Copy)]
pub struct NormalBumpEvalContext {
    pub p: Point3f,
    pub uv: Point2f,
    pub n: Normal3f,
    pub shading: NormalBumpShadingContext,
    pub dudx: Float,
    pub dudy: Float,
    pub dvdx: Float,
    pub dvdy: Float,
    pub dpdx: Vector3f,
    pub dpdy: Vector3f,
    pub face_index: u32,
}

impl NormalBumpEvalContext {
    pub fn from_surface_interaction(si: &SurfaceInteraction) -> Self {
        Self {
            p: si.p,
            uv: si.uv,
            n: si.n,
            shading: NormalBumpShadingContext {
                n: si.shading.n,
                dpdu: si.shading.dpdu,
                dpdv: si.shading.dpdv,
                dndu: si.shading.dndu,
                dndv: si.shading.dndv,
            },
            dudx: si.dudx,
            dudy: si.dudy,
            dvdx: si.dvdx,
            dvdy: si.dvdy,
            dpdx: si.dpdx,
            dpdy: si.dpdy,
            face_index: si.face_index,
        }
    }

    fn as_texture_eval_context(&self) -> TextureEvalContext {
        TextureEvalContext::new(
            self.p,
            self.dpdx,
            self.dpdy,
            self.n,
            self.uv,
            self.dudx,
            self.dudy,
            self.dvdx,
            self.dvdy,
            self.face_index,
        )
    }
}

pub fn normal_map(
    normal_map: &NormalMap,
    ctx: &NormalBumpEvalContext,
) -> Option<(Vector3f, Vector3f)> {
    let uv = Point2f::new(ctx.uv.x, 1.0 - ctx.uv.y);
    let mut ns = Vector3f::new(
        2.0 * normal_map.bilerp_channel(uv, 0) - 1.0,
        2.0 * normal_map.bilerp_channel(uv, 1) - 1.0,
        2.0 * normal_map.bilerp_channel(uv, 2) - 1.0,
    );
    if ns.length_squared() == 0.0 || ctx.shading.dpdu.length_squared() == 0.0 {
        return None;
    }
    ns = ns.normalize();

    let frame = Frame::from_xz(ctx.shading.dpdu.normalize(), ctx.shading.n);
    let ns = frame.from_local(ns);
    let ulen = ctx.shading.dpdu.length();
    let vlen = ctx.shading.dpdv.length();
    let dpdu_base = gram_schmidt(&ctx.shading.dpdu, &ns);
    if dpdu_base.length_squared() == 0.0 {
        return None;
    }
    let dpdu = dpdu_base.normalize() * ulen;
    let dpdv = Vector3f::cross(&ns, &dpdu).normalize() * vlen;
    Some((dpdu, dpdv))
}

pub fn bump_map(displacement: &FloatTexture, ctx: &NormalBumpEvalContext) -> (Vector3f, Vector3f) {
    let mut shifted_ctx = *ctx;

    let mut du = 0.5 * (Float::abs(ctx.dudx) + Float::abs(ctx.dudy));
    if du == 0.0 {
        du = 0.0005;
    }
    shifted_ctx.p = ctx.p + du * ctx.shading.dpdu;
    shifted_ctx.uv = ctx.uv + Vector2f::new(du, 0.0);
    let u_displace = displacement.evaluate(&shifted_ctx.as_texture_eval_context());

    let mut dv = 0.5 * (Float::abs(ctx.dvdx) + Float::abs(ctx.dvdy));
    if dv == 0.0 {
        dv = 0.0005;
    }
    shifted_ctx.p = ctx.p + dv * ctx.shading.dpdv;
    shifted_ctx.uv = ctx.uv + Vector2f::new(0.0, dv);
    let v_displace = displacement.evaluate(&shifted_ctx.as_texture_eval_context());
    let displace = displacement.evaluate(&ctx.as_texture_eval_context());

    let dpdu = ctx.shading.dpdu
        + (u_displace - displace) / du * ctx.shading.n
        + displace * ctx.shading.dndu;
    let dpdv = ctx.shading.dpdv
        + (v_displace - displace) / dv * ctx.shading.n
        + displace * ctx.shading.dndv;
    (dpdu, dpdv)
}

pub fn get_normal_map(
    mp: &TextureParameterDictionary,
) -> Result<Option<Arc<NormalMap>>, PbrtError> {
    let filename = mp.get_one_filename("normalmap", "");
    if filename.is_empty() {
        return Ok(None);
    }
    Ok(Some(Arc::new(NormalMap::read(&filename)?)))
}

pub fn apply_normal_or_bump(
    normal_map_texture: &Option<Arc<NormalMap>>,
    displacement: &Option<Arc<FloatTexture>>,
    si: &mut SurfaceInteraction,
) {
    let ctx = NormalBumpEvalContext::from_surface_interaction(si);
    if let Some(normal_map_texture) = normal_map_texture {
        if let Some((dpdu, dpdv)) = normal_map(normal_map_texture, &ctx) {
            si.set_shading_geometry_from_tangents(
                &dpdu,
                &dpdv,
                &ctx.shading.dndu,
                &ctx.shading.dndv,
                false,
            );
        }
    } else if let Some(displacement) = displacement {
        let (dpdu, dpdv) = bump_map(displacement, &ctx);
        si.set_shading_geometry_from_tangents(
            &dpdu,
            &dpdv,
            &ctx.shading.dndu,
            &ctx.shading.dndv,
            false,
        );
    }
}

pub fn apply_bump(displacement: &Arc<FloatTexture>, si: &mut SurfaceInteraction) {
    let ctx = NormalBumpEvalContext::from_surface_interaction(si);
    let (dpdu, dpdv) = bump_map(displacement, &ctx);
    si.set_shading_geometry_from_tangents(
        &dpdu,
        &dpdv,
        &ctx.shading.dndu,
        &ctx.shading.dndv,
        false,
    );
}

fn repeat_index(index: i32, size: i32) -> i32 {
    let mut rem = index % size;
    if rem < 0 {
        rem += size;
    }
    rem
}
