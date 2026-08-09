use crate::paramdict::*;
use crate::textures::TextureEvalContext;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::transform::*;

#[derive(Clone)]
pub enum TextureMapping2D {
    UV(UVMapping),
    Spherical(SphericalMapping),
    Cylindrical(CylindricalMapping),
    Planar(PlanarMapping),
}

impl TextureMapping2D {
    pub fn map(&self, ctx: &TextureEvalContext) -> (Point2f, Vector2f, Vector2f) {
        match self {
            TextureMapping2D::UV(m) => m.map(ctx),
            TextureMapping2D::Spherical(m) => m.map(ctx),
            TextureMapping2D::Cylindrical(m) => m.map(ctx),
            TextureMapping2D::Planar(m) => m.map(ctx),
        }
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &ParameterDictionary,
    ) -> Result<Self, PbrtError> {
        let mapping = parameters.get_one_string("mapping", "uv");
        match mapping.as_ref() {
            "uv" => {
                let su = parameters.get_one_float("uscale", 1.0);
                let sv = parameters.get_one_float("vscale", 1.0);
                let du = parameters.get_one_float("udelta", 0.0);
                let dv = parameters.get_one_float("vdelta", 0.0);
                Ok(TextureMapping2D::UV(UVMapping::new(su, sv, du, dv)))
            }
            "spherical" => {
                let texture_from_render = render_from_texture.inverse();
                Ok(TextureMapping2D::Spherical(SphericalMapping::new(
                    &texture_from_render,
                )))
            }
            "cylindrical" => {
                let texture_from_render = render_from_texture.inverse();
                Ok(TextureMapping2D::Cylindrical(CylindricalMapping::new(
                    &texture_from_render,
                )))
            }
            "planar" => {
                let v1 = parameters.get_one_vector3f("v1", &Vector3f::new(1.0, 0.0, 0.0));
                let v2 = parameters.get_one_vector3f("v2", &Vector3f::new(0.0, 1.0, 0.0));
                let du = parameters.get_one_float("udelta", 0.0);
                let dv = parameters.get_one_float("vdelta", 0.0);
                let texture_from_render = render_from_texture.inverse();
                Ok(TextureMapping2D::Planar(PlanarMapping::new(
                    &texture_from_render,
                    &v1,
                    &v2,
                    du,
                    dv,
                )))
            }
            _ => {
                log::error!("2D texture mapping \"{}\" unknown", mapping);
                let su = parameters.get_one_float("uscale", 1.0);
                let sv = parameters.get_one_float("vscale", 1.0);
                let du = parameters.get_one_float("udelta", 0.0);
                let dv = parameters.get_one_float("vdelta", 0.0);
                Ok(TextureMapping2D::UV(UVMapping::new(su, sv, du, dv)))
            }
        }
    }
}

#[derive(Clone)]
pub struct UVMapping {
    pub su: Float,
    pub sv: Float,
    pub du: Float,
    pub dv: Float,
}

impl UVMapping {
    pub fn new(su: Float, sv: Float, du: Float, dv: Float) -> Self {
        UVMapping { su, sv, du, dv }
    }

    pub fn map(&self, ctx: &TextureEvalContext) -> (Point2f, Vector2f, Vector2f) {
        // Compute texture differentials for 2D identity mapping
        let dstdx = Vector2f::new(self.su * ctx.dudx, self.sv * ctx.dvdx);
        let dstdy = Vector2f::new(self.su * ctx.dudy, self.sv * ctx.dvdy);
        let st = Point2f::new(self.su * ctx.uv[0] + self.du, self.sv * ctx.uv[1] + self.dv);
        return (st, dstdx, dstdy);
    }
}

#[derive(Clone)]
pub struct SphericalMapping {
    texture_from_render: Transform,
}

impl SphericalMapping {
    pub fn new(texture_from_render: &Transform) -> Self {
        SphericalMapping {
            texture_from_render: *texture_from_render,
        }
    }

    pub fn map(&self, ctx: &TextureEvalContext) -> (Point2f, Vector2f, Vector2f) {
        let pt = self.texture_from_render.transform_point(&ctx.p);
        let x2y2 = pt.x * pt.x + pt.y * pt.y;
        let sqrt_x2y2 = Float::sqrt(x2y2);
        let dsdp = Vector3f::new(-pt.y, pt.x, 0.0) / (2.0 * PI * x2y2);
        let dtdp = Vector3f::new(pt.x * pt.z / sqrt_x2y2, pt.y * pt.z / sqrt_x2y2, -sqrt_x2y2)
            / (PI * (x2y2 + pt.z * pt.z));

        let dpdx = self.texture_from_render.transform_vector(&ctx.dpdx);
        let dpdy = self.texture_from_render.transform_vector(&ctx.dpdy);
        let dstdx = Vector2f::new(Vector3f::dot(&dsdp, &dpdx), Vector3f::dot(&dtdp, &dpdx));
        let dstdy = Vector2f::new(Vector3f::dot(&dsdp, &dpdy), Vector3f::dot(&dtdp, &dpdy));

        let vec = (pt - Point3f::new(0.0, 0.0, 0.0)).normalize();
        let st = Point2f::new(
            spherical_theta(&vec) * INV_PI,
            spherical_phi(&vec) * INV_2_PI,
        );
        (st, dstdx, dstdy)
    }
}

#[derive(Clone)]
pub struct CylindricalMapping {
    texture_from_render: Transform,
}

impl CylindricalMapping {
    pub fn new(texture_from_render: &Transform) -> Self {
        CylindricalMapping {
            texture_from_render: *texture_from_render,
        }
    }

    pub fn map(&self, ctx: &TextureEvalContext) -> (Point2f, Vector2f, Vector2f) {
        let pt = self.texture_from_render.transform_point(&ctx.p);
        let x2y2 = pt.x * pt.x + pt.y * pt.y;
        let dsdp = Vector3f::new(-pt.y, pt.x, 0.0) / (2.0 * PI * x2y2);
        let dtdp = Vector3f::new(0.0, 0.0, 1.0);

        let dpdx = self.texture_from_render.transform_vector(&ctx.dpdx);
        let dpdy = self.texture_from_render.transform_vector(&ctx.dpdy);
        let dstdx = Vector2f::new(Vector3f::dot(&dsdp, &dpdx), Vector3f::dot(&dtdp, &dpdx));
        let dstdy = Vector2f::new(Vector3f::dot(&dsdp, &dpdy), Vector3f::dot(&dtdp, &dpdy));

        let st = Point2f::new((PI + Float::atan2(pt.y, pt.x)) * INV_2_PI, pt.z);
        (st, dstdx, dstdy)
    }
}

#[derive(Clone)]
pub struct PlanarMapping {
    texture_from_render: Transform,
    vs: Vector3f,
    vt: Vector3f,
    ds: Float,
    dt: Float,
}

impl PlanarMapping {
    pub fn new(
        texture_from_render: &Transform,
        vs: &Vector3f,
        vt: &Vector3f,
        ds: Float,
        dt: Float,
    ) -> Self {
        PlanarMapping {
            texture_from_render: *texture_from_render,
            vs: *vs,
            vt: *vt,
            ds,
            dt,
        }
    }

    pub fn map(&self, ctx: &TextureEvalContext) -> (Point2f, Vector2f, Vector2f) {
        let vec = self.texture_from_render.transform_point(&ctx.p);
        let dpdx = self.texture_from_render.transform_vector(&ctx.dpdx);
        let dpdy = self.texture_from_render.transform_vector(&ctx.dpdy);

        let st = Vector2f::new(
            self.ds + Vector3f::dot(&vec, &self.vs),
            self.dt + Vector3f::dot(&vec, &self.vt),
        );

        let dstdx = Vector2f::new(
            Vector3f::dot(&dpdx, &self.vs),
            Vector3f::dot(&dpdx, &self.vt),
        );

        let dstdy = Vector2f::new(
            Vector3f::dot(&dpdy, &self.vs),
            Vector3f::dot(&dpdy, &self.vt),
        );

        return (st, dstdx, dstdy);
    }
}
