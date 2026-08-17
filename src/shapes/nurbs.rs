use crate::base::shape::Shape;
use crate::paramdict::*;

use crate::shapes::*;
use crate::util::base::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

use super::create_triangle_mesh;

pub struct NURBS;

impl NURBS {
    pub fn create(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        params: &ParameterDictionary,
    ) -> Result<Vec<Shape>, PbrtError> {
        create_nurbs(o2w, w2o, reverse_orientation, params)
    }
}

// knot example
// 1 patch
// nu = 4, uorder = 4, uknots = [0, 1, 2, 3, 4, 5, 6, 7]
// uniform : [0, 1, 2, 3, 4, 5, 6, 7]
// bezier : [0, 0, 0, 0, 1, 1, 1, 1]
// endpoint : [0, 0, 0, 0, 1, 1, 1, 1]

// 2 patches
// nu = 7, uorder = 4, uknots = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// uniform : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
// bezier : [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
// endpoint : [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]

// NURBS Evaluation Functions
fn knot_offset(knot: &[Float], order: usize, t: Float) -> usize {
    let first_knot = order - 1;
    assert!(first_knot < knot.len());

    let mut knot_offset = first_knot;
    while t > knot[knot_offset + 1] {
        knot_offset += 1;
    }
    assert!(t >= knot[knot_offset] && t <= knot[knot_offset + 1]);
    return knot_offset;
}

// doesn't handle flat out discontinuities in the curve...

#[derive(Debug, Default, Copy, Clone)]
struct Homogeneous3 {
    x: Float,
    y: Float,
    z: Float,
    w: Float,
}
impl Homogeneous3 {
    pub fn new(x: Float, y: Float, z: Float, w: Float) -> Self {
        Self { x, y, z, w }
    }
}

struct OffsetArray<'a, T> {
    pub offset: i32,
    pub array: &'a [T],
}
impl<'a, T> OffsetArray<'a, T> {
    pub fn new(offset: i32, array: &'a [T]) -> Self {
        Self { offset, array }
    }
}
impl<'a, T> std::ops::Index<i32> for OffsetArray<'a, T> {
    type Output = T;
    fn index(&self, index: i32) -> &Self::Output {
        let i = self.offset as i32 + index;
        assert!(i >= 0);
        let i = i as usize;
        &self.array[i]
    }
}
impl<'a, T> OffsetArray<'a, T> {
    fn len(&self) -> usize {
        (self.array.len() as i32 - self.offset) as usize
    }
}

fn nurbs_evaluate(
    order: i32,
    knot: &[Float],
    cp: &OffsetArray<Homogeneous3>,
    cp_stride: i32,
    t: Float,
) -> (Homogeneous3, Vector3f) {
    let np = cp.len() as i32;
    let knot_offset = knot_offset(knot, order as usize, t) as i32;
    let knot = OffsetArray::new(knot_offset as i32, knot);

    assert!(knot_offset >= order - 1);
    let cp_offset = knot_offset - order + 1;
    assert!(cp_offset < np);

    let mut cp_work = vec![Homogeneous3::default(); order as usize];
    for i in 0..order {
        let k = (cp_offset + i) * cp_stride;
        cp_work[i as usize] = cp[k];
    }

    let order = order as i32;
    for i in 0..order - 2 {
        for j in 0..order - 1 - i {
            let alpha = (knot[1 + j] - t) / (knot[1 + j] - knot[j + 2 - order + i]);
            assert!(alpha >= 0.0 && alpha <= 1.0);
            let j = j as usize;
            cp_work[j].x = alpha * cp_work[j].x + (1.0 - alpha) * cp_work[j + 1].x;
            cp_work[j].y = alpha * cp_work[j].y + (1.0 - alpha) * cp_work[j + 1].y;
            cp_work[j].z = alpha * cp_work[j].z + (1.0 - alpha) * cp_work[j + 1].z;
            cp_work[j].w = alpha * cp_work[j].w + (1.0 - alpha) * cp_work[j + 1].w;
        }
    }

    let alpha = (knot[1] - t) / (knot[1] - knot[0]);
    assert!(alpha >= 0.0 && alpha <= 1.0);
    let x = alpha * cp_work[0].x + (1.0 - alpha) * cp_work[1].x;
    let y = alpha * cp_work[0].y + (1.0 - alpha) * cp_work[1].y;
    let z = alpha * cp_work[0].z + (1.0 - alpha) * cp_work[1].z;
    let w = alpha * cp_work[0].w + (1.0 - alpha) * cp_work[1].w;
    let val = Homogeneous3::new(x, y, z, w);

    let factor = (order - 1) as Float / (knot[1] - knot[0]);
    let dx = factor * (cp_work[1].x - cp_work[0].x);
    let dy = factor * (cp_work[1].y - cp_work[0].y);
    let dz = factor * (cp_work[1].z - cp_work[0].z);
    let dw = factor * (cp_work[1].w - cp_work[0].w);

    let dx = (dx / val.w) - (val.x * dw / (val.w * val.w));
    let dy = (dy / val.w) - (val.y * dw / (val.w * val.w));
    let dz = (dz / val.w) - (val.z * dw / (val.w * val.w));
    let deriv = Vector3f::new(dx, dy, dz);

    return (val, deriv);
}

fn nurbs_evaluate_surface(
    u_order: i32,
    u_knot: &[Float],
    u_cp: i32,
    u: Float,
    v_order: i32,
    v_knot: &[Float],
    v_cp: i32,
    v: Float,
    cp: &[Homogeneous3],
) -> (Point3f, Vector3f, Vector3f) {
    let mut iso = vec![Homogeneous3::default(); usize::max(u_order as usize, v_order as usize)];

    let u_offset = knot_offset(u_knot, u_order as usize, u) as i32;
    assert!(u_offset >= u_order - 1);
    let u_first_cp = u_offset - u_order + 1;
    for i in 0..u_order {
        let offset_cp = OffsetArray::new((u_first_cp + i) as i32, cp);
        iso[i as usize] = nurbs_evaluate(v_order, v_knot, &offset_cp, u_cp, v).0;
    }

    let v_offset = knot_offset(v_knot, v_order as usize, v) as i32;
    assert!(v_offset >= v_order - 1);
    let v_first_cp = v_offset - v_order + 1;
    assert!(v_first_cp < v_cp);

    let offset_cp = OffsetArray::new(-u_first_cp, &iso);
    let (p, dpdu) = nurbs_evaluate(u_order, u_knot, &offset_cp, 1, u);
    for i in 0..v_order {
        let offset_cp = OffsetArray::new(((v_first_cp + i) * u_cp) as i32, cp);
        iso[i as usize] = nurbs_evaluate(u_order, u_knot, &offset_cp, 1, u).0;
    }

    let offset_cp = OffsetArray::new(-v_first_cp, &iso);
    let (_p, dpdv) = nurbs_evaluate(v_order, v_knot, &offset_cp, 1, v);

    return (Point3f::new(p.x / p.w, p.y / p.w, p.z / p.w), dpdu, dpdv);
}

fn create_tesselated_mesh(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    nu: usize,
    uorder: usize,
    uknots: &[Float],
    urange: (Float, Float),
    diceu: usize,
    nv: usize,
    vorder: usize,
    vknots: &[Float],
    vrange: (Float, Float),
    dicev: usize,
    pw: &[Homogeneous3],
) -> Result<Vec<Shape>, PbrtError> {
    let u0 = urange.0;
    let u1 = urange.1;
    let v0 = vrange.0;
    let v1 = vrange.1;

    let mut ueval = vec![0.0; diceu];
    let mut veval = vec![0.0; dicev];
    let mut eval_ps = vec![Point3f::default(); diceu * dicev];
    let mut eval_ns = vec![Normal3f::default(); diceu * dicev];
    let mut uvs = vec![Point2f::default(); diceu * dicev];
    for i in 0..diceu {
        ueval[i] = lerp(i as Float / (diceu - 1) as Float, u0, u1);
    }
    for i in 0..dicev {
        veval[i] = lerp(i as Float / (dicev - 1) as Float, v0, v1);
    }

    let nu = nu as i32;
    let nv = nv as i32;
    let uorder = uorder as i32;
    let vorder = vorder as i32;
    for v in 0..dicev {
        for u in 0..diceu {
            let uu = ueval[u];
            let vv = veval[v];
            uvs[v * diceu + u] = Point2f::new(uu, vv);
            let (p, dpdu, dpdv) =
                nurbs_evaluate_surface(uorder, &uknots, nu, uu, vorder, &vknots, nv, vv, &pw);

            eval_ps[v * diceu + u] = p;
            eval_ns[v * diceu + u] = Vector3f::cross(&dpdu, &dpdv).normalize();
        }
    }

    // Generate points-polygons mesh
    let ntris = 2 * (diceu - 1) * (dicev - 1);
    let mut vertex_indices = vec![0; 3 * ntris];
    let mut index = 0;
    let vn = |u: u32, v: u32| -> u32 { v * diceu as u32 + u };
    for v in 0..dicev - 1 {
        for u in 0..diceu - 1 {
            let u = u as u32;
            let v = v as u32;
            vertex_indices[index] = vn(u, v);
            vertex_indices[index + 1] = vn(u + 1, v);
            vertex_indices[index + 2] = vn(u + 1, v + 1);
            index += 3;

            vertex_indices[index] = vn(u, v);
            vertex_indices[index + 1] = vn(u + 1, v + 1);
            vertex_indices[index + 2] = vn(u, v + 1);
            index += 3;
        }
    }
    let params = ParameterDictionary::new();
    let mesh = create_triangle_mesh(
        o2w,
        w2o,
        reverse_orientation,
        vertex_indices,
        eval_ps,
        Vec::new(),
        eval_ns,
        uvs,
        &params,
    )?;
    let mesh: Vec<Shape> = mesh.into_iter().map(Shape::Triangle).collect();
    return Ok(mesh);
}

fn get_points(params: &ParameterDictionary) -> Option<(Vec<Float>, bool)> {
    let p = params.get_points("P");
    if !p.is_empty() {
        return Some((p, false));
    }
    let p = params.get_points("Pw");
    if !p.is_empty() {
        return Some((p, true));
    }
    return None;
}

pub fn create_nurbs(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    params: &ParameterDictionary,
) -> Result<Vec<Shape>, PbrtError> {
    let nu = params.get_one_int("nu", -1);
    if nu == -1 {
        return Err(PbrtError::error(
            "Must provide number of control points \"nu\" with NURBS shape.",
        ));
    }
    let uorder = params.get_one_int("uorder", -1);
    if uorder == -1 {
        return Err(PbrtError::error(
            "Must provide u order \"uorder\" with NURBS shape.",
        ));
    }
    let uknots = params.get_floats("uknots");
    if uknots.is_empty() {
        return Err(PbrtError::error(
            "Must provide u knot vector \"uknots\" with NURBS shape.",
        ));
    }
    let nu = nu as usize;
    let uorder = uorder as usize;
    let nuknots = uknots.len();
    if nuknots != nu + uorder {
        let msg = format!(
            "Number of knots in u knot vector {} doesn't match sum of number of u control points {} and u order {}.",
            nuknots, nu, uorder
        );
        return Err(PbrtError::error(&msg));
    }

    let nv = params.get_one_int("nv", -1);
    if nv == -1 {
        return Err(PbrtError::error(
            "Must provide number of control points \"nv\" with NURBS shape.",
        ));
    }
    let vorder = params.get_one_int("vorder", -1);
    if vorder == -1 {
        return Err(PbrtError::error(
            "Must provide v order \"vorder\" with NURBS shape.",
        ));
    }
    let vknots = params.get_floats("vknots");
    if vknots.is_empty() {
        return Err(PbrtError::error(
            "Must provide v knot vector \"vknots\" with NURBS shape.",
        ));
    }
    let nv = nv as usize;
    let vorder = vorder as usize;
    let nvknots = vknots.len();
    if nvknots != nv + vorder {
        let msg = format!(
            "Number of knots in v knot vector {} doesn't match sum of number of v control points {} and v order {}.",
            nvknots, nv, vorder
        );
        return Err(PbrtError::error(&msg));
    }

    let (p, is_homogeneous) = get_points(params).ok_or(PbrtError::error(
        "Must provide control points via \"P\" or \"Pw\" parameter to NURBS shape.",
    ))?;
    let mut npts = p.len();
    if !is_homogeneous && npts % 3 == 0 {
        npts /= 3;
    } else if is_homogeneous && npts % 4 == 0 {
        npts /= 4;
    } else {
        return Err(PbrtError::error(
            "Number of control points must be multiple of 3 or 4.",
        ));
    }

    if npts != nu * nv {
        let msg = format!(
            "Number of control points {} doesn't match nu * nv = {} * {} = {}.",
            npts,
            nu,
            nv,
            nu * nv
        );
        return Err(PbrtError::error(&msg));
    }

    let mut pw = vec![Homogeneous3::default(); npts];
    if is_homogeneous {
        for i in 0..npts {
            pw[i] = Homogeneous3::new(p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3]);
        }
    } else {
        for i in 0..npts {
            pw[i] = Homogeneous3::new(p[3 * i], p[3 * i + 1], p[3 * i + 2], 1.0);
        }
    }

    let u0x = uknots[uorder - 1]; //[0, 1, 2, 3, 4, 5, 6, 7] -> 3
    let u1x = uknots[nu]; //[0, 1, 2, 3, 4, 5, 6, 7] -> 4
    let v0x = vknots[vorder - 1];
    let v1x = vknots[nv];
    let u0 = params.get_one_float("u0", u0x); //
    let u1 = params.get_one_float("u1", u1x); //
    let v0 = params.get_one_float("v0", v0x); //
    let v1 = params.get_one_float("v1", v1x); //

    let u0 = Float::clamp(u0, u0x, u1x);
    let u1 = Float::clamp(u1, u0x, u1x);
    let v0 = Float::clamp(v0, v0x, v1x);
    let v1 = Float::clamp(v1, v0x, v1x);

    let diceu = params.get_one_int("diceu", 30);
    let dicev = params.get_one_int("dicev", 30);

    let diceu = diceu.max(2) as usize;
    let dicev = dicev.max(2) as usize;

    let urange = (u0, u1);
    let vrange = (v0, v1);

    return create_tesselated_mesh(
        o2w,
        w2o,
        reverse_orientation,
        nu,
        uorder,
        &uknots,
        urange,
        diceu,
        nv,
        vorder,
        &vknots,
        vrange,
        dicev,
        &pw,
    );
}
