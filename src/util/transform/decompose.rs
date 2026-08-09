use super::matrix4x4::*;
use crate::util::base::*;
use crate::util::quaternion::*;

fn invertible(m: &Matrix4x4) -> bool {
    m.inverse().is_some()
}

fn suppress_for_scale(m: Matrix4x4) -> Matrix4x4 {
    let mut mm = Matrix4x4::identity();
    for i in 0..3 {
        mm.m[4 * i + i] = m.m[4 * i + i];
    }
    return mm;
}

// Decompose a 4x4 matrix into translation, rotation, and scale.
pub fn decompose(
    m: &Matrix4x4,
    epsilon: Float,
    max_count: i32,
) -> Option<(Vector3f, Quaternion, Vector3f)> {
    // Extract translation _T_ from transformation matrix
    let t = Vector3f::new(m.m[4 * 0 + 3], m.m[4 * 1 + 3], m.m[4 * 2 + 3]);

    // Compute new transformation matrix _M_ without translation
    let mut mm = *m;
    for i in 0..3 {
        mm.m[4 * i + 3] = 0.0;
    }
    mm.m[15] = 1.0;

    // Extract rotation _R_ from transformation matrix
    let mut r = mm;
    let mut count = 0;
    let mut norm: Float = 0.0;
    loop {
        if !invertible(&r) {
            return None;
        }
        if let Some(r_it) = r.transpose().inverse() {
            // Compute next matrix _Rnext_ in series
            let mut r_next = r;
            for i in 0..4 {
                for j in 0..4 {
                    r_next.m[4 * i + j] = lerp(0.5, r.m[4 * i + j], r_it.m[4 * i + j]);
                }
            }

            if !invertible(&r_next) {
                return None;
            }
            if let Some(ir_next) = r_next.inverse() {
                let s = suppress_for_scale(ir_next * mm);
                if let Some(is) = s.inverse() {
                    r_next = mm * is;
                } else {
                    return None;
                }
            } else {
                return None;
            }

            // Compute norm of difference between _R_ and _Rnext_
            for i in 0..3 {
                let n = Float::abs(r.m[4 * i + 0] - r_next.m[4 * i + 0])
                    + Float::abs(r.m[4 * i + 1] - r_next.m[4 * i + 1])
                    + Float::abs(r.m[4 * i + 2] - r_next.m[4 * i + 2]);
                norm = Float::max(norm, n);
            }
            r = r_next;
        } else {
            return None;
        }

        count += 1;
        if !(count < max_count && norm > epsilon) {
            break;
        }
    }

    if let Some(ir) = r.inverse() {
        let s = suppress_for_scale(ir * mm);
        if let Some(is) = s.inverse() {
            r = mm * is;
        }
        let q = Quaternion::from(r);
        let q = q.normalize();
        let s = Vector3f::new(s.m[0], s.m[5], s.m[10]);
        return Some((t, q, s));
    }
    return None;
}
